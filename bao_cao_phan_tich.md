# BÁO CÁO PHÂN TÍCH CHI TIẾT NOTEBOOK
# MobileNetV2 Phân Loại Biển Báo Giao Thông GTSRB

---

## PHẦN I: SETUP, DATASET & TIỀN XỬ LÝ

---

### CELL 1: SETUP & CÀI ĐẶT

```python
import os, sys, time, json, csv, random, math, warnings
from google.colab import drive
drive.mount('/content/drive')
```

**Phân tích:**

| Thư viện | Vai trò |
|:---|:---|
| `os, sys` | Thao tác file system, tạo thư mục checkpoint |
| `time` | Đo thời gian mỗi epoch |
| `json, csv` | Lưu training log dạng JSON và CSV |
| `random, math` | Random seed, tính cosine LR schedule |
| `warnings` | Tắt warning không cần thiết |

**Mount Google Drive:** Kết nối Colab với Drive để lưu checkpoint. Khi Colab bị ngắt (timeout 90 phút), checkpoint trên Drive không mất → có thể resume training.

**Tại sao cần?** Colab miễn phí bị giới hạn thời gian. Không lưu Drive = mất toàn bộ training progress khi disconnect.

---

### CELL 2: TẢI DATASET GTSRB

```python
raw_train = torchvision.datasets.GTSRB(root=DATA_DIR, split='train', download=True)
raw_test = torchvision.datasets.GTSRB(root=DATA_DIR, split='test', download=True)
```

**Phân tích dataset GTSRB:**

| Thuộc tính | Chi tiết |
|:---|:---|
| Tên đầy đủ | German Traffic Sign Recognition Benchmark |
| Nguồn gốc | Institut für Neuroinformatik, Đức (2011) |
| Số lớp | 43 loại biển báo giao thông |
| Train | 39,209 ảnh |
| Test | 12,630 ảnh |
| Kích thước ảnh gốc | 15×15 đến 250×250 pixels (không đồng nhất) |
| Định dạng | PPM (Portable Pixmap) |

**Cách thu thập:** Camera gắn trên xe chạy trên đường phố Đức. Mỗi biển báo được quay thành video ~30 frames, sau đó crop bounding box xung quanh biển.

**Đặc điểm quan trọng:**
- Ảnh từ video → frames liên tiếp rất giống nhau
- Điều kiện đa dạng: nắng, mưa, bóng đổ, ban đêm
- Class imbalance: class ít nhất ~210 ảnh, nhiều nhất ~2250 ảnh

**Tại sao chọn GTSRB?** Dataset chuẩn học thuật phổ biến nhất cho bài toán phân loại biển báo. Được sử dụng trong hàng nghìn paper, kết quả có thể so sánh trực tiếp.

---

### CELL 3: CẤU HÌNH HYPERPARAMETERS

```python
CONFIG = {
    'img_size': 96,
    'batch_size': 128,
    'epochs': 50,
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'num_classes': 43,
    'width_mult': 1.0,
    'warmup_epochs': 5,
    'label_smoothing': 0.1,
    'dropout': 0.2,
    'grad_clip': 5.0,
    'patience': 10,
    'save_every': 10,
    'seed': 42,
    'resume': False,
}
```

**Phân tích từng hyperparameter:**

#### img_size = 96
- Ảnh GTSRB gốc rất nhỏ (nhiều ảnh chỉ 30×30). Resize lên 96×96 giữ đủ chi tiết mà không quá tốn VRAM
- So sánh: 48×48 nhanh nhưng mất chi tiết; 224×224 chậm gấp 5× mà accuracy không tăng vì ảnh gốc nhỏ bị mờ khi phóng to

#### batch_size = 128
- Số ảnh xử lý cùng lúc trong 1 iteration
- Lớn hơn → gradient ổn định hơn, tận dụng GPU song song
- Quá lớn → hết VRAM, generalize kém hơn
- 128 là cân bằng cho GPU T4 (16GB VRAM) với input 96×96

#### lr = 0.01 (Learning Rate)
- Tốc độ cập nhật weights. Quá lớn → diverge (loss tăng). Quá nhỏ → hội tụ chậm
- 0.01 là giá trị chuẩn cho SGD với Momentum trên classification task

#### momentum = 0.9
- Tích lũy "quán tính" từ gradient trước → giúp vượt qua local minima
- 0.9 nghĩa là 90% momentum cũ + 10% gradient mới

#### weight_decay = 1e-4
- L2 Regularization: thêm penalty `λ × ||w||²` vào loss
- Buộc weights nhỏ → model đơn giản hơn → ít overfit
- 1e-4 là giá trị chuẩn, quá lớn sẽ underfit

#### warmup_epochs = 5
- 5 epoch đầu LR tăng dần từ 0.002 → 0.01
- Weights ban đầu random → gradient lớn và hỗn loạn → LR nhỏ giúp ổn định

#### label_smoothing = 0.1
- Hard label: [0, 0, 1, 0] → Soft label: [0.0023, 0.0023, 0.907, 0.0023]
- Ngăn model quá tự tin → cải thiện generalization 1-2%

#### dropout = 0.2
- Tắt ngẫu nhiên 20% neurons trong FC layer
- Buộc model không phụ thuộc vào neuron cụ thể → robust hơn

#### grad_clip = 5.0
- Giới hạn norm của gradient ≤ 5.0
- Ngăn exploding gradient khi gradient quá lớn (thường xảy ra đầu training)

#### patience = 10
- Early stopping: dừng nếu val_acc không cải thiện trong 10 epoch liên tiếp
- Tiết kiệm thời gian, tránh overfit khi train quá lâu

#### seed = 42
- Cố định random seed → kết quả có thể tái lập (reproducible)
- 42 là convention phổ biến (tham chiếu "Hitchhiker's Guide to the Galaxy")

#### Device & CUDA
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
```
- Tự phát hiện GPU → dùng GPU nếu có, CPU nếu không
- `deterministic = True`: Kết quả giống nhau mỗi lần chạy (quan trọng cho nghiên cứu)

---

### CELL 4: DATA LOADING & AUGMENTATION

#### Transform cho Training:

```python
train_transform = transforms.Compose([
    transforms.Resize((104, 104)),        # Resize lớn hơn để crop
    transforms.RandomCrop((96, 96)),       # Crop ngẫu nhiên → dịch chuyển
    transforms.RandomRotation(20),         # Xoay ±20°
    transforms.RandomAffine(translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10),
    transforms.RandomPerspective(0.3, p=0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),
    transforms.Normalize(MEAN, STD),
])
```

**Phân tích từng augmentation:**

| Augmentation | Tham số | Mô phỏng thực tế | Tại sao cần? |
|:---|:---|:---|:---|
| **Resize + RandomCrop** | 104→96 | Biển không nằm chính giữa | Học vị trí bất kỳ |
| **RandomRotation** | ±20° | Biển bị nghiêng do gió, lắp đặt | Robust với góc nghiêng |
| **RandomAffine** | translate, scale, shear | Biển xa/gần, góc xiên | Bất biến với khoảng cách |
| **RandomPerspective** | 0.3, p=0.5 | Nhìn từ xe đang chạy, góc 3D | Robust với perspective |
| **ColorJitter** | 0.5 brightness/contrast | Nắng gắt, bóng cây, ban đêm | Bất biến với ánh sáng |
| **RandomGrayscale** | p=0.05 | Camera đen trắng, IR | Học hình dạng, không chỉ màu |
| **GaussianBlur** | kernel=3 | Camera mờ, rung xe | Robust với ảnh mờ |
| **RandomErasing** | p=0.2 | Biển bị che 1 phần (cây, xe) | Robust với occlusion |
| **Normalize** | MEAN, STD | Chuẩn hóa pixel | Hội tụ nhanh, ổn định |

#### Transform cho Validation/Test:
```python
val_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
```
**Không augment** — đánh giá trên ảnh gốc để phản ánh đúng năng lực model.

#### Normalize values:
```python
MEAN = [0.3403, 0.3121, 0.3214]
STD = [0.2724, 0.2608, 0.2669]
```
Tính từ toàn bộ GTSRB training set. Chuẩn hóa giúp mỗi channel có mean≈0, std≈1 → gradient ổn định hơn.

#### Train/Val Split (85/15):
```python
val_size = int(0.15 * len(train_full))
train_dataset, val_dataset_raw = random_split(...)
```

**Tại sao cần validation set?**
- Không thể dùng test set để chọn model (data leakage)
- Val set dùng để: (1) chọn best model, (2) early stopping, (3) monitor overfitting
- 85/15 là tỷ lệ phổ biến. 80/20 cũng được

**Lưu ý:** Val set dùng `val_transform` (không augment) dù lấy từ tập train → đánh giá công bằng.

#### DataLoader:
```python
DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
```
- `shuffle=True` (train): Xáo trộn thứ tự mỗi epoch → tránh model học thuộc thứ tự
- `num_workers=2`: 2 CPU threads load data song song → GPU không phải đợi
- `pin_memory=True`: Giữ data trong pinned memory → transfer CPU→GPU nhanh hơn

---

### CELL 5: VISUALIZE DỮ LIỆU

Hiển thị 32 ảnh mẫu + biểu đồ phân bố class.

**Tại sao cần visualize?**
1. **Kiểm tra data đúng không:** Ảnh có đúng biển báo? Label đúng không?
2. **Phát hiện class imbalance:** Nếu class A có 2000 ảnh, class B chỉ 200 → model bias về class A
3. **Hiểu data trước khi train:** Biết data trông như thế nào giúp chọn augmentation phù hợp

**Inverse Normalize:**
```python
inv_normalize = transforms.Normalize(mean=[-m/s for m,s in zip(MEAN,STD)], std=[1/s for s in STD])
```
Đảo ngược normalize để hiển thị ảnh đúng màu gốc (matplotlib cần pixel range [0,1]).

---

## PHẦN II: KIẾN TRÚC MOBILENETV2

---

### CELL 6: XÂY DỰNG MOBILENETV2 TỪ ĐẦU

#### Block 1: _make_divisible()

```python
def _make_divisible(v, divisor=8, min_value=None):
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
```

**Chức năng:** Làm tròn số channels lên bội số của 8.

**Tại sao phải có?**
- GPU xử lý nhanh nhất khi số channels chia hết cho 8 (do kiến trúc CUDA sử dụng warp size = 32, và tensor cores hoạt động trên bội số 8)
- Ví dụ: `_make_divisible(24 * 0.75) = _make_divisible(18) = 16` (thay vì 18)
- Tối ưu 10-20% tốc độ inference mà không ảnh hưởng accuracy

**Khi nào được gọi?** Khi dùng `width_mult` để scale model (ví dụ width_mult=0.5 → model nhỏ hơn 2×).

---

#### Block 2: ConvBNReLU6

```python
class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
```

**Chức năng:** Khối cơ bản nhất — Conv + BatchNorm + ReLU6 đóng gói lại.

**Phân tích từng lớp con:**

**1. Conv2d (bias=False):**
- `bias=False` vì BatchNorm ngay sau đó đã có bias riêng (β parameter)
- Nếu dùng bias ở cả Conv và BN → dư thừa, tốn memory
- `padding = (kernel_size - 1) // 2`: Giữ nguyên spatial size (same padding)
- `groups`: groups=1 là conv thường, groups=in_channels là depthwise

**2. BatchNorm2d:**
- Chuẩn hóa output: `y = (x - mean) / sqrt(var + eps) * γ + β`
- `γ` (scale) và `β` (shift) là learnable parameters
- Training: tính mean/var từ mini-batch
- Inference: dùng running mean/var (tích lũy từ training)
- **Tác dụng:** Ổn định training, cho phép LR lớn hơn, regularization nhẹ

**3. ReLU6 (inplace=True):**
- `ReLU6(x) = min(max(0, x), 6)`
- `inplace=True`: Ghi đè lên input → tiết kiệm memory (không tạo tensor mới)
- Giới hạn output ≤ 6 → phù hợp quantization INT8 cho mobile deployment
- Tại sao 6? Paper gốc MobileNetV1 thực nghiệm cho thấy 6 là ngưỡng tốt nhất

---

#### Block 3: InvertedResidual — KHỐI QUAN TRỌNG NHẤT

```python
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_skip = (stride == 1 and in_channels == out_channels)

        layers = []
        # 1. Expansion (1×1 Conv) — bỏ qua nếu expand_ratio=1
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_channels, hidden_dim, kernel_size=1))
        
        # 2. Depthwise (3×3 Conv, groups=hidden_dim)
        layers.append(ConvBNReLU6(hidden_dim, hidden_dim, kernel_size=3,
                                   stride=stride, groups=hidden_dim))
        
        # 3. Projection (1×1 Conv, LINEAR — KHÔNG ReLU!)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_skip:
            return x + self.conv(x)
        else:
            return self.conv(x)
```

**Phân tích chi tiết 3 giai đoạn:**

##### Giai đoạn 1: Expansion Layer (1×1 Convolution)

```
Input: 24 channels → Expand ×6 → 144 channels
```

- **Nhiệm vụ:** Mở rộng số chiều (channels) để có nhiều "không gian" hơn cho xử lý
- **Tại sao 1×1?** Chỉ thay đổi channels, không thay đổi spatial → rất nhanh
- **Tại sao cần expand?** Depthwise conv xử lý từng channel riêng, nếu ít channels → ít features → model yếu. Expand ra nhiều channels → depthwise có nhiều features để xử lý
- **expand_ratio=1:** Bỏ qua lớp này (stage đầu tiên, input đã đủ nhỏ)

**Analogia:** Như mở rộng mặt bàn làm việc trước khi bắt đầu phân tích — cần nhiều chỗ để trải tài liệu ra.

##### Giai đoạn 2: Depthwise Convolution (3×3, groups=channels)

```
144 channels → 3×3 conv cho MỖI channel riêng → 144 channels
```

- **groups=hidden_dim:** Mỗi channel có filter 3×3 riêng, KHÔNG chia sẻ với channel khác
- **Nhiệm vụ:** Học spatial features — cạnh ở đâu, đường cong thế nào, viền ở vị trí nào
- **Tại sao depthwise?** Conv thường 3×3 từ 144→144 channels = 144×144×3×3 = 186,624 params. Depthwise = 144×1×3×3 = 1,296 params. **Giảm 144×!**

**Mỗi channel "chuyên gia" về 1 loại feature:**
- Channel 1: phát hiện cạnh ngang
- Channel 2: phát hiện cạnh dọc
- Channel 23: phát hiện góc tròn
- Channel 87: phát hiện gradient màu đỏ→trắng

##### Giai đoạn 3: Projection Layer (1×1 Conv, LINEAR)

```
144 channels → Project → 32 channels (KHÔNG có ReLU!)
```

- **Nhiệm vụ:** Nén thông tin từ chiều cao về chiều thấp — giữ lại features quan trọng
- **KHÔNG có ReLU:** Đây là điểm khác biệt quan trọng nhất so với các kiến trúc khác

**Tại sao không ReLU?**
- ReLU(x) = max(0, x): mọi giá trị âm → 0
- Ở chiều thấp (32 channels), mỗi channel mang nhiều thông tin. Nếu ReLU đặt 1 channel = 0 → mất thông tin không thể phục hồi
- Paper chứng minh toán học: manifold of interest (tập thông tin quan trọng) bị "sụp đổ" khi áp ReLU ở chiều thấp

##### Skip Connection:

```python
self.use_skip = (stride == 1 and in_channels == out_channels)
return x + self.conv(x)  # Element-wise addition
```

**Điều kiện:** Chỉ khi stride=1 (không giảm spatial) VÀ in == out channels.
- stride=2: feature map nhỏ đi, không thể cộng với input
- in ≠ out: shape khác nhau, không thể cộng

**Tại sao cần skip?**
- Giải quyết vanishing gradient: gradient "chảy thẳng" qua skip, không bị giảm qua nhiều lớp
- Model học residual F(x) = H(x) - x thay vì H(x) trực tiếp → dễ tối ưu hơn
- Cho phép xây network rất sâu (MobileNetV2 có 53 conv layers) mà không bị vanishing gradient

---

#### Block 4: MobileNetV2 — Full Network Assembly

```python
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=43, width_mult=1.0, dropout=0.2):
        inverted_residual_setting = [
            # t, c,   n, s
            [1, 16,  1, 1],    # Stage 1
            [6, 24,  2, 2],    # Stage 2 — downsample
            [6, 32,  3, 2],    # Stage 3 — downsample
            [6, 64,  4, 2],    # Stage 4 — downsample
            [6, 96,  3, 1],    # Stage 5
            [6, 160, 3, 2],    # Stage 6 — downsample
            [6, 320, 1, 1],    # Stage 7
        ]
```

**Phân tích từng stage:**

| Stage | t | c | n | s | Feature Map | Vai trò |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| First Conv | - | 32 | 1 | 1 | 96×96 | Trích xuất features cơ bản (cạnh, màu) |
| 1 | 1 | 16 | 1 | 1 | 96×96 | Nén channels, học features đơn giản |
| 2 | 6 | 24 | 2 | 2 | 48×48 | Downsample, học texture (vân, hoa văn) |
| 3 | 6 | 32 | 3 | 2 | 24×24 | Học hình dạng (tròn, tam giác, vuông) |
| 4 | 6 | 64 | 4 | 2 | 12×12 | Học phần tử (mũi tên, số, biểu tượng) |
| 5 | 6 | 96 | 3 | 1 | 12×12 | Tinh chỉnh features (không downsample) |
| 6 | 6 | 160 | 3 | 2 | 6×6 | Học ngữ nghĩa cao (loại biển báo) |
| 7 | 6 | 320 | 1 | 1 | 6×6 | Tổng hợp features cuối cùng |
| Last Conv | - | 1280 | 1 | 1 | 6×6 | Mở rộng cho classifier |

**First Conv stride=1 thay vì 2:** Paper gốc dùng stride=2 (input 224→112). Ở đây input 96×96 nhỏ hơn, stride=2 sẽ giảm xuống 48×48 quá sớm → mất thông tin. Stride=1 giữ 96×96 → qua nhiều lớp hơn trước khi giảm.

**Classifier:**
```python
self.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, 43),
)
```
- Adaptive Average Pooling: [B, 1280, 6, 6] → [B, 1280, 1, 1]
- Flatten: [B, 1280]
- Dropout: Tắt 20% neurons → chống overfit
- Linear: 1280 features → 43 classes

#### Weight Initialization (Kaiming):

```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)   # γ = 1
            nn.init.zeros_(m.bias)    # β = 0
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
```

**Tại sao cần khởi tạo weights đúng cách?**
- Random thường (Normal(0,1)): Variance tăng/giảm qua mỗi lớp → exploding/vanishing gradient
- **Kaiming initialization:** Giữ variance ổn định qua các lớp → training ổn định ngay từ đầu
- `mode='fan_out'`: Tính variance theo output channels → phù hợp cho inference
- BN γ=1, β=0: Ban đầu BN hoạt động như identity → không thay đổi output

---

## PHẦN III: TRAINING, ĐÁNH GIÁ & VISUALIZATION

---

### CELL 7: LOSS FUNCTION, OPTIMIZER, SCHEDULER

#### CrossEntropyLoss + Label Smoothing

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**CrossEntropyLoss hoạt động thế nào?**
```
Loss = -Σ y_i × log(p_i)
```
- `y_i`: target (0 hoặc 1)
- `p_i`: xác suất model dự đoán cho class i (output của softmax)
- Loss nhỏ khi p_i cao cho class đúng, lớn khi p_i thấp

**Label Smoothing (ε=0.1):**
```
Không smoothing: y = [0, 0, 1, 0, ..., 0]         → 100% chắc class 2
Có smoothing:    y = [0.0023, 0.0023, 0.907, ...]  → 90.7% class 2, 0.23% mỗi class khác
```
- Công thức: `y_smooth = y × (1 - ε) + ε / K` (K = 43 classes)
- Tác dụng: Model không cố đạt 100% confidence → output "khiêm tốn" hơn → generalize tốt hơn

#### SGD + Nesterov Momentum

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
```

**SGD cơ bản:** `w = w - lr × gradient`
- Vấn đề: Gradient noisy (chỉ tính từ mini-batch) → dao động nhiều

**+ Momentum (0.9):** `v = 0.9×v + gradient; w = w - lr×v`
- Tích lũy "quán tính" từ gradient trước → di chuyển mượt hơn
- Giúp vượt qua local minima và saddle points
- 0.9 = chuẩn, 0.99 mượt hơn nhưng phản ứng chậm

**+ Nesterov:** Tính gradient tại vị trí "nhìn trước" thay vì vị trí hiện tại
- Nhanh hơn momentum thường ~2-3% convergence speed
- Đặc biệt hiệu quả khi gần minimum

**Tại sao SGD mà không phải Adam?**
- Adam hội tụ nhanh hơn ban đầu nhưng SGD+Momentum thường đạt accuracy cao hơn ở cuối
- SGD generalize tốt hơn Adam trên nhiều benchmark (theo nhiều paper)
- Trade-off: SGD cần tune LR cẩn thận hơn

#### Warmup + Cosine Annealing LR Scheduler

```python
class WarmupCosineScheduler:
    def step(self):
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)  # Linear warmup
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + cos(π × progress))  # Cosine
```

**Warmup (Epoch 1-5):**
```
Epoch 1: lr = 0.01 × 1/5 = 0.002
Epoch 2: lr = 0.01 × 2/5 = 0.004
Epoch 3: lr = 0.01 × 3/5 = 0.006
Epoch 4: lr = 0.01 × 4/5 = 0.008
Epoch 5: lr = 0.01 × 5/5 = 0.010  ← peak
```

**Cosine Decay (Epoch 6-50):**
```
LR giảm theo đường cong cosine: 0.01 → ~0 (rất mượt)
```

**Tại sao không dùng StepLR (giảm cứng)?**
- StepLR giảm đột ngột (ví dụ ×0.1 mỗi 20 epoch) → model bị "sốc"
- Cosine giảm mượt → model tinh chỉnh dần, không bị gián đoạn
- Cosine decay được chứng minh hiệu quả hơn StepLR trên nhiều task

---

### CELL 8: TRAINING LOOP

#### Mixed Precision Training (AMP):

```python
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

with autocast():           # Forward pass bằng float16
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()  # Backward pass scale lên để giữ precision
scaler.unscale_(optimizer)     # Unscale gradient trước clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Gradient clipping
scaler.step(optimizer)         # Update weights
scaler.update()                # Cập nhật scale factor
```

**Tại sao Mixed Precision?**
- float32: 32 bits/số → chính xác nhưng chậm, tốn memory
- float16: 16 bits/số → nhanh gấp 2× trên Tensor Cores, tiết kiệm 50% VRAM
- **GradScaler:** float16 có range nhỏ → gradient nhỏ có thể = 0 (underflow). Scale lên ×1024 trước backward, scale xuống sau → giữ precision

#### Gradient Clipping:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```
- Nếu ||gradient|| > 5.0: `gradient = gradient × 5.0 / ||gradient||`
- Ngăn exploding gradient mà không thay đổi hướng gradient

#### Training Log:
```python
training_log = {
    'epoch': [], 'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [], 'lr': [], 'epoch_time': []
}
```
Ghi lại metrics mỗi epoch → phân tích sau training.

#### Checkpoint System:
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_acc': best_val_acc,
    'training_log': training_log,
}
```

**Lưu 3 loại checkpoint:**
1. `best_model.pth`: Val acc cao nhất → dùng cho evaluation
2. `checkpoint_epoch_N.pth`: Mỗi 10 epoch → backup
3. `checkpoint_latest.pth`: Mỗi epoch → resume khi disconnect

**Resume logic:**
```python
if CONFIG['resume'] and os.path.exists(latest_ckpt):
    # Load model, optimizer, scheduler, training_log
    # Tiếp tục từ epoch đã dừng
```

#### Early Stopping:
```python
if patience_counter >= CONFIG['patience']:  # 10 epochs không cải thiện
    break
```
Tránh train thừa → tiết kiệm thời gian, tránh overfit.

---

### CELL 9-13: ĐÁNH GIÁ & VISUALIZATION

#### Confusion Matrix (43×43):
- Hàng = True label, Cột = Predicted label
- Đường chéo chính = dự đoán đúng
- Off-diagonal = nhầm lẫn → phát hiện class nào model hay nhầm

#### Per-class Accuracy:
- Bar chart cho từng class → nhanh chóng thấy class nào yếu
- Màu: Đỏ < 85%, Vàng 85-95%, Xanh > 95%

#### Classification Report:
```
Precision = TP / (TP + FP)   → "Khi model nói X, đúng bao nhiêu %?"
Recall    = TP / (TP + FN)   → "Trong tất cả X thực, model tìm được bao nhiêu %?"
F1-score  = 2 × P × R / (P + R) → Trung bình hài hòa
```

#### Overall Metrics Bar Chart:
- Accuracy, Precision, Recall, F1-Score trên toàn bộ test set
- Đánh giá tổng thể: nếu cả 4 đều > 95% → model rất tốt

---

### CELL 14: MODEL ANALYSIS

```python
total_params = sum(p.numel() for p in model.parameters())
# ~2.2M parameters → ~9 MB (float32)
```

**So sánh params theo layer:**
- `features` (backbone): ~2.15M → 98% tổng params
- `classifier` (FC): ~55K → 2% tổng params

Phần lớn "trí tuệ" nằm ở backbone (feature extraction), classifier chỉ là bước quyết định cuối cùng.

---

### CELL 15: DỰ ĐOÁN ẢNH BÊN NGOÀI

```python
predict_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
```

**Pipeline dự đoán:** Ảnh bất kỳ → Resize 96×96 → Normalize → Model → Softmax → Top-5 classes

**Lưu ý quan trọng:**
- Model chỉ phân loại, KHÔNG detect → ảnh phải crop sẵn biển báo
- Model train trên biển Đức (GTSRB) → accuracy thấp hơn trên biển nước khác (domain shift)
- Confidence thấp (<50%) thường = model không chắc chắn, cần kiểm tra lại

---

## PHẦN IV: TỔNG KẾT

### Strengths (Điểm mạnh):
1. Model tự xây 100% từ đầu — hiểu rõ từng dòng code
2. Kiến trúc theo paper gốc — đảm bảo tính học thuật
3. Training pipeline hoàn chỉnh: augmentation, scheduling, checkpointing, logging
4. Evaluation đa chiều: confusion matrix, per-class, overall metrics
5. Hỗ trợ resume training — phù hợp Colab miễn phí

### Weaknesses (Điểm yếu):
1. Val accuracy cao ảo do đặc thù video GTSRB
2. Chỉ phân loại, chưa detect
3. Chỉ biển Đức, chưa test biển Việt Nam

### Key Takeaways:
- **Inverted Residual** = trái tim MobileNetV2 → narrow-wide-narrow + skip + linear bottleneck
- **Depthwise Separable** = giảm 8× params → lightweight cho mobile
- **ReLU6 + Linear Bottleneck** = thiết kế cho mobile deployment
- **Label Smoothing + Warmup + Cosine** = training stable và generalize tốt
