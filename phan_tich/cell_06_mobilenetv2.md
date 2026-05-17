# CELL 6: KIẾN TRÚC MOBILENETV2 — XÂY DỰNG TỪ ĐẦU

## 1. Tổng quan

Cell này xây dựng **toàn bộ kiến trúc MobileNetV2** từ đầu, không sử dụng model pretrained. Gồm 4 thành phần:
1. `_make_divisible()` — Hàm tiện ích
2. `ConvBNReLU6` — Khối Conv cơ bản
3. `InvertedResidual` — Khối xây dựng chính (bottleneck)
4. `MobileNetV2` — Lắp ráp toàn bộ network

---

## 2. Block 1: _make_divisible()

```python
def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
```

### Chức năng
Làm tròn số channels lên bội số gần nhất của 8.

### Tại sao phải có?
- **Tối ưu GPU:** CUDA cores hoạt động theo warp (32 threads), Tensor Cores xử lý theo bội 8. Channels chia hết cho 8 → GPU tính toán hiệu quả hơn 10-20%
- **Khi nào cần?** Khi dùng `width_mult` ≠ 1.0 (ví dụ 0.75): `24 × 0.75 = 18` → làm tròn thành 16 hoặc 24
- **Điều kiện an toàn:** `if new_v < 0.9 * v` → nếu làm tròn xuống quá 10% so với giá trị gốc → tăng lên 1 bậc divisor

### Ví dụ
```
_make_divisible(24)   = 24  (đã chia hết 8)
_make_divisible(18)   = 16  (18 → 16)
_make_divisible(100)  = 104 (100 → 104)
_make_divisible(7)    = 8   (min_value = 8)
```

---

## 3. Block 2: ConvBNReLU6

```python
class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
```

### Chức năng
Đóng gói 3 lớp thường đi cùng nhau: Convolution → BatchNorm → ReLU6.

### Phân tích từng lớp con

#### 3.1 Conv2d

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups, bias=False)
```

**Convolution là gì?** Trượt bộ lọc (kernel/filter) qua ảnh, tại mỗi vị trí tính tích chập (element-wise multiply rồi sum) → tạo feature map.

**Các tham số quan trọng:**

| Tham số | Ý nghĩa | Ảnh hưởng |
|:---|:---|:---|
| `kernel_size` | Kích thước filter (3×3 hoặc 1×1) | 3×3 nhìn vùng lớn hơn, 1×1 chỉ kết hợp channels |
| `stride` | Bước nhảy | stride=2 → output nhỏ đi 2× (downsampling) |
| `padding` | Đệm viền | `(k-1)//2` giữ nguyên spatial size |
| `groups` | Số nhóm | groups=1: conv thường, groups=channels: depthwise |
| `bias=False` | Không dùng bias | Vì BatchNorm đã có bias (β parameter) |

**Tại sao bias=False?**
BatchNorm tính: `y = γ × (x - μ) / σ + β`. Nếu Conv có bias b: `y = γ × (wx + b - μ) / σ + β`. Bias b bị triệt tiêu bởi `-μ` → dư thừa. Bỏ bias tiết kiệm memory.

#### 3.2 BatchNorm2d

```python
nn.BatchNorm2d(num_features)
```

**Chuẩn hóa batch:** `y = γ × (x - μ_batch) / √(σ²_batch + ε) + β`

- `μ_batch, σ²_batch`: mean và variance tính từ mini-batch hiện tại
- `γ, β`: learnable parameters (scale và shift)
- `ε = 1e-5`: ngăn chia cho 0

**Tại sao cần BatchNorm?**
1. **Internal Covariate Shift:** Phân bố output mỗi lớp thay đổi khi weights cập nhật → lớp sau phải liên tục thích nghi → training chậm. BN cố định phân bố → training nhanh hơn
2. **Cho phép LR lớn hơn:** Output ổn định → gradient ổn định → có thể dùng LR lớn hơn mà không sợ diverge
3. **Regularization nhẹ:** μ và σ tính từ mini-batch (không phải toàn bộ data) → tạo noise nhẹ → tác dụng regularization

**Training vs Inference:**
- Training: tính μ, σ từ mini-batch + cập nhật running_mean, running_var (exponential moving average)
- Inference: dùng running_mean, running_var (đã ổn định) → kết quả deterministic

#### 3.3 ReLU6

```python
nn.ReLU6(inplace=True)
```

**Công thức:** `ReLU6(x) = min(max(0, x), 6)`

**So sánh các activation:**

| Activation | Công thức | Range | Vấn đề |
|:---|:---|:---|:---|
| ReLU | max(0, x) | [0, +∞) | Output không giới hạn → khó quantize |
| ReLU6 | min(max(0,x), 6) | [0, 6] | **Phù hợp mobile** |
| Sigmoid | 1/(1+e^-x) | (0, 1) | Vanishing gradient |
| Swish | x × sigmoid(x) | (-∞, +∞) | Tốn compute |

**Tại sao chọn ReLU6?**
- Giới hạn output ≤ 6 → phạm vi nhỏ → dễ biểu diễn bằng fixed-point INT8
- Khi quantize float32 → INT8: cần biết range để map. [0,6] dễ map hơn [0,1000]
- Paper MobileNetV1 thực nghiệm: ReLU6 cho kết quả tốt nhất khi quantize

**inplace=True:** Ghi đè kết quả lên tensor input → không tạo tensor mới → tiết kiệm memory. Chỉ dùng khi không cần giữ input tensor.

---

## 4. Block 3: InvertedResidual — TRÁI TIM CỦA MOBILENETV2

```python
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_skip = (stride == 1 and in_channels == out_channels)
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_channels, hidden_dim, kernel_size=1))
        layers.append(ConvBNReLU6(hidden_dim, hidden_dim, kernel_size=3,
                                   stride=stride, groups=hidden_dim))
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)
```

### 4.1 Kiến trúc 3 giai đoạn

```
Input (narrow)  ─┐
                 │
    ┌────────────▼────────────┐
    │ 1×1 Conv (Expansion)    │  narrow → WIDE  (+BN +ReLU6)
    │ Mở rộng channels ×t     │
    ├─────────────────────────┤
    │ 3×3 Depthwise Conv      │  WIDE → WIDE   (+BN +ReLU6)
    │ Xử lý spatial features  │
    ├─────────────────────────┤
    │ 1×1 Conv (Projection)   │  WIDE → narrow  (+BN, KHÔNG ReLU!)
    │ Nén lại channels        │
    └────────────┬────────────┘
                 │
Output (narrow) ─┤── + Skip Connection (nếu stride=1, in==out)
```

### 4.2 Giai đoạn 1: Expansion (1×1 Pointwise Conv)

```python
if expand_ratio != 1:
    layers.append(ConvBNReLU6(in_channels, hidden_dim, kernel_size=1))
# hidden_dim = in_channels × expand_ratio
```

**Chức năng:** Mở rộng số channels lên `expand_ratio` lần (thường ×6).

**Ví dụ:** Input 24 channels → Expand ×6 → 144 channels

**Tại sao cần mở rộng?**
- Depthwise conv (giai đoạn 2) xử lý TỪNG channel riêng biệt
- Nếu ít channels (24) → ít features → hạn chế khả năng biểu diễn
- Mở rộng ra 144 channels → depthwise có 144 bộ lọc riêng → phong phú hơn

**1×1 Conv:**
- Không thay đổi spatial size (không nhìn vùng lân cận)
- Chỉ kết hợp thông tin giữa channels (cross-channel interaction)
- Rất nhanh: chỉ multiply-accumulate, không cần nhìn neighbors

**Khi expand_ratio = 1:** Bỏ qua lớp này (stage đầu tiên, input 32ch → output 16ch, không cần expand)

### 4.3 Giai đoạn 2: Depthwise Convolution (3×3)

```python
ConvBNReLU6(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim)
```

**Chức năng:** Học spatial features (vị trí, hình dạng, cạnh) cho MỖI channel riêng biệt.

**Depthwise vs Standard Convolution:**

```
Standard Conv 3×3 (144 input → 144 output):
- 1 filter kết hợp TẤT CẢ 144 input channels
- Params: 144 × 144 × 3 × 3 = 186,624

Depthwise Conv 3×3 (groups=144):
- MỖI channel có 1 filter 3×3 riêng
- KHÔNG kết hợp giữa channels
- Params: 144 × 1 × 3 × 3 = 1,296
- Giảm 144× parameters!
```

**Tại sao depthwise hiệu quả?**
- Tách spatial learning (depthwise) và channel mixing (pointwise) thành 2 bước
- Paper "Xception" chứng minh: kết quả gần bằng conv thường với ít params hơn nhiều
- Mỗi channel "chuyên gia" về 1 loại feature spatial

**Ý nghĩa filter 3×3:**
- Nhìn vùng lân cận 3×3 pixels → phát hiện cạnh, góc, texture
- Nhiều lớp 3×3 chồng nhau → receptive field ngày càng lớn
- 2 lớp 3×3 = receptive field 5×5 nhưng ít params hơn 1 lớp 5×5

### 4.4 Giai đoạn 3: Projection (1×1 Linear Bottleneck)

```python
nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
nn.BatchNorm2d(out_channels),
# KHÔNG CÓ ReLU!
```

**Chức năng:** Nén channels từ wide (144) về narrow (32).

**ĐÂY LÀ ĐIỂM KHÁC BIỆT QUAN TRỌNG NHẤT CỦA MOBILENETV2:**

**Tại sao KHÔNG có ReLU ở đây?**

Paper "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (Sandler et al., 2018) chứng minh:

1. ReLU(x) = max(0, x) → mọi giá trị âm trở thành 0
2. Ở chiều cao (144 channels): mất vài channels không ảnh hưởng nhiều
3. Ở chiều thấp (32 channels): mỗi channel mang thông tin QUAN TRỌNG
4. Nếu ReLU đặt 5/32 channels = 0 → mất 15% thông tin → KHÔNG thể phục hồi

**Giải thích toán học:**
- Dữ liệu nằm trên một manifold (đa tạp) trong không gian cao chiều
- Khi project xuống không gian thấp chiều (32D), ReLU có thể "cắt" mất phần manifold ở vùng âm
- Linear activation giữ nguyên manifold → không mất thông tin

**Thực nghiệm:** Paper báo cáo accuracy giảm 2-3% nếu thêm ReLU ở projection layer.

### 4.5 Skip Connection (Residual Connection)

```python
self.use_skip = (stride == 1 and in_channels == out_channels)

def forward(self, x):
    if self.use_skip:
        return x + self.conv(x)  # Element-wise addition
    else:
        return self.conv(x)
```

**Điều kiện:** stride=1 (không downsample) VÀ in_channels == out_channels (cùng shape).

**Tại sao stride=2 không có skip?**
- stride=2: spatial giảm 2× (ví dụ 48×48 → 24×24)
- Input shape [B, 24, 48, 48] ≠ Output shape [B, 32, 24, 24]
- Không thể cộng 2 tensor khác shape

**Tại sao cần skip connection?**

1. **Vanishing Gradient Problem:**
   - Network sâu (53 layers) → gradient phải nhân qua nhiều lớp
   - Mỗi lần nhân: gradient nhỏ đi → đến lớp đầu gradient ≈ 0 → lớp đầu không học được
   - Skip cho gradient "đường tắt" chảy trực tiếp → mọi lớp đều nhận đủ gradient

2. **Residual Learning:**
   - Thay vì học H(x) trực tiếp, model học F(x) = H(x) - x (phần "khác biệt")
   - F(x) thường nhỏ và dễ học hơn H(x)
   - Trường hợp tệ nhất: F(x) ≈ 0 → output ≈ input (không làm gì cũng không gây hại)

3. **Feature Reuse:**
   - Input features không bị mất → lớp sau có thể dùng lại features từ lớp trước

### 4.6 Tại sao gọi "Inverted"?

| | ResNet (Traditional) | MobileNetV2 (Inverted) |
|:---|:---|:---|
| Cấu trúc | Wide → Narrow → Wide | **Narrow → Wide → Narrow** |
| Skip | Ở lớp wide (nhiều channels) | Ở lớp **narrow** (ít channels) |
| Memory | Skip tensor lớn (wide) | Skip tensor nhỏ (narrow) → **tiết kiệm** |
| Activation | ReLU ở mọi lớp | **Không ReLU** ở projection |

"Inverted" vì đảo ngược so với ResNet bottleneck truyền thống.

---

## 5. Block 4: MobileNetV2 — Full Network

```python
class MobileNetV2(nn.Module):
    inverted_residual_setting = [
        # t,  c,   n, s
        [1,  16,  1, 1],
        [6,  24,  2, 2],
        [6,  32,  3, 2],
        [6,  64,  4, 2],
        [6,  96,  3, 1],
        [6,  160, 3, 2],
        [6,  320, 1, 1],
    ]
```

### 5.1 Bảng kiến trúc chi tiết

| Layer | Operator | t | c | n | s | Output Size | Skip? | Vai trò |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 0 | Conv2d 3×3 | - | 32 | 1 | 1 | 96×96 | - | Features cơ bản: cạnh, màu, gradient |
| 1 | Bottleneck | 1 | 16 | 1 | 1 | 96×96 | No | Nén channels, lọc features thô |
| 2 | Bottleneck | 6 | 24 | 2 | 2 | 48×48 | 1 skip | Texture: vân bề mặt, pattern lặp |
| 3 | Bottleneck | 6 | 32 | 3 | 2 | 24×24 | 2 skip | Hình dạng: tròn, tam giác, vuông |
| 4 | Bottleneck | 6 | 64 | 4 | 2 | 12×12 | 3 skip | Phần tử: mũi tên, số, biểu tượng |
| 5 | Bottleneck | 6 | 96 | 3 | 1 | 12×12 | 2 skip | Tinh chỉnh: kết hợp phần tử |
| 6 | Bottleneck | 6 | 160 | 3 | 2 | 6×6 | 2 skip | Ngữ nghĩa: loại biển cụ thể |
| 7 | Bottleneck | 6 | 320 | 1 | 1 | 6×6 | No | Tổng hợp toàn bộ features |
| 8 | Conv2d 1×1 | - | 1280 | 1 | 1 | 6×6 | - | Mở rộng cho classifier |
| 9 | AvgPool | - | 1280 | 1 | - | 1×1 | - | Global pooling |
| 10 | FC | - | 43 | 1 | - | - | - | Phân loại cuối cùng |

**t** = expansion ratio | **c** = output channels | **n** = repeat | **s** = stride (block đầu)

### 5.2 First Conv: stride=1 thay vì 2

```python
features = [ConvBNReLU6(3, input_channels, kernel_size=3, stride=1)]
# NOTE: stride=1 thay vì stride=2 trong paper gốc
```

**Paper gốc (ImageNet, 224×224):** stride=2 → 112×112 ngay lớp đầu
**Project này (GTSRB, 96×96):** stride=1 → giữ 96×96

**Tại sao?** Input 96×96 nhỏ hơn 224×224. Nếu stride=2 → 48×48 ngay lập tức → mất thông tin quá sớm. Giữ stride=1 cho model nhiều lớp hơn để xử lý trước khi downsample.

### 5.3 Adaptive Average Pooling

```python
x = nn.functional.adaptive_avg_pool2d(x, (1, 1))  # [B, 1280, 6, 6] → [B, 1280, 1, 1]
x = torch.flatten(x, 1)                            # [B, 1280, 1, 1] → [B, 1280]
```

**Chức năng:** Lấy giá trị trung bình toàn bộ spatial dimensions cho mỗi channel.

**Tại sao "Adaptive"?** Hoạt động với BẤT KỲ input size nào → output luôn 1×1. Không cần hardcode kernel size.

**Ý nghĩa:** Nén thông tin "ở đâu" → chỉ giữ "có feature gì". Channel 100 trung bình = 0.8 → "có feature 100 mạnh". Bất kể feature đó ở góc nào của ảnh.

### 5.4 Classifier

```python
self.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, 43),
)
```

- **Dropout(0.2):** Regularization cuối cùng trước FC → tránh overfit
- **Linear(1280, 43):** 1280 features → 43 xác suất (1 cho mỗi class)

### 5.5 Weight Initialization (Kaiming)

```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
```

**Vấn đề:** Nếu weights khởi tạo random thường (Normal(0,1)):
- Qua nhiều lớp: variance tăng (exploding) hoặc giảm (vanishing)
- Gradient cũng tăng/giảm tương tự → training bất ổn

**Giải pháp — Kaiming Initialization (He et al., 2015):**
```
weights ~ Normal(0, sqrt(2 / fan_out))
```
- `fan_out` = out_channels × kernel_size²
- Giữ variance output ≈ 1 qua mỗi lớp → gradient ổn định
- Thiết kế riêng cho ReLU (half of outputs = 0 → cần ×2)

**BN initialization:** γ=1, β=0 → BN ban đầu = identity (không thay đổi output)
**Linear initialization:** Normal(0, 0.01) → weights nhỏ, không gây bias ban đầu

## 6. Tổng kết
Cell này là **trái tim** của toàn bộ project. Mỗi thành phần đều có lý do tồn tại rõ ràng, từ _make_divisible cho tối ưu phần cứng đến linear bottleneck cho bảo toàn thông tin. Kiến trúc MobileNetV2 đạt sự cân bằng hiếm có giữa accuracy và efficiency.
