# Dàn ý Báo cáo: MobileNetV2 – Phân loại Biển báo Giao thông

---

## Phần 3 – Mô hình hóa: MobileNetV2

---

### 3.1. Giới thiệu mô hình MobileNetV2

#### 3.1.1. Thông số mô hình

**Cố định:**

| Thông số | Giá trị |
|---|---|
| **Tham số** | ~3.4M |
| **MACs** | ~300M |
| **Kích thước model** | ~14MB |
| **Độ sâu** | 53 tầng |
| **Latency (Pixel 1, CPU)** | ~75ms |

**Mặc định – có thể điều chỉnh:**

| Thông số | Mặc định | Tùy chỉnh |
|---|---|---|
| **Input size** | 224 × 224 × 3 | 96 / 128 / 160 / 192 / 224 |
| **Số lớp đầu ra** | 1000 (ImageNet) | Tùy bài toán |

#### 3.1.2. Lịch sử phát triển & Điểm nổi bật so với V1

MobileNetV2 là mạng CNN được Google giới thiệu năm 2018, thiết kế chuyên biệt để chạy hiệu quả trên thiết bị di động và hệ thống nhúng. Với ~3.4M tham số, ~300M MACs và kích thước ~14MB, mô hình đủ nhẹ để chạy thời gian thực ngay trên điện thoại phổ thông.

**Lịch sử phát triển dòng MobileNet:**
- **MobileNetV1 (2017)** – Howard et al., Google: lần đầu áp dụng Depthwise Separable Convolution, giảm ~8–9× tham số so với VGG/Inception nhưng chưa có skip connection.
- **MobileNetV2 (2018)** – Sandler et al., Google: bổ sung Inverted Residual Block và Linear Bottleneck, cải thiện đáng kể độ chính xác và khả năng quantize.
- **MobileNetV3 (2019)** – Howard et al.: tích hợp NAS, Hard-Swish và SE block, tối ưu thêm ~25% latency so với V2.

**MobileNetV1 gặp phải những vấn đề:**
- Mạng càng sâu, quá trình học càng kém ổn định, độ chính xác khó cải thiện thêm
- Khả năng biểu diễn đặc trưng bị giới hạn, chưa khai thác hết thông tin từ dữ liệu
- Một phần thông tin bị mất không thể phục hồi trong quá trình xử lý
- Khó triển khai hiệu quả trên phần cứng hạn chế mà không giảm đáng kể độ chính xác

**MobileNetV2 giải quyết trực tiếp từng điểm:**

| | MobileNetV1 | MobileNetV2 | Giải quyết vấn đề gì |
|---|---|---|---|
| **Skip connection** | ✗ | ✓ Inverted Residual | Mạng sâu học ổn định hơn |
| **Bottleneck** | Wide → Wide | Narrow → **Wide** → Narrow | Biểu diễn đặc trưng phong phú hơn |
| **Activation cuối block** | ReLU | **Linear** (không activation) | Bảo toàn thông tin qua từng block |
| **Quantization-friendly** | Trung bình | ✓ Tốt hơn nhờ ReLU6 | Triển khai hiệu quả trên mobile/INT8 |
| **Tham số** | ~4.2M | ~3.4M | Nhẹ hơn 19% |
| **MACs** | ~569M | ~300M | Tính toán ít hơn ~47% |
| **Top-1 ImageNet** | 70.6% | 72.0% | Chính xác hơn 1.4% |
| **Backbone cho task khác** | Hạn chế | ✓ SSDLite, DeepLabV3+ | Tái sử dụng cho nhiều bài toán |

**Phân tích sâu hai đổi mới cốt lõi:**

- **Inverted Residual Block**: Skip connection nối giữa hai lớp hẹp, phần tính toán nặng (Depthwise Conv) diễn ra ở không gian mở rộng — vừa tiết kiệm bộ nhớ, vừa giữ gradient xuyên suốt mạng.
- **Linear Bottleneck**: Tầng projection cuối không dùng ReLU — khi số kênh nhỏ, ReLU set nhiều giá trị về 0 gây mất thông tin vĩnh viễn. Giữ tuyến tính ở đây bảo toàn toàn bộ thông tin trước khi truyền sang block tiếp theo.

#### 3.1.3. Ứng dụng thực tế

| Bài toán | Mô hình | Bộ dữ liệu | Kết quả |
|---|---|---|---|
| **Phân loại ảnh** | MobileNetV2 | ImageNet | 72.0% Top-1 |
| **Phát hiện đối tượng** | SSDLite + MobileNetV2 | COCO | 22 mAP, 75ms / Pixel Phone |
| **Phân đoạn ngữ nghĩa** | DeepLabV3+ + MobileNetV2 | PASCAL VOC | 75.32% mIoU |
| **Nhận dạng khuôn mặt** | MobileNetV2 + ArcFace | LFW | 99.2% accuracy |
| **Phân tích video** | MobileNetV2 + LSTM | UCF-101 | ~82% accuracy |

> **Vai trò chung:** MobileNetV2 đóng vai trò **backbone trích xuất đặc trưng** — phần đầu giữ nguyên từ pretrained ImageNet, chỉ thay phần đầu ra tùy bài toán.

---

### 3.2. Nền tảng lý thuyết

#### 3.2.1. Tích chập truyền thống (Standard Convolution)
- Công thức MACs: `H × W × k² × C_in × C_out`
- Ví dụ: kernel 3×3, input 224×224×3 → output 224×224×32
- Nhược điểm: chi phí tính toán rất lớn khi C_in, C_out cao
- Hình minh họa: kernel 3×3 trượt trên feature map

#### 3.2.2. Depthwise Separable Convolution
- **Depthwise Conv**: mỗi kênh đầu vào có 1 kernel 3×3 riêng → lọc không gian độc lập
- **Pointwise Conv (1×1)**: kết hợp thông tin giữa các kênh
- So sánh MACs: `ratio = 1/C_out + 1/k²` → giảm ~8–9× so với Conv chuẩn
- Hình minh họa: tách Conv 3×3 thành DWConv + PWConv

#### 3.2.3. Residual Connection (Skip Connection)
- Nguồn gốc từ ResNet (He et al., 2016)
- Vai trò: gradient truyền thẳng về các tầng đầu, chống vanishing gradient
- Điều kiện áp dụng: stride=1 và số kênh đầu vào = đầu ra
- Hình minh họa: luồng skip qua block

#### 3.2.4. Inverted Residual vs Bottleneck thường
- ResNet: wide → narrow → wide (skip nối lớp rộng)
- MobileNetV2: **narrow → wide → narrow** (skip nối lớp hẹp)
- Lý do: Depthwise Conv hiệu quả hơn ở không gian nhiều kênh
- Hình so sánh hai kiểu bottleneck cạnh nhau

#### 3.2.5. Linear Bottleneck – Lý thuyết Manifold
- Giả thuyết: thông tin hữu ích nằm trên manifold chiều thấp
- ReLU phá hủy thông tin khi chiều thấp (không thể phục hồi giá trị âm bị set về 0)
- Giải pháp: bỏ ReLU ở tầng Projection cuối mỗi block
- Minh họa thực nghiệm từ paper gốc: so sánh có/không Linear Bottleneck

#### 3.2.6. Hàm kích hoạt ReLU6
- Định nghĩa: `ReLU6(x) = min(max(0, x), 6)`
- Lý do giới hạn ở 6: ổn định khi quantize sang INT8, tránh giá trị quá lớn
- Đồ thị so sánh ReLU vs ReLU6

#### 3.2.7. Batch Normalization
- Chuẩn hóa output sau mỗi Conv → ổn định quá trình huấn luyện
- Cho phép dùng learning rate cao hơn, hội tụ nhanh hơn
- Trong MobileNetV2: áp dụng sau mỗi Conv, trừ tầng Project cuối block

---

### 3.3. Kiến trúc chi tiết

#### 3.3.1. Sơ đồ tổng thể
- Hình: sơ đồ toàn bộ mạng từ input đến output
- Luồng: Input → Conv2D → [IRB × 17] → Conv2D 1×1 → GAP → Dropout → FC → Softmax

#### 3.3.2. Bảng cấu hình chính thức (từ paper gốc)

| Tầng | Loại | t | c | n | s |
|------|------|---|---|---|---|
| conv2d | – | – | 32 | 1 | 2 |
| bottleneck | IRB | 1 | 16 | 1 | 1 |
| bottleneck | IRB | 6 | 24 | 2 | 2 |
| bottleneck | IRB | 6 | 32 | 3 | 2 |
| bottleneck | IRB | 6 | 64 | 4 | 2 |
| bottleneck | IRB | 6 | 96 | 3 | 1 |
| bottleneck | IRB | 6 | 160 | 3 | 2 |
| bottleneck | IRB | 6 | 320 | 1 | 1 |
| conv2d 1×1 | – | – | 1280 | 1 | 1 |
| avgpool 7×7 | – | – | – | 1 | – |
| FC | – | – | k | 1 | – |

*t = expansion factor, c = output channels, n = số lần lặp, s = stride*

#### 3.3.3. Inverted Residual Block – Giải phẫu chi tiết
- **Bước 1 – Expand**: 1×1 Conv, kênh × t, BN + ReLU6
- **Bước 2 – Depthwise**: 3×3 DWConv, stride s, BN + ReLU6
- **Bước 3 – Project**: 1×1 Conv, thu về c kênh, BN (không activation)
- Khi nào có skip: stride=1 AND input_channels = output_channels
- Khi t=1 (block đầu tiên): bỏ bước Expand
- Hình minh họa: sơ đồ luồng dữ liệu trong 1 IRB, so sánh có/không skip

#### 3.3.4. Phân tích feature map qua từng giai đoạn

| Giai đoạn | Tầng | Feature map | Kênh |
|---|---|---|---|
| Input | – | 224 × 224 | 3 |
| Conv đầu | stride 2 | 112 × 112 | 32 |
| IRB nhóm 1 | stride 1 | 112 × 112 | 16 |
| IRB nhóm 2 | stride 2 | 56 × 56 | 24 |
| IRB nhóm 3 | stride 2 | 28 × 28 | 32 |
| IRB nhóm 4 | stride 2 | 14 × 14 | 64 |
| IRB nhóm 5 | stride 1 | 14 × 14 | 96 |
| IRB nhóm 6 | stride 2 | 7 × 7 | 160 |
| IRB nhóm 7 | stride 1 | 7 × 7 | 320 |
| Conv cuối 1×1 | stride 1 | 7 × 7 | 1280 |
| GAP | – | 1 × 1 | 1280 |

#### 3.3.5. Width Multiplier (α) và Resolution Multiplier (ρ)
- α ∈ (0, 1]: thu nhỏ số kênh toàn mạng → giảm params và MACs theo α²
- ρ: giảm độ phân giải đầu vào → giảm MACs theo ρ²
- Bảng trade-off accuracy vs latency:

| α | Input | Params | MACs | Top-1 |
|---|---|---|---|---|
| 0.35 | 224 | 1.66M | 59M | 60.3% |
| 0.50 | 224 | 1.97M | 97M | 65.4% |
| 0.75 | 224 | 2.61M | 209M | 69.8% |
| 1.0 | 224 | 3.4M | 300M | 72.0% |
| 1.4 | 224 | 6.06M | 582M | 74.7% |

---

### 3.4. Phân tích độ phức tạp tính toán

#### 3.4.1. Phân tích MACs toàn mạng
- Tổng: ~300M MACs (α=1.0, input 224×224)
- So sánh: MobileNetV1 ~569M, ResNet-50 ~4.1G, VGG-16 ~15.5G
- Biểu đồ so sánh MACs vs accuracy các mô hình

#### 3.4.2. Phân tích MACs trong 1 IRB
- Với input H×W×c, expansion t, output c':
  - Expand (1×1): `H × W × c × tc`
  - Depthwise (3×3): `H × W × tc × 9`
  - Project (1×1): `H × W × tc × c'`
- Tổng 1 IRB: `H·W·tc·(c + 9 + c')`
- So sánh với Conv 3×3 chuẩn: `H·W·9·c·c'`

#### 3.4.3. Phân bổ tham số
- Biểu đồ tròn: % tham số ở Conv đầu / các IRB / Conv cuối / FC
- IRB chiếm ~80% tổng tham số
- FC chỉ chiếm ~5% → GAP thay Flatten giúp tiết kiệm đáng kể

#### 3.4.4. Memory footprint
- Activation memory lớn nhất tại tầng Expand (tc kênh)
- Nhưng skip connection chỉ cần lưu tensor hẹp (c kênh) → tiết kiệm hơn ResNet
- Bảng so sánh memory activation qua từng giai đoạn

---

### 3.5. Huấn luyện MobileNetV2 cho bài toán phân loại biển báo

#### 3.5.1. Chiến lược Transfer Learning
- Tải pretrained weights từ ImageNet (đã học đặc trưng hình ảnh tổng quát)
- Thay lớp FC cuối: 1280 → số lớp biển báo
- **Giai đoạn 1**: Freeze toàn bộ backbone, chỉ train FC → hội tụ nhanh
- **Giai đoạn 2**: Unfreeze dần từ cuối lên, fine-tune toàn mạng → tối ưu sâu hơn
- Lý do dùng transfer learning: dataset biển báo nhỏ hơn ImageNet nhiều lần

#### 3.5.2. Cấu hình huấn luyện

| Hyperparameter | Giai đoạn 1 (Freeze) | Giai đoạn 2 (Fine-tune) |
|---|---|---|
| **Optimizer** | Adam | Adam |
| **Learning rate** | 1e-3 | 1e-4 |
| **Batch size** | 32 | 32 |
| **Epochs** | 10–20 | 30–50 |
| **Dropout** | 0.2 | 0.2 |
| **Weight decay** | 1e-4 | 1e-4 |
| **LR Scheduler** | – | StepLR / CosineAnnealing |

#### 3.5.3. Code triển khai

```python
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Load pretrained
model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

# Thay classifier
num_classes = 43  # GTSRB có 43 lớp biển báo
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# Giai đoạn 1: Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
```

```python
# Giai đoạn 2: Unfreeze fine-tune
for param in model.features.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

#### 3.5.4. Vòng lặp huấn luyện

```python
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)
```

---

### 3.6. Kết quả thực nghiệm

#### 3.6.1. Đồ thị quá trình huấn luyện
- **Hình 3.1**: Loss curve – train loss vs validation loss theo epoch
- **Hình 3.2**: Accuracy curve – train accuracy vs validation accuracy theo epoch
- Nhận xét: mô hình hội tụ ở epoch thứ bao nhiêu, có overfit không, validation loss có tăng trở lại không

#### 3.6.2. Đánh giá trên tập Test

| Metric | Giá trị |
|---|---|
| **Accuracy** | – % |
| **Precision (macro avg)** | – % |
| **Recall (macro avg)** | – % |
| **F1-score (macro avg)** | – % |
| **Loss (test)** | – |
| **Inference time / ảnh (CPU)** | – ms |
| **Inference time / ảnh (GPU)** | – ms |

#### 3.6.3. Báo cáo chi tiết theo từng lớp (Classification Report)

```
                    precision  recall  f1-score  support
Biển báo cấm...        0.xx    0.xx     0.xx      xxx
Biển báo nguy...       0.xx    0.xx     0.xx      xxx
...
macro avg              0.xx    0.xx     0.xx      xxx
weighted avg           0.xx    0.xx     0.xx      xxx
```

- Nhận xét: lớp nào có F1 thấp nhất, lý do tại sao

#### 3.6.4. Confusion Matrix
- **Hình 3.3**: Ma trận nhầm lẫn trên toàn bộ tập test
- Highlight các cặp lớp hay nhầm nhau nhất
- Nhận xét: biển báo nào dễ nhầm, lý do (hình dạng tương tự, màu sắc giống nhau...)

#### 3.6.5. Grad-CAM Visualization
- **Hình 3.4**: Heatmap Grad-CAM trên một số ảnh biển báo
- Ví dụ dự đoán đúng: mô hình nhìn vào đúng vùng biển báo
- Ví dụ dự đoán sai: mô hình bị phân tâm bởi nền, ánh sáng...
- Nhận xét: mô hình có học đúng đặc trưng của biển báo không

#### 3.6.6. Một số ví dụ dự đoán minh họa
- **Hình 3.5**: Grid ảnh – dự đoán đúng (nhãn thật vs nhãn dự đoán + confidence)
- **Hình 3.6**: Grid ảnh – dự đoán sai (nhãn thật vs nhãn dự đoán + confidence)
- Phân tích nguyên nhân các trường hợp sai: ảnh mờ, góc nghiêng, che khuất

---

### 3.7. Phân tích và Ablation Study

#### 3.7.1. Ảnh hưởng của Transfer Learning
- So sánh: train từ đầu (random init) vs dùng pretrained ImageNet
- Bảng kết quả: accuracy, thời gian hội tụ, số epoch cần thiết

| Chiến lược | Accuracy | Epochs hội tụ |
|---|---|---|
| Train từ đầu | – % | – |
| Freeze + Fine-tune | – % | – |
| Fine-tune toàn bộ | – % | – |

#### 3.7.2. Ảnh hưởng của Learning Rate
- Thử các learning rate khác nhau: 1e-2, 1e-3, 1e-4
- Đồ thị loss curve theo từng LR
- Nhận xét: LR nào cho kết quả tốt nhất, tại sao

#### 3.7.3. Ảnh hưởng của Data Augmentation
- So sánh: không augment vs có augment
- Nhận xét: augmentation giúp giảm overfit như thế nào trên dataset biển báo

---

### 3.8. So sánh với mô hình khác

| | **MobileNetV2** | Mô hình 2 | Mô hình 3 |
|---|---|---|---|
| **Accuracy (Test)** | – % | – % | – % |
| **Precision (macro)** | – % | – % | – % |
| **Recall (macro)** | – % | – % | – % |
| **F1-score (macro)** | – % | – % | – % |
| **Số tham số** | ~3.4M | – | – |
| **Thời gian train / epoch** | – s | – s | – s |
| **Inference time / ảnh** | – ms | – ms | – ms |
| **Kích thước model** | ~14MB | – | – |

- **Hình 3.7**: Biểu đồ cột so sánh accuracy các mô hình
- **Hình 3.8**: Biểu đồ so sánh accuracy vs inference time (scatter plot)
- Nhận xét: MobileNetV2 đứng ở đâu trong trade-off accuracy vs tốc độ

---

### 3.9. Ưu điểm, hạn chế và hướng cải thiện

#### 3.9.1. Ưu điểm trong bài toán biển báo
- Nhẹ (~14MB), phù hợp chạy trên dashcam, camera giao thông nhúng
- Pretrained ImageNet giúp hội tụ nhanh dù dataset biển báo nhỏ
- Dễ fine-tuning, ít tốn tài nguyên huấn luyện
- Kết quả ổn định, ít overfit nhờ Batch Normalization và Dropout

#### 3.9.2. Hạn chế
- Độ chính xác thấp hơn các mô hình lớn hơn (ResNet-50, EfficientNet)
- Khó phân biệt biển báo có hình dạng tương tự, chỉ khác màu hoặc ký hiệu nhỏ
- Nhạy cảm với ảnh chất lượng thấp, mờ, góc chụp bất thường
- Depthwise Conv kém hiệu quả trên GPU do khó parallelize

#### 3.9.3. Hướng cải thiện
- Tăng cường augmentation đặc thù: giả lập che khuất, ảnh đêm, mưa
- Kết hợp Attention (CBAM / SE block) vào backbone để tập trung vào biển báo
- Thử MobileNetV3 hoặc EfficientNet-B0 để so sánh thêm
- Áp dụng Knowledge Distillation từ mô hình lớn hơn để cải thiện accuracy

#### 3.9.4. Các mô hình kế thừa MobileNetV2
- **MobileNetV3**: NAS + Hard-Swish + SE block, nhanh hơn ~25%
- **EfficientNet**: compound scaling đồng thời depth/width/resolution
- **MobileViT**: kết hợp Transformer, hiểu context toàn cục tốt hơn
- **GhostNet**: Ghost module tái sử dụng feature map, giảm thêm tham số

---

## Tài liệu tham khảo

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). **MobileNetV2: Inverted residuals and linear bottlenecks.** *CVPR 2018.* [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)
2. Howard, A. G., et al. (2017). **MobileNets: Efficient convolutional neural networks for mobile vision applications.** [arXiv:1704.04861](https://arxiv.org/abs/1704.04861)
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep residual learning for image recognition.** *CVPR 2016.*
4. Howard, A., et al. (2019). **Searching for MobileNetV3.** *ICCV 2019.*
5. Tan, M., & Le, Q. V. (2019). **EfficientNet: Rethinking model scaling for CNNs.** *ICML 2019.*
6. Stallkamp, J., et al. (2012). **Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition.** *Neural Networks.*
7. Selvaraju, R. R., et al. (2017). **Grad-CAM: Visual explanations from deep networks via gradient-based localization.** *ICCV 2017.*
8. Ioffe, S., & Szegedy, C. (2015). **Batch normalization: Accelerating deep network training.** *ICML 2015.*

---

*Ghi chú:*
- *Điền kết quả thực nghiệm vào các ô (–) sau khi train xong.*
- *Mỗi hình cần có tiêu đề bên dưới: "Hình 3.X: Mô tả".*
- *Mỗi bảng cần có tiêu đề bên trên: "Bảng 3.X: Mô tả".*
- *Trích dẫn theo chuẩn IEEE: [1], [2],... sau mỗi luận điểm lấy từ tài liệu.*
