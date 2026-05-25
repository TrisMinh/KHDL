# Dàn ý Báo cáo: MobileNetV2 – Kiến trúc, Lý thuyết & Triển khai

---

## 1. Giới thiệu tổng quan

### 1.1. Bối cảnh và động lực nghiên cứu
- Thách thức triển khai Deep Learning trên thiết bị di động / edge
- Hạn chế về bộ nhớ, tính toán, năng lượng
- Nhu cầu mô hình nhẹ nhưng vẫn chính xác cao

### 1.2. Lịch sử phát triển MobileNet
- MobileNetV1 (2017) – Howard et al., Google
- MobileNetV2 (2018) – Sandler et al., Google
- MobileNetV3 (2019) – So sánh ngắn để đặt V2 vào bức tranh toàn cảnh

### 1.3. Đóng góp chính của MobileNetV2
- Inverted Residual Block
- Linear Bottleneck
- Hiệu quả vượt trội so với V1 và các mô hình nặng cùng thời

### 1.4. Ứng dụng thực tế
- Phân loại ảnh, phát hiện đối tượng (SSD), phân đoạn ngữ nghĩa (DeepLabV3)
- Nhận dạng khuôn mặt, phân tích video thời gian thực trên điện thoại

---

## 2. Nền tảng lý thuyết

### 2.1. Tích chập truyền thống (Standard Convolution)
- Công thức tính số phép nhân-cộng (MACs)
- Ví dụ: kernel 3×3, input H×W×C_in → output H×W×C_out
- Nhược điểm: chi phí tính toán O(k² · C_in · C_out · H · W)

### 2.2. Depthwise Separable Convolution
- **Depthwise Convolution**: lọc không gian riêng từng kênh
  - Mỗi kênh đầu vào có 1 kernel 3×3 riêng
  - Output: H × W × C (cùng số kênh)
- **Pointwise Convolution (1×1 Conv)**: kết hợp thông tin kênh
  - Kernel 1×1 × C_in → C_out
- So sánh MACs: giảm ~8–9× so với Conv chuẩn
- Công thức: `MACs_ratio = 1/C_out + 1/k²`

### 2.3. Residual Connection (Skip Connection)
- Nguồn gốc từ ResNet (He et al., 2016)
- Vai trò: chống vanishing gradient, cho phép mạng rất sâu
- Điều kiện áp dụng skip: stride=1 và số kênh đầu vào = đầu ra

### 2.4. Bottleneck trong ResNet vs MobileNetV2
- ResNet: wide → narrow → wide (thu hẹp giữa)
- MobileNetV2: **narrow → wide → narrow** (mở rộng giữa = Inverted)
- Lý do: Depthwise hoạt động hiệu quả hơn ở không gian chiều cao (nhiều kênh)

### 2.5. Linear Bottleneck – Lý thuyết Manifold
- Giả thuyết: thông tin hữu ích nằm trên manifold chiều thấp
- ReLU phá hủy thông tin khi chiều quá thấp (set về 0)
- Giải pháp: bỏ ReLU ở tầng Projection (1×1 cuối block)
- Bằng chứng thực nghiệm từ paper gốc

### 2.6. Hàm kích hoạt ReLU6
- Định nghĩa: `ReLU6(x) = min(max(0, x), 6)`
- Lý do giới hạn ở 6: ổn định khi quantize sang INT8
- So sánh với ReLU, Swish, Hard-Swish

### 2.7. Batch Normalization
- Vai trò trong MobileNetV2: chuẩn hóa sau mỗi Conv
- Tương tác với Linear Bottleneck (không BN trước skip add?)

---

## 3. Kiến trúc MobileNetV2 chi tiết

### 3.1. Tổng quan kiến trúc
- Sơ đồ toàn bộ mạng (bảng cấu hình chính thức từ paper)
- Input: 224×224×3
- Output: 1000 lớp (ImageNet)

### 3.2. Bảng cấu hình chi tiết (từ paper gốc)

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
| conv2d 1×1 / FC | – | – | k | 1 | – |

*t = expansion factor, c = output channels, n = số lần lặp, s = stride*

### 3.3. Inverted Residual Block (IRB) – Giải phẫu chi tiết
- **Bước 1 – Expand**: 1×1 Conv, kênh × t, BN + ReLU6
- **Bước 2 – Depthwise**: 3×3 DWConv, BN + ReLU6
- **Bước 3 – Project**: 1×1 Conv, thu về c kênh, BN (không activation)
- Khi nào có skip: stride=1 AND input_channels = output_channels
- Khi t=1 (block đầu tiên): bỏ bước Expand

### 3.4. Phân tích từng giai đoạn của mạng
- **Giai đoạn 1** (conv đầu): trích xuất đặc trưng cơ bản, stride 2
- **Giai đoạn 2–8** (các nhóm IRB): tăng dần độ sâu đặc trưng, giảm spatial
- **Giai đoạn cuối** (Conv 1×1 + GAP + FC): phân loại
- Phân tích feature map size qua từng giai đoạn

### 3.5. Width Multiplier (α) và Resolution Multiplier (ρ)
- Width multiplier α ∈ (0, 1]: thu nhỏ số kênh toàn mạng
- Resolution multiplier ρ: giảm độ phân giải đầu vào
- Bảng trade-off: accuracy vs latency theo α
- Công thức tính MACs theo α và ρ

### 3.6. So sánh với các kiến trúc khác
- MobileNetV1 vs V2: Depthwise same, khác ở residual & linear
- ShuffleNet, SqueezeNet, EfficientNet-B0
- Bảng so sánh: params, MACs, Top-1, latency trên Pixel phone

---

## 4. Phân tích độ phức tạp tính toán

### 4.1. Phân tích MACs (Multiply-Accumulate Operations)
- Tính MACs từng tầng
- Tổng: ~300M MACs (α=1.0, 224×224)
- So sánh MobileNetV1: ~569M MACs

### 4.2. Số lượng tham số
- Tổng: ~3.4M parameters
- Phân bổ: bao nhiêu % ở các IRB, bao nhiêu ở FC
- Biểu đồ phân bổ tham số

### 4.3. Bottleneck phân tích chi tiết theo IRB
- Với t=6, input c kênh: expand lên 6c, depthwise 6c, project về c'
- Tính MACs cho 1 IRB: `MACs = H·W·(t·c·c_in + t·c·9 + c_out·t·c)`
- Phân tích tỷ trọng từng bước trong block

### 4.4. Memory footprint
- Bộ nhớ activation qua từng giai đoạn
- Tại sao IRB tiết kiệm memory hơn bottleneck thường

---

## 5. Huấn luyện mô hình

### 5.1. Bộ dữ liệu
- ImageNet ILSVRC 2012: 1.28M train, 50K val, 1000 lớp
- CIFAR-10/100, Oxford Flowers, Food-101 (fine-tuning)

### 5.2. Cấu hình huấn luyện gốc (paper)
- Optimizer: RMSProp, decay 0.9, momentum 0.9
- Learning rate: 0.045, decay mỗi 2.5 epoch với factor 0.98
- Batch size: 96
- Weight decay: 4×10⁻⁵
- Dropout: 0.2 trước FC
- Epochs: 400

### 5.3. Data Augmentation
- Random crop, horizontal flip
- Color jitter, random erasing
- MixUp / CutMix (nếu fine-tuning hiện đại)

### 5.4. Transfer Learning & Fine-tuning
- Tải pretrained weights (ImageNet)
- Chiến lược: freeze backbone, fine-tune classifier
- Chiến lược: unfreeze dần từ cuối lên đầu
- Learning rate schedule cho fine-tuning

### 5.5. Knowledge Distillation
- Dùng teacher model lớn (ResNet-50) → student MobileNetV2
- Soft label loss + hard label loss
- Kết quả cải thiện Top-1 thêm ~1–2%

---

## 6. Quantization và tối ưu hóa triển khai

### 6.1. Post-Training Quantization (PTQ)
- INT8 quantization: weight + activation
- Tại sao ReLU6 thuận lợi cho quantize
- Công cụ: TensorFlow Lite, PyTorch, ONNX Runtime

### 6.2. Quantization-Aware Training (QAT)
- Giả lập quantize trong forward pass lúc train
- Cải thiện độ chính xác sau quantize
- So sánh PTQ vs QAT: accuracy drop

### 6.3. Pruning
- Unstructured pruning: loại bỏ weight nhỏ
- Structured pruning: loại kênh/filter toàn bộ
- Magnitude-based, gradient-based pruning

### 6.4. Model Conversion
- PyTorch → ONNX → TensorRT
- TensorFlow → TFLite
- Keras → CoreML (iOS)

### 6.5. Benchmarking trên thiết bị thực
- Latency trên Pixel 1 Phone (CPU): ~75ms
- So sánh latency với V1, ShuffleNet
- Power consumption

---

## 7. Triển khai thực tế (Implementation)

### 7.1. Triển khai từ đầu bằng PyTorch

```python
import torch
import torch.nn as nn

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, stride=1, groups=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, 3, stride, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, stride, expand_ratio):
        super().__init__()
        self.use_skip = stride == 1 and in_c == out_c
        hidden = int(in_c * expand_ratio)
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_c, hidden, stride=1))  # Expand
        layers += [
            ConvBNReLU(hidden, hidden, stride=stride, groups=hidden),  # DWConv
            nn.Conv2d(hidden, out_c, 1, bias=False),  # Project
            nn.BatchNorm2d(out_c),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_skip else self.conv(x)
```

### 7.2. Sử dụng pretrained model (torchvision)

```python
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()
```

### 7.3. Fine-tuning cho bài toán tùy chỉnh

```python
# Thay thế classifier
num_classes = 10
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False
```

### 7.4. Inference pipeline

```python
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img = Image.open("image.jpg")
x = transform(img).unsqueeze(0)
with torch.no_grad():
    logits = model(x)
    pred = logits.argmax(dim=1)
```

### 7.5. Export sang TFLite

```python
# TensorFlow
import tensorflow as tf
model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),
                                           weights='imagenet')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # INT8 PTQ
tflite_model = converter.convert()
with open('mobilenetv2.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 7.6. Triển khai trên Android / iOS
- Android: TFLite + ML Kit / NNAPI acceleration
- iOS: CoreML conversion với coremltools
- Flutter: tích hợp tflite_flutter

---

## 8. Thực nghiệm và đánh giá

### 8.1. Đánh giá trên ImageNet
- Top-1: 72.0%, Top-5: 90.6% (α=1.0, 224×224)
- Bảng kết quả theo width multiplier α = {0.35, 0.5, 0.75, 1.0, 1.4}
- Bảng kết quả theo input resolution {96, 128, 160, 192, 224}

### 8.2. Ứng dụng Detection: SSDLite + MobileNetV2
- Thay thế backbone VGG trong SSD bằng MobileNetV2
- SSDLite: depthwise separable predictor heads
- Kết quả COCO: 22 mAP, 75ms trên Pixel phone
- So sánh với YOLOv3-tiny

### 8.3. Ứng dụng Segmentation: DeepLabV3+
- MobileNetV2 làm encoder trong DeepLabV3+
- Kết quả PASCAL VOC: ~75% mIoU
- So sánh tốc độ vs độ chính xác với ResNet-101 backbone

### 8.4. Phân tích lỗi (Error Analysis)
- Confusion matrix các lớp dễ nhầm
- Grad-CAM visualization: mô hình "nhìn" vào đâu
- Các loại ảnh khó: ảnh nhỏ, góc bất thường, nhiều đối tượng

### 8.5. Ablation Study
- Bỏ Linear Bottleneck → accuracy giảm bao nhiêu?
- Bỏ Skip Connection → accuracy giảm bao nhiêu?
- Thay ReLU6 bằng ReLU → ảnh hưởng quantize

---

## 9. Ưu điểm, hạn chế và hướng phát triển

### 9.1. Ưu điểm
- Rất nhẹ (~3.4M params), chạy tốt trên CPU mobile
- Hiệu quả tính toán cao nhờ Depthwise + Inverted Residual
- Dễ quantize sang INT8 nhờ ReLU6
- Backbone đa năng cho detection, segmentation

### 9.2. Hạn chế
- Độ chính xác thấp hơn EfficientNet-B0 với cùng FLOPs
- Depthwise Conv không được tối ưu tốt trên GPU (kém hiệu quả với cuDNN)
- Khó parallelize hơn Conv chuẩn trên một số phần cứng
- Không có attention mechanism → kém trên bài toán cần context dài

### 9.3. Các cải tiến và hậu duệ
- **MobileNetV3**: Neural Architecture Search (NAS) + Hard-Swish + SE block
- **EfficientNet**: compound scaling (depth + width + resolution)
- **MobileViT**: kết hợp Transformer vào MobileNet
- **GhostNet**: Ghost module thay thế Conv chuẩn

### 9.4. Hướng nghiên cứu tương lai
- NAS tự động tìm kiến trúc mobile
- Knowledge Distillation từ Vision Transformer
- Hardware-aware design cho NPU chuyên dụng
- Federated learning với mô hình nhẹ

---

## 10. Kết luận

### 10.1. Tóm tắt đóng góp
- Inverted Residual + Linear Bottleneck là đóng góp cốt lõi
- Cân bằng tốt giữa accuracy, speed, memory

### 10.2. Ý nghĩa thực tiễn
- Mở đường cho AI trên thiết bị đầu cuối (on-device AI)
- Nền tảng cho hàng loạt ứng dụng mobile thời gian thực

### 10.3. Bài học thiết kế
- Hiệu quả tính toán phải được tính từ giai đoạn thiết kế
- Linear bottleneck: đơn giản nhưng có nền tảng lý thuyết vững chắc

---

## Tài liệu tham khảo

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). **MobileNetV2: Inverted residuals and linear bottlenecks.** *CVPR 2018.* [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)
2. Howard, A. G., et al. (2017). **MobileNets: Efficient convolutional neural networks for mobile vision applications.** [arXiv:1704.04861](https://arxiv.org/abs/1704.04861)
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep residual learning for image recognition.** *CVPR 2016.*
4. Howard, A., et al. (2019). **Searching for MobileNetV3.** *ICCV 2019.*
5. Tan, M., & Le, Q. V. (2019). **EfficientNet: Rethinking model scaling for CNNs.** *ICML 2019.*
6. Liu, W., et al. (2016). **SSD: Single shot multibox detector.** *ECCV 2016.*
7. Chen, L. C., et al. (2018). **Encoder-Decoder with Atrous Separable Convolution for Semantic Segmentation.** *ECCV 2018.*

---

*Ghi chú: Mỗi mục nên có ít nhất 1 hình minh họa, bảng số liệu, hoặc đoạn code. Phần lý thuyết (2, 3, 4) cần công thức toán học rõ ràng. Phần triển khai (7) cần code chạy được và kết quả thực nghiệm cụ thể.*
