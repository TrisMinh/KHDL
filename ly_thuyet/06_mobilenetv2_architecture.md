# Phần 6: Kiến trúc MobileNetV2 trong notebook

## 1. Mục tiêu

Phần này giải thích sâu Cell 6 trong notebook, nơi định nghĩa:

```python
ConvBNReLU6
InvertedResidual
MobileNetV2
```

Mục tiêu là hiểu:

```text
CNN là gì
convolution làm gì
depthwise separable convolution tiết kiệm thế nào
inverted residual là gì
linear bottleneck là gì
skip connection giúp gì
stride ảnh hưởng feature map và VRAM ra sao
classifier cuối tạo output class như thế nào
```

Notebook xây dựng MobileNetV2 từ đầu, không dùng model pretrained.

## 2. CNN nhìn ảnh như thế nào?

CNN học đặc trưng bằng convolution kernel.

Ảnh RGB đầu vào:

```text
[3, 224, 224]
```

Các layer đầu thường học:

```text
cạnh
đường thẳng
góc
màu cơ bản
```

Layer sâu hơn học:

```text
hình tròn biển cấm
viền đỏ
nền xanh
chữ số tốc độ
ký hiệu công trường
```

Cuối cùng classifier dùng đặc trưng đó để dự đoán class.

## 3. Convolution thường

Convolution dùng kernel trượt trên ảnh.

Với kernel `3x3`, mỗi output pixel được tính từ vùng `3x3` quanh vị trí tương ứng.

Nếu input có `Cin` kênh, output có `Cout` kênh, kernel `KxK`, số tham số:

```text
Params_regular = K * K * Cin * Cout
```

Ví dụ:

```text
K = 3
Cin = 32
Cout = 64
```

Số tham số:

```text
3 * 3 * 32 * 64 = 18,432
```

Convolution thường mạnh nhưng tốn tham số và tính toán.

## 4. Ý tưởng MobileNet

MobileNet được thiết kế để nhẹ:

```text
ít tham số
ít tính toán
phù hợp thiết bị hạn chế
vẫn đủ tốt cho ảnh
```

MobileNetV2 dùng:

```text
depthwise separable convolution
inverted residual
linear bottleneck
```

## 5. Depthwise separable convolution

Thay vì convolution thường, MobileNet tách thành hai bước:

```text
1. Depthwise convolution
2. Pointwise convolution 1x1
```

### 5.1. Depthwise convolution

Depthwise convolution áp dụng một kernel riêng cho từng input channel.

Số tham số:

```text
Params_depthwise = K * K * Cin
```

Nó học đặc trưng không gian trong từng kênh, nhưng chưa trộn thông tin giữa các kênh.

### 5.2. Pointwise convolution

Pointwise convolution là convolution `1x1`.

Số tham số:

```text
Params_pointwise = Cin * Cout
```

Nó trộn thông tin giữa các channel.

### 5.3. Tổng tham số

```text
Params_separable = K*K*Cin + Cin*Cout
```

So với convolution thường:

```text
Params_regular = K*K*Cin*Cout
```

Tỷ lệ:

```text
Params_separable / Params_regular
= 1/Cout + 1/(K*K)
```

Với `K=3`, `Cout=64`:

```text
1/64 + 1/9 ≈ 0.1267
```

Tức chỉ khoảng 12.7% tham số của convolution thường.

## 6. `ConvBNReLU6`

Notebook định nghĩa:

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

Block gồm:

```text
Conv2d -> BatchNorm2d -> ReLU6
```

### 6.1. Vì sao `bias=False`

Convolution không dùng bias vì sau đó có BatchNorm.

BatchNorm đã có tham số shift:

```text
y = gamma * normalized_x + beta
```

`beta` đóng vai trò bias. Vì vậy bias trong Conv2d là thừa.

### 6.2. Vì sao có BatchNorm

BatchNorm chuẩn hóa activation trong batch, giúp:

```text
training ổn định hơn
hội tụ nhanh hơn
giảm nhạy với khởi tạo
```

### 6.3. Vì sao ReLU6

ReLU:

```text
max(0, x)
```

ReLU6:

```text
min(max(0, x), 6)
```

ReLU6 giới hạn activation trong `[0, 6]`, thường dùng trong MobileNet để ổn định và phù hợp triển khai thiết bị nhẹ.

## 7. Inverted residual block

Notebook định nghĩa:

```python
class InvertedResidual(nn.Module):
```

Block này là lõi của MobileNetV2.

Cấu trúc:

```text
input hẹp
-> expand rộng bằng 1x1 conv
-> depthwise 3x3 conv
-> project hẹp bằng 1x1 conv tuyến tính
-> optional skip connection
```

## 8. Vì sao gọi là inverted residual?

Trong ResNet truyền thống, residual block thường:

```text
wide -> narrow -> wide
```

MobileNetV2 làm ngược:

```text
narrow -> wide -> narrow
```

Vì vậy gọi là inverted residual.

Bottleneck hẹp ở input/output, phần xử lý chính mở rộng channel ở giữa.

## 9. Expansion layer

Code:

```python
hidden_dim = int(round(in_channels * expand_ratio))
if expand_ratio != 1:
    layers.append(ConvBNReLU6(in_channels, hidden_dim, kernel_size=1))
```

Nếu:

```text
in_channels = 32
expand_ratio = 6
```

thì:

```text
hidden_dim = 192
```

Mục đích:

```text
mở rộng không gian channel để model học đặc trưng phong phú hơn
```

## 10. Depthwise layer

Code:

```python
layers.append(ConvBNReLU6(
    hidden_dim,
    hidden_dim,
    kernel_size=3,
    stride=stride,
    groups=hidden_dim
))
```

Trong PyTorch:

```python
groups=hidden_dim
```

khi `in_channels = out_channels = hidden_dim` nghĩa là depthwise convolution.

Mỗi channel được convolution riêng.

`stride` có thể là 1 hoặc 2:

```text
stride=1: giữ kích thước feature map
stride=2: giảm kích thước feature map
```

## 11. Projection layer

Code:

```python
layers.extend([
    nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
    nn.BatchNorm2d(out_channels),
])
```

Projection đưa channel từ `hidden_dim` về `out_channels`.

Điểm quan trọng:

```text
không có ReLU sau projection
```

Đây là linear bottleneck.

## 12. Linear bottleneck

Vì sao projection không ReLU?

Ở không gian hẹp, ReLU có thể làm mất thông tin do cắt giá trị âm về 0.

Nếu bottleneck ít chiều, mất thông tin khó phục hồi.

MobileNetV2 giữ projection tuyến tính:

```text
Conv1x1 + BatchNorm, không ReLU
```

để bảo toàn thông tin tốt hơn.

## 13. Skip connection

Code:

```python
self.use_skip = (stride == 1 and in_channels == out_channels)
```

Forward:

```python
if self.use_skip:
    return x + self.conv(x)
else:
    return self.conv(x)
```

Skip chỉ dùng khi input và output cùng shape.

Điều kiện:

```text
stride = 1
in_channels = out_channels
```

Nếu stride 2, kích thước feature map thay đổi, không cộng trực tiếp được.

## 14. Vì sao skip connection giúp train?

Block có skip:

```text
y = x + F(x)
```

Gradient:

```text
∂L/∂x = ∂L/∂y * (1 + ∂F/∂x)
```

Thành phần `1` giúp gradient đi thẳng qua block. Điều này giảm vanishing gradient và giúp mạng sâu train dễ hơn.

## 15. Cấu hình stage trong notebook

Notebook dùng:

```python
inverted_residual_setting = [
    [1, 16,  1, 1],
    [6, 24,  2, 2],
    [6, 32,  3, 2],
    [6, 64,  4, 2],
    [6, 96,  3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]
```

Mỗi dòng:

```text
[t, c, n, s]
```

Ý nghĩa:

| Ký hiệu | Ý nghĩa |
|---|---|
| `t` | expansion ratio |
| `c` | output channels |
| `n` | số block lặp lại |
| `s` | stride của block đầu trong stage |

Ví dụ:

```python
[6, 32, 3, 2]
```

nghĩa là:

```text
expand ratio = 6
output channel = 32
lặp 3 block
block đầu stride 2
các block sau stride 1
```

## 16. First convolution và OOM

Notebook dùng:

```python
first_stride = 2 if IMG_SIZE >= 160 else 1
features = [ConvBNReLU6(3, input_channels, kernel_size=3, stride=first_stride)]
```

Với ảnh `224x224`, `first_stride = 2`.

Lý do:

```text
stride=2 giảm feature map sớm
giảm VRAM
giảm tính toán
gần thiết kế MobileNetV2 gốc hơn
```

Nếu dùng stride 1 với ảnh 224x224, feature map lớn hơn, dễ CUDA OOM.

## 17. Feature map thay đổi kích thước thế nào?

Giả sử input:

```text
224 x 224
```

First conv stride 2:

```text
112 x 112
```

Các stage stride 2 tiếp tục giảm:

```text
112 -> 56 -> 28 -> 14 -> 7
```

Feature map nhỏ dần nhưng channel tăng dần.

Đây là pattern phổ biến trong CNN:

```text
spatial size giảm
channel depth tăng
```

## 18. `width_mult` và `_make_divisible`

Code:

```python
output_channels = _make_divisible(c * width_mult)
```

Nếu `width_mult = 1.0`, channel giữ gần như chuẩn.

Nếu `width_mult = 0.75`, channel giảm.

`_make_divisible` đảm bảo channel chia hết cho 8:

```python
new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
```

Lý do:

```text
tối ưu phần cứng
tránh số channel lẻ gây kém hiệu quả
```

## 19. Adaptive average pooling

Forward cuối:

```python
x = self.features(x)
x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
x = torch.flatten(x, 1)
x = self.classifier(x)
```

`adaptive_avg_pool2d(x, (1,1))` biến feature map:

```text
[B, C, H, W]
```

thành:

```text
[B, C, 1, 1]
```

Nó lấy trung bình mỗi channel trên toàn bộ không gian.

Sau flatten:

```text
[B, C]
```

## 20. Classifier

Code:

```python
self.classifier = nn.Sequential(
    nn.Dropout(p=dropout),
    nn.Linear(last_channels, num_classes),
)
```

`Linear` đưa vector feature về số class.

Nếu có 12 class:

```text
output shape = [B, 12]
```

Output này là logits, chưa phải xác suất.

Softmax dùng khi đánh giá/predict:

```python
probs = torch.softmax(outputs, dim=1)
```

## 21. Khởi tạo trọng số

Convolution:

```python
nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
```

Kaiming initialization phù hợp với ReLU/ReLU6.

BatchNorm:

```python
weight = 1
bias = 0
```

Linear:

```python
normal mean=0, std=0.01
```

Đây là train từ đầu. Không có pretrained weights.

## 22. Vì sao model không phải pretrained?

Notebook tạo model:

```python
model = MobileNetV2(
    num_classes=CONFIG['num_classes'],
    width_mult=CONFIG['width_mult'],
    dropout=CONFIG['dropout']
).to(device)
```

Không có:

```python
weights=...
pretrained=True
torchvision.models.mobilenet_v2(...)
```

Vì vậy trọng số ban đầu là random initialization.

## 23. Test forward pass

Notebook tạo dummy:

```python
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
out = model(dummy)
```

Mục đích:

```text
kiểm tra model forward được
kiểm tra output shape đúng
phát hiện lỗi dimension sớm
```

Nếu `NUM_CLASSES = 12`, output mong đợi:

```text
[1, 12]
```

## 24. Parameter count

Notebook đếm:

```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Ý nghĩa:

```text
total_params: tổng số trọng số
trainable_params: số trọng số được cập nhật khi train
```

Vì không freeze layer, hai số thường bằng nhau.

Model size FP32 xấp xỉ:

```text
total_params * 4 bytes
```

vì mỗi float32 có 4 byte.

## 25. Vì sao MobileNetV2 phù hợp bài này?

Bài toán:

```text
phân loại biển báo crop
12 class
ảnh 224x224
Colab GPU giới hạn
```

MobileNetV2 phù hợp vì:

```text
nhẹ hơn nhiều CNN lớn
train nhanh
ít tham số
vẫn học tốt đặc trưng ảnh
phù hợp khi muốn triển khai thực tế
```

## 26. Giới hạn của MobileNetV2

MobileNetV2 classifier không tự định vị biển báo.

Nó cũng có thể yếu nếu:

```text
ảnh ngoài khác xa train
biển báo bị che nhiều
ảnh quá mờ
crop sai
class rất giống nhau nhưng chi tiết nhỏ bị mất
```

Nếu muốn xử lý ảnh nguyên cảnh, cần detector.

## 27. Câu hỏi phản biện thường gặp

### 27.1. Vì sao không dùng ResNet?

ResNet mạnh nhưng thường nặng hơn. MobileNetV2 nhẹ hơn, phù hợp bài toán có ảnh crop và số class không quá nhiều.

### 27.2. Vì sao first stride là 2?

Với ảnh 224x224, stride 2 giảm feature map sớm, tiết kiệm VRAM và theo thiết kế gốc MobileNetV2. Nếu stride 1, GPU dễ OOM.

### 27.3. Vì sao projection không ReLU?

Đó là linear bottleneck của MobileNetV2. ReLU ở bottleneck hẹp có thể làm mất thông tin.

### 27.4. Vì sao output không softmax trong forward?

`CrossEntropyLoss` của PyTorch nhận logits trực tiếp và tự xử lý log-softmax bên trong. Nếu thêm softmax trước loss sẽ không đúng cách dùng chuẩn.

Softmax chỉ dùng khi muốn đọc xác suất lúc evaluation/predict.

## 28. Cách trình bày trong báo cáo

Có thể viết:

```text
Mô hình sử dụng kiến trúc MobileNetV2 được xây dựng từ đầu. Thành phần chính là các inverted residual block gồm expansion 1x1 convolution, depthwise 3x3 convolution và projection 1x1 convolution tuyến tính. Việc sử dụng depthwise separable convolution giúp giảm đáng kể số tham số và phép tính so với convolution thông thường. Skip connection được dùng khi input và output có cùng kích thước, giúp gradient truyền tốt hơn qua mạng sâu. Cuối mạng, adaptive average pooling và một fully connected layer được dùng để tạo logits cho các lớp biển báo.
```

## 29. Checklist hiểu kiến trúc

```text
[ ] Biết CNN dùng convolution để học đặc trưng ảnh
[ ] Biết convolution thường tốn K*K*Cin*Cout tham số
[ ] Biết depthwise separable conv tách depthwise và pointwise
[ ] Biết inverted residual là narrow-wide-narrow
[ ] Biết linear bottleneck không dùng ReLU ở projection
[ ] Biết skip chỉ dùng khi shape giống nhau
[ ] Biết stride 2 giảm feature map và VRAM
[ ] Biết classifier output logits, chưa softmax
[ ] Biết model hiện tại không pretrained
```

## 30. Kết luận phần 6

MobileNetV2 trong notebook là CNN nhẹ, phù hợp bài toán phân loại ảnh crop.

Ý tưởng cốt lõi:

```text
Depthwise separable convolution giảm tham số.
Inverted residual mở rộng channel để học đặc trưng rồi nén lại.
Linear bottleneck giữ thông tin ở không gian hẹp.
Skip connection giúp train mạng sâu.
Stride hợp lý giúp giảm VRAM.
Classifier cuối tạo logits cho từng class.
```

Hiểu kiến trúc này giúp giải thích vì sao model vừa nhẹ vừa đạt kết quả cao trên bài toán biển báo đã crop.
