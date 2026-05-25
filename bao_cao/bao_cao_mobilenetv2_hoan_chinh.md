# BÁO CÁO CHUYÊN ĐỀ

# KIẾN TRÚC MOBILENETV2 VÀ SE ATTENTION TRONG BÀI TOÁN PHÂN LOẠI BIỂN BÁO GIAO THÔNG

**Phạm vi báo cáo:** tập trung vào phần mô hình MobileNetV2, đặc biệt là kiến trúc, nguyên lý thiết kế, phân tích block, độ phức tạp tính toán, cách triển khai trong notebook và phiên bản cải tiến có SE Attention. Các phần thu thập dữ liệu, crawl dữ liệu và tiền xử lý chỉ được nhắc ở mức bối cảnh vì do phần khác của nhóm phụ trách.

---

## Mục lục

1. Phạm vi và vai trò của MobileNetV2 trong đề tài
2. Tổng quan MobileNetV2
3. Nền tảng: từ convolution thường đến depthwise separable convolution
4. Các thành phần kiến trúc chính
5. Inverted Residual Block, Linear Bottleneck và SE Attention
6. Kiến trúc MobileNetV2 đầy đủ
7. Phân tích tensor shape, tham số và MACs
8. Giải phẫu từng stage trong MobileNetV2 notebook
9. Triển khai MobileNetV2 từ đầu trong notebook
10. MobileNetV2 như bộ trích xuất đặc trưng cho biển báo giao thông
11. Huấn luyện và đánh giá mô hình MobileNetV2
12. Nhận xét, hạn chế và hướng cải thiện
13. Danh sách hình cần bổ sung cho phiên bản SE Attention
14. Kết luận
15. Tài liệu tham khảo

---

# 1. Phạm vi và vai trò của MobileNetV2 trong đề tài

## 1.1. Bài toán đang giải quyết

Trong đề tài phân loại biển báo giao thông, dữ liệu đầu vào cho mô hình là ảnh biển báo đã được crop vùng biển báo, đưa về cùng kích thước, chuyển RGB và chuẩn hóa. Mô hình không trực tiếp nhận ảnh đường phố nguyên cảnh để tự tìm vị trí biển báo. Nói cách khác, bài toán của phần mô hình là:

```text
Input:  ảnh biển báo đã crop
Output: nhãn lớp biển báo
```

Đây là bài toán **image classification**, không phải object detection. Nếu hệ thống cần hoạt động trên ảnh đường phố nguyên cảnh, cần thêm một detector ở trước, ví dụ YOLO hoặc SSD, để tìm bounding box biển báo rồi mới đưa vùng crop vào MobileNetV2.

![Bài toán classification](<../ly_thuyet/Ảnh info grafic/Phần 1/1.1 Bài toán classification.png>)

Trong pipeline tổng thể, MobileNetV2 nằm ở khâu mô hình hóa dữ liệu. Các bước thu thập, lọc ảnh, crop, resize, RGB, chia train/validation/test và augmentation là các bước chuẩn bị dữ liệu. Phần này chỉ sử dụng kết quả cuối cùng của pipeline dữ liệu để huấn luyện classifier.

## 1.2. Vì sao phần mô hình cần tập trung vào kiến trúc

Khi báo cáo về MobileNetV2, nếu chỉ ghi "dùng MobileNetV2 để phân loại ảnh" thì chưa đủ. Điểm quan trọng là phải giải thích vì sao kiến trúc này nhẹ nhưng vẫn hiệu quả. MobileNetV2 không chỉ là một CNN nhỏ; nó có các quyết định thiết kế rõ ràng:

- Thay convolution thường bằng depthwise separable convolution để giảm phép tính.
- Dùng bottleneck ngược, tức `narrow -> wide -> narrow`, thay vì bottleneck truyền thống.
- Dùng skip connection giữa các tensor hẹp để giảm bộ nhớ.
- Bỏ activation ở projection layer để tạo linear bottleneck.
- Dùng ReLU6 để thuận lợi cho tính toán thấp-bit và thiết bị di động.
- Dùng global average pooling thay vì flatten toàn bộ feature map để giảm tham số classifier.
- Ở phiên bản cải tiến v7, bổ sung SE Attention vào inverted residual block để mô hình tự học mức độ quan trọng của từng channel đặc trưng.

Những điểm này là trọng tâm của báo cáo.

## 1.3. Vị trí của MobileNetV2 trong pipeline đề tài

Pipeline rút gọn:

```text
SplitData/train, val, test
-> ImageFolder
-> Transform + Normalize
-> MobileNetV2 / MobileNetV2 + SE Attention
-> logits
-> CrossEntropyLoss
-> Accuracy / Precision / Recall / F1
```

![Kiến trúc tổng hệ thống](<../ly_thuyet/Ảnh info grafic/Phần 9/9.1. Hình kiến trúc tổng hệ thống.png>)

Các phần dữ liệu chỉ cần nêu ngắn:

- Dataset custom có hơn 10.000 ảnh.
- Ảnh đã được crop biển báo.
- Dữ liệu được chia train/validation/test.
- Train có thể dùng augmentation, validation/test giữ nguyên.

Từ đây trở đi, báo cáo tập trung vào kiến trúc MobileNetV2.

---

# 2. Tổng quan MobileNetV2

## 2.1. Lịch sử ngắn của dòng MobileNet

MobileNet là họ mô hình CNN được thiết kế cho môi trường hạn chế tài nguyên như điện thoại, thiết bị nhúng, camera thông minh và hệ thống edge AI. MobileNetV1 giới thiệu việc sử dụng depthwise separable convolution trong classification. MobileNetV2 kế thừa ý tưởng đó nhưng bổ sung hai đổi mới chính: **inverted residual** và **linear bottleneck**. MobileNetV3 sau đó tiếp tục tối ưu bằng neural architecture search, hard-swish và squeeze-and-excitation.

Trong phạm vi đề tài này, MobileNetV2 được chọn vì cân bằng tốt giữa:

- Độ chính xác.
- Số tham số.
- Chi phí tính toán.
- Tốc độ inference.
- Độ phù hợp với ảnh kích thước nhỏ và trung bình.

## 2.2. Thông số tham khảo của MobileNetV2 chuẩn

Theo paper gốc, MobileNetV2 bản chuẩn với width multiplier 1.0 và input 224x224 có khoảng 3.4 triệu tham số và khoảng 300 triệu multiply-adds. Trên ImageNet, mô hình đạt Top-1 khoảng 72.0%, tốt hơn MobileNetV1 trong khi nhẹ hơn về số phép tính.

| Mô hình | Top-1 ImageNet | Params | MAdds | CPU Pixel 1 |
|---|---:|---:|---:|---:|
| MobileNetV1 | 70.6% | 4.2M | 575M | 113 ms |
| ShuffleNet 1.5 | 71.5% | 3.4M | 292M | - |
| MobileNetV2 | 72.0% | 3.4M | 300M | 75 ms |
| MobileNetV2 1.4 | 74.7% | 6.9M | 585M | 143 ms |

Bảng này cho thấy MobileNetV2 không cố đạt độ chính xác cao nhất bằng mọi giá. Mục tiêu của nó là tối ưu trade-off giữa accuracy và cost. Điều này phù hợp với bài toán biển báo giao thông, nơi mô hình có thể cần chạy nhanh trên camera hoặc thiết bị nhúng.

## 2.3. MobileNetV2 khác gì so với CNN thông thường

CNN thông thường thường xếp nhiều convolution 3x3 và tăng dần số kênh qua các stage. Cách này mạnh nhưng tốn chi phí vì convolution thường trộn không gian và kênh trong cùng một phép toán.

MobileNetV2 tách bài toán thành hai phần:

```text
Lọc không gian: depthwise convolution
Trộn kênh: pointwise convolution 1x1
```

Sau đó, trong mỗi block, MobileNetV2 mở rộng số kênh ở giữa để tăng khả năng biểu diễn, rồi nén lại về tensor hẹp để tiết kiệm bộ nhớ và tạo shortcut.

![CNN nhìn ảnh như thế nào](<../ly_thuyet/Ảnh info grafic/Phần 4/4.1. Hình CNN nhìn ảnh như thế nào.png>)

---

# 3. Nền tảng: từ convolution thường đến depthwise separable convolution

## 3.1. Convolution thường

Convolution thường dùng một tập kernel để quét qua feature map. Với input có kích thước:

```text
H x W x C_in
```

kernel có kích thước:

```text
k x k
```

và output có `C_out` kênh, số tham số của convolution thường là:

```text
Params_regular = k x k x C_in x C_out
```

Số phép nhân-cộng xấp xỉ:

```text
MACs_regular = H x W x k x k x C_in x C_out
```

![Convolution thường](<../ly_thuyet/Ảnh info grafic/Phần 4/4.2. Hình convolution thường.png>)

Ví dụ:

```text
H = 48, W = 48
k = 3
C_in = 32
C_out = 64
```

Số tham số:

```text
3 x 3 x 32 x 64 = 18,432
```

Số MACs:

```text
48 x 48 x 3 x 3 x 32 x 64 = 42,467,328
```

Convolution thường mạnh vì mỗi output channel nhìn toàn bộ input channel. Tuy nhiên, chính vì trộn cả không gian lẫn kênh trong một phép toán, chi phí tăng rất nhanh khi số kênh lớn.

## 3.2. Depthwise convolution

Depthwise convolution xử lý từng kênh độc lập. Nếu input có `C_in` kênh, mỗi kênh có một kernel riêng. Số output channel thường bằng `C_in`.

Số tham số:

```text
Params_depthwise = k x k x C_in
```

Số MACs:

```text
MACs_depthwise = H x W x k x k x C_in
```

![Depthwise convolution](<../ly_thuyet/Ảnh info grafic/Phần 4/4.4. Hình depthwise convolution.png>)

Depthwise convolution học đặc trưng không gian như cạnh, góc, texture, hình dạng cục bộ. Nhưng nó không trộn thông tin giữa các kênh. Vì vậy, nếu chỉ dùng depthwise convolution, mô hình sẽ thiếu khả năng kết hợp đặc trưng giữa các channel.

## 3.3. Pointwise convolution 1x1

Pointwise convolution là convolution kernel 1x1. Nó không nhìn vùng lân cận theo không gian, nhưng trộn thông tin giữa các kênh.

Số tham số:

```text
Params_pointwise = C_in x C_out
```

Số MACs:

```text
MACs_pointwise = H x W x C_in x C_out
```

![Pointwise convolution](<../ly_thuyet/Ảnh info grafic/Phần 4/4.5. Hình pointwise convolution 1x1.png>)

Pointwise convolution đóng vai trò rất quan trọng trong MobileNet. Nó là nơi các kênh được kết hợp để tạo feature mới.

## 3.4. Depthwise separable convolution

Depthwise separable convolution ghép hai bước:

```text
Depthwise 3x3 -> Pointwise 1x1
```

Tổng tham số:

```text
Params_ds = k x k x C_in + C_in x C_out
```

Tổng MACs:

```text
MACs_ds = H x W x (k x k x C_in + C_in x C_out)
```

So với convolution thường:

```text
Params_ds / Params_regular
= (k x k x C_in + C_in x C_out) / (k x k x C_in x C_out)
= 1/C_out + 1/(k x k)
```

Với `k = 3` và `C_out = 64`:

```text
1/64 + 1/9 ≈ 0.1267
```

Tức depthwise separable convolution chỉ cần khoảng 12.7% số tham số của convolution thường trong ví dụ này.

![Regular conv vs depthwise separable conv](<../ly_thuyet/Ảnh info grafic/Phần 4/4.3. Hình regular conv vs depthwise separable conv.png>)

## 3.5. Ví dụ tính toán cụ thể

Vẫn dùng ví dụ:

```text
H = 48, W = 48, k = 3, C_in = 32, C_out = 64
```

Convolution thường:

```text
Params_regular = 3 x 3 x 32 x 64 = 18,432
MACs_regular   = 48 x 48 x 18,432 = 42,467,328
```

Depthwise separable convolution:

```text
Params_depthwise = 3 x 3 x 32 = 288
Params_pointwise = 32 x 64 = 2,048
Params_ds        = 2,336
```

```text
MACs_ds = 48 x 48 x 2,336 = 5,382,144
```

Tỷ lệ:

```text
5.38M / 42.47M ≈ 12.7%
```

Như vậy, cùng input/output channel, depthwise separable convolution giảm khoảng 7.9 lần chi phí. Đây là nền tảng giúp MobileNetV2 nhẹ.

---

# 4. Các thành phần kiến trúc chính

## 4.1. Conv-BN-ReLU6

Trong notebook, block cơ bản được định nghĩa là `ConvBNReLU6`:

```python
class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
```

Block này gồm:

```text
Conv2d -> BatchNorm2d -> ReLU6
```

![Conv BN ReLU6](<../ly_thuyet/Ảnh info grafic/Phần 4/4.6. Hình Conv → BN → ReLU6.png>)

## 4.2. Vì sao Conv2d không dùng bias

Trong code, `bias=False`. Lý do là sau Conv2d có BatchNorm. BatchNorm có tham số `beta` đóng vai trò shift:

```text
y = gamma x normalized_x + beta
```

Nếu convolution có bias trước BatchNorm, bias đó phần lớn bị hấp thụ bởi phép chuẩn hóa và tham số `beta`. Vì vậy bỏ bias giúp giảm tham số thừa và là thiết kế phổ biến khi dùng `Conv -> BN`.

## 4.3. Batch Normalization

BatchNorm chuẩn hóa activation theo mini-batch:

```text
x_hat = (x - mean_batch) / sqrt(var_batch + epsilon)
y = gamma x x_hat + beta
```

Vai trò trong MobileNetV2:

- Ổn định phân phối activation.
- Giúp gradient mượt hơn.
- Cho phép training ổn định hơn.
- Đóng vai trò regularization nhẹ.

Trong MobileNetV2, BatchNorm được đặt sau các convolution, kể cả projection layer. Tuy nhiên projection layer không có ReLU phía sau.

## 4.4. ReLU6

ReLU6 được định nghĩa:

```text
ReLU6(x) = min(max(0, x), 6)
```

So với ReLU thường:

```text
ReLU(x) = max(0, x)
```

ReLU6 giới hạn activation tối đa ở 6.

![ReLU vs ReLU6](<../ly_thuyet/Ảnh info grafic/Phần 4/4.7. Hình ReLU vs ReLU6.png>)

Lý do dùng ReLU6:

- Giữ activation trong khoảng hữu hạn.
- Thuận lợi hơn cho quantization.
- Phù hợp với mục tiêu triển khai mobile/edge.
- Giảm rủi ro activation quá lớn ở các layer sâu.

Trong notebook, ReLU6 được dùng ở expansion và depthwise convolution, nhưng không dùng ở projection cuối block.

---

# 5. Inverted Residual Block, Linear Bottleneck và SE Attention

## 5.1. Đây là phần lõi của MobileNetV2

Nếu chỉ nhớ một ý về MobileNetV2, đó là:

```text
MobileNetV2 = Depthwise Separable Conv + Inverted Residual + Linear Bottleneck
```

Trong phiên bản v7, notebook bổ sung **SE Attention** vào block chính:

```text
MobileNetV2-SE = Depthwise Separable Conv + Inverted Residual + Linear Bottleneck + SE Attention
```

Inverted Residual Block trong notebook:

```python
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_skip = (stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_channels, hidden_dim, kernel_size=1))

        layers.append(ConvBNReLU6(hidden_dim, hidden_dim,
                                  kernel_size=3,
                                  stride=stride,
                                  groups=hidden_dim))

        if use_attention:
            layers.append(SEBlock(hidden_dim, reduction=attention_reduction))

        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)
```

Luồng tổng quát:

```text
Input hẹp
-> 1x1 expansion
-> 3x3 depthwise
-> SE attention
-> 1x1 projection tuyến tính
-> cộng skip nếu shape giống nhau
```

![Inverted residual block](<../ly_thuyet/Ảnh info grafic/Phần 4/4.8. Hình inverted residual block.png>)

## 5.2. Residual block truyền thống

Trong ResNet bottleneck truyền thống, cấu trúc thường là:

```text
wide -> narrow -> wide
```

Mục tiêu là giảm chi phí của convolution 3x3 bằng cách nén số kênh ở giữa rồi mở rộng lại. Skip connection nối giữa các tensor rộng.

Hình dưới là residual block trong paper MobileNetV2, dùng để đối chiếu với inverted residual.

![Residual block trong paper MobileNetV2](<assets/mobilenetv2/paper_residual_block.png>)

*Hình: residual block truyền thống, nguồn từ bản HTML ar5iv của paper MobileNetV2 [1].*

## 5.3. Inverted residual

MobileNetV2 đảo ngược cách làm:

```text
narrow -> wide -> narrow
```

Nó mở rộng channel ở giữa để tăng khả năng biểu diễn, dùng depthwise convolution ở không gian rộng, rồi nén về tensor hẹp. Skip connection nối giữa các bottleneck hẹp, không nối giữa tensor rộng.

![Inverted residual từ paper MobileNetV2](<assets/mobilenetv2/paper_inverted_residual_block.png>)

*Hình: inverted residual block trong paper MobileNetV2, nguồn từ ar5iv/CVF [1].*

So sánh:

| Tiêu chí | ResNet bottleneck | MobileNetV2 inverted residual |
|---|---|---|
| Luồng channel | wide -> narrow -> wide | narrow -> wide -> narrow |
| Vị trí skip | tensor rộng | tensor hẹp |
| Conv chính | regular 3x3 | depthwise 3x3 |
| Mục tiêu | giảm chi phí conv 3x3 | tăng biểu diễn ở giữa, giảm memory ở ngoài |
| Activation cuối block | thường có non-linearity | projection tuyến tính |

## 5.4. Expansion layer

Expansion layer là convolution 1x1 ở đầu block:

```text
C_in -> t x C_in
```

Trong đó `t` là expansion ratio, thường bằng 6.

Ví dụ:

```text
Input:  24 channels
t = 6
Hidden: 144 channels
```

![Expansion layer](<../ly_thuyet/Ảnh info grafic/Phần 4/4.10. Hình expansion layer.png>)

Vì depthwise convolution xử lý từng channel độc lập, nếu số channel quá nhỏ, nó có ít không gian để học biến đổi phi tuyến. Expansion layer mở rộng channel trước khi depthwise convolution, giúp block có không gian biểu diễn rộng hơn.

## 5.5. Depthwise layer trong block

Sau expansion, block dùng depthwise convolution 3x3:

```python
ConvBNReLU6(hidden_dim, hidden_dim,
            kernel_size=3,
            stride=stride,
            groups=hidden_dim)
```

Trong PyTorch, khi:

```text
groups = hidden_dim
in_channels = hidden_dim
out_channels = hidden_dim
```

thì đó là depthwise convolution. Mỗi channel có một filter 3x3 riêng.

Stride của depthwise layer quyết định block có giảm spatial size hay không:

```text
stride = 1 -> giữ H, W
stride = 2 -> giảm H, W còn một nửa
```

## 5.6. Projection layer

Projection layer là convolution 1x1 cuối block:

```text
t x C_in -> C_out
```

Trong code:

```python
nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False)
nn.BatchNorm2d(out_channels)
```

Điểm quan trọng:

```text
Không có ReLU sau projection.
```

Đây là **linear bottleneck**.

## 5.7. Linear bottleneck

Linear bottleneck là một trong hai đóng góp chính của MobileNetV2. Ý tưởng: khi tensor đã bị nén về ít channel, mỗi channel chứa nhiều thông tin hơn. Nếu đặt ReLU ở đây, các giá trị âm bị cắt về 0, làm mất thông tin khó phục hồi.

Vì vậy, MobileNetV2 giữ projection layer tuyến tính:

```text
Conv 1x1 -> BatchNorm -> không ReLU
```

![Linear bottleneck](<../ly_thuyet/Ảnh info grafic/Phần 4/4.11. Hình linear bottleneck.png>)

Paper gốc có thí nghiệm ablation cho thấy linear bottleneck hoạt động tốt hơn việc đặt ReLU6 trong bottleneck.

![Ablation linear bottleneck](<assets/mobilenetv2/paper_linear_bottleneck_ablation.png>)

*Hình: so sánh linear bottleneck với ReLU6 trong bottleneck, nguồn từ paper MobileNetV2 [1].*

## 5.8. Skip connection trong inverted residual

Skip connection chỉ được dùng khi:

```text
stride = 1
in_channels = out_channels
```

Code:

```python
self.use_skip = (stride == 1 and in_channels == out_channels)

def forward(self, x):
    if self.use_skip:
        return x + self.conv(x)
    return self.conv(x)
```

Nếu stride = 2, spatial size thay đổi, ví dụ:

```text
[B, C, 48, 48] -> [B, C_out, 24, 24]
```

Không thể cộng trực tiếp với input.

Skip connection giúp:

- Gradient truyền qua mạng sâu dễ hơn.
- Mô hình học phần dư thay vì học toàn bộ ánh xạ.
- Giữ lại đặc trưng hữu ích từ block trước.
- Giảm rủi ro degradation khi tăng độ sâu.

![Skip connection](<../ly_thuyet/Ảnh info grafic/Phần 4/4.12. Hình skip connection.png>)

Paper gốc cũng so sánh shortcut giữa bottleneck, shortcut giữa expansion và không dùng residual. Kết quả cho thấy shortcut giữa bottleneck là lựa chọn hiệu quả hơn.

![Ablation residual connection](<assets/mobilenetv2/paper_residual_ablation.png>)

*Hình: ảnh hưởng của các kiểu residual connection, nguồn từ paper MobileNetV2 [1].*

## 5.9. Bottleneck convolution trong paper

Hình sau minh họa một dạng bottleneck convolution trong paper, cho thấy vai trò của depthwise 3x3, ReLU6 và 1x1 convolution.

![Bottleneck convolution](<assets/mobilenetv2/paper_bottleneck_conv.png>)

*Hình: bottleneck convolution, nguồn từ paper MobileNetV2 [1].*

## 5.10. Công thức chi phí của một inverted residual block

Giả sử input block có kích thước:

```text
H x W x C_in
```

expansion ratio là `t`, output channel là `C_out`, kernel depthwise là `k x k`.

Expansion 1x1:

```text
MACs_expand = H x W x C_in x (t x C_in)
```

Depthwise 3x3:

```text
MACs_depthwise = H' x W' x (t x C_in) x k x k
```

Projection 1x1:

```text
MACs_project = H' x W' x (t x C_in) x C_out
```

Nếu stride = 1 thì `H' = H`, `W' = W`.

Tổng xấp xỉ:

```text
MACs_block = H x W x t x C_in x C_in
           + H' x W' x t x C_in x k^2
           + H' x W' x t x C_in x C_out
```

Trong paper, công thức được viết gọn theo kích thước block và expansion factor. Ý nghĩa chính là: block có thêm một 1x1 expansion so với MobileNetV1, nhưng nhờ input/output bottleneck hẹp, tổng chi phí vẫn thấp và memory hiệu quả.

## 5.11. SE Attention trong phiên bản v7

SE là viết tắt của **Squeeze-and-Excitation**. Đây là attention theo channel, tức mô hình học xem channel đặc trưng nào nên được nhấn mạnh và channel nào nên giảm ảnh hưởng.

Trong một feature map có dạng:

```text
C x H x W
```

mỗi channel có thể chứa một kiểu đặc trưng khác nhau, ví dụ cạnh tròn, vùng đỏ, viền trắng, ký hiệu bên trong biển báo hoặc nhiễu từ nền ảnh. SE block xử lý theo ba bước:

```text
Squeeze    : Global Average Pooling, C x H x W -> C x 1 x 1
Excitation : mạng nhỏ C -> C/reduction -> C
Reweight   : nhân trọng số attention vào từng channel
```

Code rút gọn trong notebook v7:

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        reduced_channels = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(self.pool(x))
        return x * scale
```

Ý nghĩa của `reduction = 4`:

```text
C -> C/4 -> C
```

Ví dụ nếu hidden channel trong block là 96:

```text
96 -> 24 -> 96
```

Giá trị đầu ra của `Sigmoid` nằm trong khoảng 0 đến 1. Channel nào hữu ích cho phân loại sẽ có trọng số lớn hơn, channel nào ít hữu ích hoặc gây nhiễu sẽ bị giảm. Quá trình này được học tự động thông qua backpropagation từ loss phân loại.

Trong v7, SE được đặt sau depthwise convolution và trước projection:

```text
Input
-> 1x1 expansion
-> 3x3 depthwise convolution
-> SE attention
-> 1x1 linear projection
-> skip connection nếu đủ điều kiện
```

Lý do đặt ở vị trí này là sau depthwise convolution, từng channel đã chứa đặc trưng không gian riêng. SE lúc này có thể đánh giá channel nào quan trọng trước khi projection nén tensor về bottleneck hẹp.

> Cần bổ sung hình: sơ đồ `Expansion -> Depthwise -> SE -> Projection -> Skip` cho phiên bản v7.

---

# 6. Kiến trúc MobileNetV2 đầy đủ

## 6.1. Bảng kiến trúc theo paper

MobileNetV2 được mô tả bằng các stage. Mỗi dòng có dạng:

```text
t, c, n, s
```

Trong đó:

| Ký hiệu | Ý nghĩa |
|---|---|
| `t` | expansion factor |
| `c` | số output channels |
| `n` | số block lặp lại |
| `s` | stride của block đầu trong stage |

Bảng kiến trúc chuẩn:

| Tầng | Operator | t | c | n | s |
|---|---|---:|---:|---:|---:|
| 1 | conv2d | - | 32 | 1 | 2 |
| 2 | bottleneck | 1 | 16 | 1 | 1 |
| 3 | bottleneck | 6 | 24 | 2 | 2 |
| 4 | bottleneck | 6 | 32 | 3 | 2 |
| 5 | bottleneck | 6 | 64 | 4 | 2 |
| 6 | bottleneck | 6 | 96 | 3 | 1 |
| 7 | bottleneck | 6 | 160 | 3 | 2 |
| 8 | bottleneck | 6 | 320 | 1 | 1 |
| 9 | conv2d 1x1 | - | 1280 | 1 | 1 |
| 10 | avgpool | - | - | 1 | - |
| 11 | classifier | - | số lớp | 1 | - |

Đặc điểm:

- Các stage đầu giữ nhiều thông tin không gian.
- Các stage giữa học hình dạng và bộ phận.
- Các stage cuối học đặc trưng ngữ nghĩa.
- Số kênh tăng dần khi spatial size giảm.

## 6.2. Kiến trúc trong notebook

Notebook định nghĩa:

```python
inverted_residual_setting = [
    [1,  16,  1, 1],
    [6,  24,  2, 2],
    [6,  32,  3, 2],
    [6,  64,  4, 2],
    [6,  96,  3, 1],
    [6,  160, 3, 2],
    [6,  320, 1, 1],
]
```

Notebook hiện dùng input 224x224. Với kích thước này, first conv dùng stride 2 giống cấu hình MobileNetV2 chuẩn để giảm spatial size từ đầu và kiểm soát chi phí tính toán:

```text
Input 224x224
-> Conv đầu stride 2
-> các stage bottleneck giảm dần spatial size
```

Với ảnh biển báo đã được resize/crop về 224x224, mô hình có đủ độ phân giải để học viền, ký hiệu, chữ số và mũi tên. Đổi lại, input 224x224 tốn thời gian train và VRAM hơn so với các kích thước nhỏ như 96x96.

## 6.3. Luồng feature map với input 224x224

Với cấu hình notebook:

| Stage | Operator | Output size | Channels | Ý nghĩa đặc trưng |
|---|---|---:|---:|---|
| Input | ảnh RGB | 224x224 | 3 | pixel ảnh biển báo |
| Conv đầu | 3x3 s=2 | 112x112 | 32 | cạnh, màu, gradient |
| Stage 1 | bottleneck t=1 | 112x112 | 16 | lọc đặc trưng thô |
| Stage 2 | bottleneck t=6 | 56x56 | 24 | texture, viền |
| Stage 3 | bottleneck t=6 | 28x28 | 32 | hình tròn/tam giác/chữ số |
| Stage 4 | bottleneck t=6 | 14x14 | 64 | ký hiệu, mũi tên, cấu trúc |
| Stage 5 | bottleneck t=6 | 14x14 | 96 | kết hợp bộ phận |
| Stage 6 | bottleneck t=6 | 7x7 | 160 | đặc trưng ngữ nghĩa |
| Stage 7 | bottleneck t=6 | 7x7 | 320 | tổng hợp class-level |
| Conv cuối | 1x1 | 7x7 | 1280 | feature vector giàu thông tin |
| GAP | avgpool | 1x1 | 1280 | gom không gian |
| Linear | classifier | - | num_classes | logits |

![Feature map size reduction](<../ly_thuyet/Ảnh info grafic/Phần 4/4.13. Hình feature map size reduction.png>)

## 6.4. Vì sao first conv stride 2 hợp lý với input 224

Trong paper gốc, input ImageNet là 224x224 nên first conv stride 2 đưa feature map về 112x112. Notebook hiện cũng dùng input 224x224, vì vậy stride 2 là hợp lý để giảm chi phí tính toán ngay từ đầu mà vẫn giữ đủ chi tiết không gian.

Lý do dùng stride 2 ở first conv:

- Giữ đúng tinh thần kiến trúc MobileNetV2 chuẩn với input 224x224.
- Giảm spatial size từ 224x224 xuống 112x112 để tiết kiệm VRAM và thời gian train.
- Vẫn giữ đủ chi tiết vì ảnh đầu vào lớn hơn so với cấu hình nhỏ như 96x96.
- Phù hợp với dữ liệu đã resize/crop về kích thước thống nhất.

Nếu dùng stride 1 với input 224x224, feature map đầu giữ 224x224 quá lâu, làm tăng đáng kể chi phí tính toán và bộ nhớ. Vì vậy trong notebook, stride đầu được chọn theo điều kiện `first_stride = 2 if IMG_SIZE >= 160 else 1`.

## 6.5. Adaptive average pooling và classifier

Sau các feature layers, tensor có dạng:

```text
[B, 1280, 7, 7]
```

Adaptive average pooling đưa về:

```text
[B, 1280, 1, 1]
```

Sau flatten:

```text
[B, 1280]
```

Classifier:

```python
self.classifier = nn.Sequential(
    nn.Dropout(p=dropout),
    nn.Linear(last_channels, num_classes),
)
```

![Adaptive average pooling](<../ly_thuyet/Ảnh info grafic/Phần 4/4.14. Hình adaptive average pooling.png>)

![Classifier cuối](<../ly_thuyet/Ảnh info grafic/Phần 4/4.15. Hình classifier cuối.png>)

Ưu điểm của global average pooling:

- Giảm mạnh số tham số so với flatten toàn bộ feature map.
- Buộc mỗi channel học một loại đặc trưng tổng quát.
- Giúp mô hình bớt phụ thuộc vị trí chính xác của đặc trưng.
- Phù hợp với ảnh classification đã crop.

## 6.6. Kiến trúc v7 sau khi thêm SE Attention

Phiên bản v7 không thay đổi số stage chính của MobileNetV2. Điểm khác biệt nằm trong từng inverted residual block: sau depthwise convolution có thêm SE block.

| Thành phần | MobileNetV2 v6 | MobileNetV2-SE v7 |
|---|---|---|
| Input | ảnh biển báo crop | ảnh biển báo crop |
| Block chính | Inverted Residual | Inverted Residual + SE |
| Attention | không có | SE channel attention |
| Vị trí attention | - | sau depthwise, trước projection |
| `attention_reduction` | - | 4 |
| Checkpoint/log | `mobilenetv2_gtsrb` | `mobilenetv2_gtsrb_v7_attention` |

Luồng block v7:

```text
Input hẹp
-> 1x1 expand
-> 3x3 depthwise
-> SE attention
-> 1x1 project linear
-> skip nếu stride=1 và channel bằng nhau
```

Việc thêm SE làm số tham số tăng nhẹ vì mỗi block có thêm hai layer 1x1 trong attention. Tuy nhiên, so với toàn bộ MobileNetV2, phần tăng thêm vẫn nhỏ và phù hợp với mục tiêu mô hình nhẹ.

> Cần bổ sung hình: bảng so sánh model size/parameter count giữa v6 và v7 sau khi chạy notebook.

---

# 7. Phân tích tensor shape, tham số và MACs

## 7.1. Phân tích một block cụ thể

Xét stage:

```text
Input: 48 x 48 x 24
Block: t = 6, C_out = 32, stride = 2
```

Expansion:

```text
48 x 48 x 24 -> 48 x 48 x 144
Params = 24 x 144 = 3,456
```

Depthwise 3x3 stride 2:

```text
48 x 48 x 144 -> 24 x 24 x 144
Params = 3 x 3 x 144 = 1,296
```

Projection:

```text
24 x 24 x 144 -> 24 x 24 x 32
Params = 144 x 32 = 4,608
```

Tổng tham số convolution:

```text
3,456 + 1,296 + 4,608 = 9,360
```

Nếu dùng convolution thường 3x3 trực tiếp từ 24 sang 32:

```text
Params_regular = 3 x 3 x 24 x 32 = 6,912
```

Ở ví dụ này inverted residual nhiều tham số hơn một conv 3x3 đơn lẻ vì có thêm expansion. Nhưng block MobileNetV2 không chỉ thay một conv; nó tạo một transformation giàu hơn gồm 1x1 expansion, depthwise spatial filtering và 1x1 projection. So sánh đúng hơn là với một residual block/chuỗi convolution có cùng khả năng biểu diễn. Khi xét toàn mạng, MobileNetV2 vẫn hiệu quả vì phần spatial filtering nặng được xử lý bằng depthwise convolution, còn input/output bottleneck giữ memory thấp.

## 7.2. Chi phí nằm chủ yếu ở đâu

Trong MobileNetV2, chi phí tính toán chủ yếu nằm ở các convolution 1x1, không phải depthwise 3x3. Lý do:

```text
Depthwise 3x3: k x k x C
Pointwise 1x1: C_in x C_out
```

Khi số kênh lớn, 1x1 convolution có nhiều phép nhân hơn. Vì vậy thiết kế bottleneck hẹp ở input/output rất quan trọng: nó giữ chi phí 1x1 không tăng quá mạnh.

## 7.3. Memory footprint

MobileNetV2 không chỉ giảm MACs mà còn quan tâm đến memory. Inverted residual nối shortcut giữa tensor hẹp, nghĩa là tensor cần giữ lại cho residual connection có số channel ít hơn so với nối shortcut ở tensor rộng.

Ví dụ:

```text
Shortcut ở expansion: cần giữ 144 channel
Shortcut ở bottleneck: chỉ cần giữ 24 hoặc 32 channel
```

Điều này đặc biệt quan trọng trên thiết bị di động, nơi băng thông bộ nhớ và cache thường là nút thắt.

## 7.4. Width multiplier

MobileNetV2 có thể điều chỉnh độ rộng bằng `width_mult`.

```text
actual_channels = base_channels x width_mult
```

Nếu `width_mult = 1.0`, dùng cấu hình chuẩn. Nếu `width_mult = 0.75`, số kênh giảm, mô hình nhẹ hơn nhưng accuracy có thể giảm.

Notebook có `_make_divisible` để làm tròn số kênh về bội số của 8:

```python
def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
```

Lý do:

- Channel chia hết cho 8 thường hiệu quả hơn trên phần cứng.
- Tránh số channel lẻ gây kém tối ưu.
- Giữ cấu trúc mạng ổn định khi scale width.

## 7.5. Resolution multiplier

Ngoài width, độ phân giải input cũng ảnh hưởng mạnh đến MACs:

```text
MACs tỷ lệ xấp xỉ với H x W
```

Notebook hiện chọn input 224x224, tức dùng độ phân giải đầy đủ hơn so với các cấu hình nhỏ như 96x96. Nếu so với 96x96, chi phí theo không gian tăng xấp xỉ:

```text
(224 x 224) / (96 x 96) ≈ 5.44
```

Điều này làm thời gian train và VRAM tăng, nhưng đổi lại mô hình giữ được nhiều chi tiết hơn trên biển báo, đặc biệt là chữ số, mũi tên, vạch trắng hoặc ký hiệu nhỏ.

Với ảnh biển báo đã crop, input 224x224 phù hợp khi mục tiêu là ưu tiên độ chính xác và khả năng phân biệt chi tiết nhỏ:

- Giữ nhiều chi tiết hình học hơn.
- Phù hợp với MobileNetV2 chuẩn.
- Tốn thời gian train hơn cấu hình nhỏ.
- Cần batch size vừa phải để tránh OOM.

---

# 8. Giải phẫu từng stage trong MobileNetV2 notebook

## 8.1. Vì sao cần phân tích theo stage

MobileNetV2 không phải là một chuỗi layer giống nhau. Mỗi stage có vai trò riêng trong quá trình biến đổi ảnh đầu vào thành vector đặc trưng. Nếu chỉ ghi bảng kiến trúc `t, c, n, s` thì vẫn chưa đủ để hiểu mô hình. Cần đọc bảng đó theo ba câu hỏi:

```text
1. Stage này nhận feature map kích thước bao nhiêu?
2. Stage này tăng/giảm channel và spatial size như thế nào?
3. Stage này học loại đặc trưng nào của ảnh biển báo?
```

Với input 224x224, notebook có thể dùng first conv stride 2 giống MobileNetV2 chuẩn mà vẫn giữ đủ chi tiết sau tầng đầu. Việc phân tích theo stage giúp thấy rõ mô hình giảm spatial size dần từ 224x224 xuống 7x7, đồng thời tăng số channel để chuyển từ đặc trưng cạnh/màu sang đặc trưng ngữ nghĩa.

## 8.2. Stage 0: Input và convolution đầu

Input của mô hình:

```text
[B, 3, 224, 224]
```

Trong đó:

```text
B: batch size
3: RGB channels
224 x 224: kích thước ảnh sau resize hoặc dữ liệu đã chuẩn bị sẵn
```

Convolution đầu:

```text
Conv 3x3, stride 2, output 32 channels
```

Output:

```text
[B, 32, 112, 112]
```

Vai trò:

- Chuyển ảnh RGB 3 kênh thành 32 feature channels.
- Học các cạnh, vùng màu, gradient sáng tối.
- Giảm spatial size còn 112x112 để tiết kiệm chi phí nhưng vẫn giữ đủ chi tiết ban đầu.

Với input 224x224, sau stride 2 feature map vẫn còn 112x112, đủ để giữ các đặc trưng như viền đỏ, nền xanh, nền trắng, cạnh tam giác, cạnh tròn và chữ số.

## 8.3. Stage 1: Bottleneck t=1, c=16, n=1, s=1

Cấu hình:

```text
t = 1
c = 16
n = 1
s = 1
```

Output:

```text
[B, 16, 112, 112]
```

Stage này không dùng expansion vì `t=1`. Nó chủ yếu nén 32 channels từ convolution đầu về 16 channels, giữ spatial size 112x112. Đây là bước lọc đặc trưng thô, loại bớt kênh không cần thiết và chuẩn bị cho các stage sâu hơn.

Vì output channel khác input channel, block này không dùng skip connection. Skip chỉ dùng khi input và output cùng shape.

Ý nghĩa với biển báo:

- Giữ lại đặc trưng cơ bản về màu và viền.
- Giảm channel để tiết kiệm tính toán.
- Chưa cần học ký hiệu phức tạp.

## 8.4. Stage 2: Bottleneck t=6, c=24, n=2, s=2

Cấu hình:

```text
t = 6
c = 24
n = 2
s = 2
```

Block đầu của stage dùng stride 2:

```text
112 x 112 -> 56 x 56
```

Output cuối stage:

```text
[B, 24, 56, 56]
```

Stage này có 2 block. Block đầu giảm spatial size, block thứ hai giữ spatial size. Block thứ hai có thể dùng skip connection nếu input/output cùng channel 24.

Vai trò:

- Bắt đầu giảm độ phân giải để giảm chi phí.
- Tăng channel từ 16 lên 24 để tăng biểu diễn.
- Học texture, viền, vùng màu, các cạnh rõ của biển báo.

Với biển báo, đây là giai đoạn mô hình bắt đầu nhận ra các pattern như viền tròn, viền tam giác, vùng màu đỏ/xanh/trắng. Vì spatial size vẫn còn 56x56, các chi tiết hình học lớn vẫn được giữ tốt.

## 8.5. Stage 3: Bottleneck t=6, c=32, n=3, s=2

Cấu hình:

```text
t = 6
c = 32
n = 3
s = 2
```

Output:

```text
[B, 32, 28, 28]
```

Stage này có 3 block:

- Block đầu giảm spatial size từ 56x56 xuống 28x28.
- Hai block sau giữ 28x28 và có thể dùng skip connection.

Vai trò:

- Học hình dạng tổng quát: tròn, tam giác, bát giác, hình thoi.
- Bắt đầu học ký hiệu lớn bên trong biển.
- Tăng receptive field, giúp mỗi điểm feature nhìn vùng ảnh rộng hơn.

Đây là stage quan trọng với biển báo vì hình dạng biển thường quyết định nhóm lớp. Ví dụ, biển cảnh báo thường tam giác, biển cấm thường tròn viền đỏ, biển hiệu lệnh thường tròn nền xanh.

## 8.6. Stage 4: Bottleneck t=6, c=64, n=4, s=2

Cấu hình:

```text
t = 6
c = 64
n = 4
s = 2
```

Output:

```text
[B, 64, 14, 14]
```

Stage này giảm spatial size xuống 14x14 nhưng tăng channel lên 64. Số block nhiều hơn, nên mô hình có nhiều bước phi tuyến để kết hợp đặc trưng.

Vai trò:

- Học các bộ phận bên trong biển báo.
- Kết hợp hình dạng ngoài với ký hiệu trong.
- Phân biệt các class cùng nhóm.

Ví dụ:

- Hai biển đều hình tròn viền đỏ nhưng khác ký hiệu cấm bên trong.
- Hai biển đều nền xanh nhưng khác hướng mũi tên.
- Các biển tốc độ khác nhau chỉ khác chữ số.

Ở 14x14, chi tiết nhỏ đã được nén, nhưng kênh nhiều hơn giúp mô hình biểu diễn ý nghĩa của đặc trưng thay vì giữ pixel gốc.

## 8.7. Stage 5: Bottleneck t=6, c=96, n=3, s=1

Cấu hình:

```text
t = 6
c = 96
n = 3
s = 1
```

Output:

```text
[B, 96, 14, 14]
```

Điểm đáng chú ý là stage này không giảm spatial size. Nó giữ 14x14 nhưng tăng channel từ 64 lên 96.

Vai trò:

- Tăng chiều biểu diễn mà không làm mất thêm thông tin không gian.
- Cho mô hình thêm thời gian xử lý ở độ phân giải trung gian.
- Kết hợp đặc trưng bộ phận thành cấu trúc class.

Với ảnh biển báo, giữ 14x14 ở stage này là hợp lý vì mô hình vẫn cần phân biệt ký hiệu, chữ số hoặc mũi tên. Nếu giảm xuống 7x7 quá sớm, một số chi tiết class có thể bị mất.

## 8.8. Stage 6: Bottleneck t=6, c=160, n=3, s=2

Cấu hình:

```text
t = 6
c = 160
n = 3
s = 2
```

Output:

```text
[B, 160, 7, 7]
```

Stage này chuyển từ 14x14 xuống 7x7. Đây là giai đoạn đặc trưng trở nên ngữ nghĩa hơn, tức mỗi vị trí trên feature map đã nhìn một vùng lớn của ảnh gốc.

Vai trò:

- Tổng hợp thông tin toàn cục hơn.
- Phân biệt class ở mức object-level.
- Giảm spatial size để chuẩn bị cho pooling cuối.

Ở đây, mô hình không còn chủ yếu học pixel hoặc cạnh đơn lẻ. Nó học tổ hợp đặc trưng: hình dạng biển + màu nền + ký hiệu + bố cục.

## 8.9. Stage 7: Bottleneck t=6, c=320, n=1, s=1

Cấu hình:

```text
t = 6
c = 320
n = 1
s = 1
```

Output:

```text
[B, 320, 7, 7]
```

Stage cuối của bottleneck tăng channel lên 320 nhưng giữ spatial size 7x7. Vì chỉ có một block, nó đóng vai trò tổng hợp đặc trưng trước convolution 1x1 cuối.

Vai trò:

- Tạo feature representation sâu.
- Chuẩn bị cho expansion cuối lên 1280 channels.
- Gom các đặc trưng class-level.

## 8.10. Conv 1x1 cuối: 320 -> 1280

Sau các bottleneck, MobileNetV2 dùng convolution 1x1 cuối:

```text
[B, 320, 7, 7] -> [B, 1280, 7, 7]
```

Mục đích:

- Mở rộng số kênh trước classifier.
- Tạo vector đặc trưng giàu thông tin hơn.
- Trộn mạnh thông tin giữa các kênh.

Convolution 1x1 cuối không học quan hệ không gian mới, nhưng trộn mạnh thông tin giữa các kênh. Đây là bước biến các đặc trưng stage cuối thành embedding có chiều cao hơn.

## 8.11. Global average pooling: 7x7 -> 1x1

Global average pooling:

```text
[B, 1280, 7, 7] -> [B, 1280, 1, 1]
```

Mỗi channel được lấy trung bình trên toàn bộ 7x7 vị trí. Có thể hiểu mỗi channel như một detector đặc trưng. Nếu channel đó phản ứng mạnh ở nhiều vị trí, giá trị trung bình cao; nếu không phát hiện đặc trưng, giá trị thấp.

Với ảnh crop biển báo, global average pooling hợp lý vì đối tượng chính đã nằm trong ảnh. Mô hình không cần biết chính xác ký hiệu nằm ở pixel nào; nó cần biết ảnh có những đặc trưng nào.

## 8.12. Tổng hợp số block và skip connection

Số bottleneck block:

```text
1 + 2 + 3 + 4 + 3 + 3 + 1 = 17 block
```

Số stage có downsampling:

```text
Stage 2, 3, 4, 6
```

Các block đầu stage có stride 2 thường không dùng skip. Các block còn lại trong cùng stage thường có thể dùng skip nếu channel không đổi.

| Stage | Số block | Downsample? | Skip khả dụng |
|---|---:|---|---:|
| t=1, c=16 | 1 | không | 0 |
| t=6, c=24 | 2 | có ở block đầu | 1 |
| t=6, c=32 | 3 | có ở block đầu | 2 |
| t=6, c=64 | 4 | có ở block đầu | 3 |
| t=6, c=96 | 3 | không | 2 hoặc 3 tùy input channel stage |
| t=6, c=160 | 3 | có ở block đầu | 2 |
| t=6, c=320 | 1 | không | 0 nếu channel vào khác 320 |

Ý nghĩa:

- Các block downsample thay đổi shape nên không cộng shortcut.
- Các block lặp lại trong cùng stage giữ shape nên tận dụng residual learning.
- Càng sâu, skip giúp gradient đi qua nhiều layer dễ hơn.

## 8.13. Receptive field trong MobileNetV2

Receptive field là vùng ảnh gốc mà một điểm trên feature map có thể "nhìn thấy". Ở layer đầu, kernel 3x3 chỉ nhìn vùng nhỏ. Qua nhiều convolution và stride, receptive field tăng dần.

Trong MobileNetV2:

- Các depthwise 3x3 tăng receptive field theo không gian.
- Các stride 2 làm mỗi điểm feature map đại diện cho vùng ảnh lớn hơn.
- Các pointwise 1x1 không tăng receptive field nhưng trộn thông tin channel.

Điều này phù hợp với biển báo:

- Layer đầu nhìn cạnh nhỏ.
- Layer giữa nhìn một phần ký hiệu.
- Layer sâu nhìn gần như toàn bộ biển báo.

## 8.14. Vì sao không dùng flatten trực tiếp

Nếu tensor cuối là:

```text
[B, 1280, 7, 7]
```

Flatten trực tiếp sẽ tạo vector:

```text
1280 x 7 x 7 = 62,720 chiều
```

Nếu classifier từ 62,720 chiều sang 12 class:

```text
62,720 x 12 = 752,640 tham số
```

Nếu dùng global average pooling:

```text
1280 x 12 = 15,360 tham số
```

Số tham số classifier giảm mạnh. Với dữ liệu custom không quá lớn, điều này giúp giảm overfitting.

## 8.15. Tóm tắt vai trò từng vùng mạng

| Vùng mạng | Vai trò chính | Liên hệ biển báo |
|---|---|---|
| Conv đầu | cạnh, màu, gradient | viền đỏ, nền xanh, chữ trắng |
| Stage 1-2 | đặc trưng thô | hình dạng ngoài |
| Stage 3-4 | hình dạng và bộ phận | tròn/tam giác/mũi tên/số |
| Stage 5 | kết hợp bộ phận | phân biệt class cùng nhóm |
| Stage 6-7 | đặc trưng ngữ nghĩa | nhận diện loại biển |
| Conv 1x1 cuối | embedding giàu thông tin | vector đại diện ảnh |
| GAP + Linear | phân loại | logits từng class |

---

# 9. Triển khai MobileNetV2 từ đầu trong notebook

## 9.1. Không dùng pretrained

Notebook xây dựng model:

```python
model = MobileNetV2(
    num_classes=CONFIG['num_classes'],
    width_mult=CONFIG['width_mult'],
    dropout=CONFIG['dropout']
).to(device)
```

Không có:

```python
torchvision.models.mobilenet_v2(weights=...)
```

Vì vậy đây là mô hình **train from scratch**. Trọng số ban đầu được khởi tạo ngẫu nhiên bằng Kaiming initialization cho convolution, BatchNorm weight = 1, BatchNorm bias = 0, Linear weight nhỏ.

Điều này cần nói rõ trong báo cáo vì khác với transfer learning. Nếu dùng pretrained, mô hình đã học đặc trưng ImageNet. Còn ở đây, toàn bộ đặc trưng biển báo được học từ dữ liệu custom.

## 9.2. Mapping giữa lý thuyết và code

| Lý thuyết MobileNetV2 | Code trong notebook |
|---|---|
| Conv + BN + ReLU6 | `ConvBNReLU6` |
| Depthwise convolution | `groups=hidden_dim` |
| Expansion 1x1 | `ConvBNReLU6(in_channels, hidden_dim, kernel_size=1)` |
| Projection 1x1 tuyến tính | `nn.Conv2d(hidden_dim, out_channels, 1)` + `BatchNorm`, không ReLU |
| Skip connection | `return x + self.conv(x)` |
| Stage config | `inverted_residual_setting` |
| Global average pooling | `adaptive_avg_pool2d(x, (1, 1))` |
| Classifier | `Dropout + Linear` |

Với phiên bản v7 có SE Attention, bảng mapping bổ sung:

| Thành phần v7 | Code trong notebook |
|---|---|
| SE attention block | `SEBlock` |
| Squeeze | `nn.AdaptiveAvgPool2d(1)` |
| Excitation | `Conv2d(C, C/reduction, 1) -> ReLU -> Conv2d(C/reduction, C, 1) -> Sigmoid` |
| Reweight channel | `return x * scale` |
| Bật/tắt attention | `CONFIG['attention_enabled']` |
| Mức nén SE | `CONFIG['attention_reduction']` |

Vị trí SE trong code v7:

```python
layers.append(ConvBNReLU6(hidden_dim, hidden_dim,
                          kernel_size=3,
                          stride=stride,
                          groups=hidden_dim))

if self.use_attention:
    layers.append(SEBlock(hidden_dim, reduction=attention_reduction))

layers.extend([
    nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
    nn.BatchNorm2d(out_channels),
])
```

## 9.3. Forward pass

Forward pass trong MobileNetV2:

```python
def forward(self, x):
    x = self.features(x)
    x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x
```

Output là logits:

```text
[batch_size, num_classes]
```

Không đặt softmax trong `forward` vì `CrossEntropyLoss` của PyTorch nhận logits trực tiếp. Softmax chỉ dùng khi cần hiển thị xác suất trong quá trình dự đoán.

## 9.4. Test output shape

Notebook kiểm tra bằng dummy tensor:

```python
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
out = model(dummy)
```

Nếu dataset có 12 class:

```text
out.shape = [1, 12]
```

Nếu dataset có 43 class:

```text
out.shape = [1, 43]
```

Kiểm tra này giúp phát hiện lỗi dimension trước khi train thật.

## 9.5. Vì sao dropout đặt ở classifier

Dropout được đặt trước Linear cuối:

```python
nn.Dropout(p=dropout)
nn.Linear(last_channels, num_classes)
```

Dropout giúp giảm overfitting bằng cách tắt ngẫu nhiên một phần feature trong train. Vì ảnh biển báo custom có thể dễ và crop sạch, dropout là regularization cần thiết để mô hình không quá tự tin vào một vài feature cụ thể.

## 9.6. Cấu hình train hiện tại liên quan kiến trúc

Cấu hình đang dùng:

| Tham số | Giá trị |
|---|---:|
| `resize_size` | 96 |
| `batch_size` | 32 |
| `lr` | 3e-4 |
| `warmup_epochs` | 10 |
| `weight_decay` | 5e-4 |
| `label_smoothing` | 0.15 |
| `dropout` | 0.35 |
| `augment_enabled` | 1 |

Các tham số này không thay đổi kiến trúc lõi, nhưng ảnh hưởng cách mô hình học:

- `dropout` tác động trực tiếp lên classifier cuối.
- `resize_size` ảnh hưởng kích thước feature map.
- `batch_size` ảnh hưởng BatchNorm statistics.
- `weight_decay` và `label_smoothing` giúp giảm overconfidence.
- `lr` thấp hơn làm quá trình cập nhật chậm hơn và ổn định hơn.

---

# 10. MobileNetV2 như bộ trích xuất đặc trưng cho biển báo giao thông

## 10.1. Tầng đầu học gì

Các layer đầu của CNN thường học:

- Cạnh ngang/dọc.
- Vùng màu đỏ, xanh, trắng, vàng.
- Viền biển báo.
- Gradient sáng tối.

Với biển báo, các đặc trưng này rất quan trọng vì hình dạng và màu sắc là tín hiệu mạnh.

Ví dụ:

```text
Biển cấm: hình tròn, viền đỏ
Biển hiệu lệnh: hình tròn, nền xanh
Biển cảnh báo: tam giác, viền đỏ
Biển stop: bát giác đỏ
```

## 10.2. Tầng giữa học gì

Các stage giữa học các pattern phức tạp hơn:

- Chữ số giới hạn tốc độ.
- Mũi tên rẽ trái/rẽ phải.
- Hình người đi bộ.
- Ký hiệu công trường.
- Ký hiệu cấm vượt, cấm vào.

Ở giai đoạn này, spatial size đã giảm nhưng vẫn còn đủ để giữ cấu trúc biển báo.

## 10.3. Tầng sâu học gì

Các stage cuối học đặc trưng gần với class:

- Tổ hợp màu + hình dạng + ký hiệu.
- Phân biệt các biển tương tự.
- Tổng hợp toàn bộ object thay vì chi tiết cục bộ.

Sau global average pooling, vector 1280 chiều có thể xem là embedding của ảnh biển báo. Classifier cuối chỉ cần ánh xạ embedding này sang class.

## 10.4. Vì sao MobileNetV2 hợp với ảnh biển báo crop

Ảnh biển báo crop có đặc điểm:

- Object chính nằm gần trung tâm.
- Nhiều class có hình học rõ.
- Nền ít quan trọng hơn object.
- Kích thước ảnh không quá lớn.

MobileNetV2 phù hợp vì:

- Không cần mô hình quá nặng để học đặc trưng.
- Depthwise separable convolution đủ cho pattern hình học.
- Skip connection giúp mạng đủ sâu để phân biệt class tương tự.
- Global average pooling giảm phụ thuộc vị trí nhỏ của biển trong crop.

## 10.5. Trường hợp dễ nhầm

MobileNetV2 có thể nhầm khi:

- Hai class chỉ khác chữ số nhỏ.
- Ảnh bị mờ làm mất ký hiệu.
- Crop mất một phần biển.
- Ánh sáng làm sai màu biển.
- Class có quá ít ảnh train.

Các trường hợp này nên được phân tích bằng confusion matrix, per-class accuracy và ảnh dự đoán sai.

---

# 11. Huấn luyện và đánh giá mô hình MobileNetV2

## 11.1. Loss và optimizer

Mô hình dùng `CrossEntropyLoss` cho classification nhiều lớp:

```python
criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
```

Optimizer:

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=CONFIG['lr'],
    momentum=CONFIG['momentum'],
    weight_decay=CONFIG['weight_decay'],
    nesterov=True
)
```

Với MobileNetV2 train from scratch, SGD + momentum là lựa chọn hợp lý vì thường generalize tốt trong image classification.

## 11.2. Learning rate schedule

Scheduler:

```text
Warmup -> Cosine Annealing
```

Warmup giúp mô hình ổn định khi weights còn random. Cosine decay giảm learning rate mượt ở cuối training để tinh chỉnh nghiệm.

![Learning rate warmup](<../ly_thuyet/Ảnh info grafic/Phần 5/5.7. Hình learning rate warmup.png>)

![Cosine annealing](<../ly_thuyet/Ảnh info grafic/Phần 5/5.8. Hình cosine annealing.png>)

## 11.3. Kết quả validation hiện tại

Log huấn luyện của MobileNetV2 baseline cho thấy mô hình học rất nhanh:

| Epoch | Train Acc | Val Acc | Nhận xét |
|---:|---:|---:|---|
| 1 | 15.94% | 25.29% | bắt đầu học |
| 3 | 60.99% | 76.73% | đã học đặc trưng chính |
| 4 | 71.78% | 84.61% | gần đạt mốc 85% |
| 5 | 83.74% | 97.72% | validation tăng mạnh |
| 8 | 96.72% | 99.67% | gần bão hòa |
| 12 | 98.52% | 99.95% | validation gần 100% |
| 17 | 99.05% | 99.95% | ổn định |

Kết quả này cho thấy MobileNetV2 đủ mạnh cho dữ liệu biển báo custom. Tuy nhiên, vì validation tăng quá nhanh, cần ghi nhận rằng dataset có thể dễ hoặc validation có phân phối rất gần train.

Với phiên bản v7 có SE Attention, log ban đầu cũng cho thấy mô hình học nhanh. Ví dụ các epoch đầu:

| Epoch | Train Acc | Val Acc | Nhận xét |
|---:|---:|---:|---|
| 1 | 18.56% | 25.88% | bắt đầu học |
| 4 | 61.87% | 74.17% | đã học đặc trưng chính |
| 7 | 80.87% | 89.99% | vượt mốc 85% |
| 8 | 88.80% | 97.28% | validation tăng mạnh |
| 9 | 92.63% | 97.93% | tiếp tục cải thiện |

Bảng v7 trên mới là log trong quá trình train, chưa phải kết quả cuối. Khi train v7 đủ epoch, báo cáo cần bổ sung thêm best validation accuracy, test accuracy và F1-score cuối cùng để so sánh công bằng với v6.

> Cần bổ sung hình: training/validation diagnostics của v7 sau khi train xong.
>
> Cần bổ sung hình: bảng so sánh v6 và v7 về `best_val_acc`, `test_acc`, `test_f1`, số tham số và thời gian mỗi epoch.

## 11.4. Overfitting và underfitting khi đọc curve

Underfitting:

```text
train acc thấp
val acc thấp
train loss cao
val loss cao
```

Overfitting:

```text
train acc tăng cao
val acc đứng yên hoặc giảm
train loss giảm
val loss tăng
```

Good fit:

```text
train acc cao
val acc cao gần train
train loss giảm
val loss thấp hoặc ổn định
```

Với log hiện tại, mô hình chưa có dấu hiệu overfitting rõ. Validation cao hơn train có thể do train đang dùng augmentation và dropout, còn validation thì sạch hơn.

## 11.5. Đánh giá cuối

Báo cáo cuối nên có:

- Accuracy.
- Precision macro.
- Recall macro.
- F1 macro.
- Confusion matrix.
- Normalized confusion matrix.
- Per-class accuracy.
- Per-class precision, recall và F1.
- Confidence distribution cho dự đoán đúng/sai.
- Mean confidence và số lỗi theo từng class.
- Một số ảnh dự đoán đúng.
- Một số ảnh dự đoán sai.

![Confusion matrix](<../ly_thuyet/Ảnh info grafic/Phần 7/7.5. Hình confusion matrix.png>)

![Per-class accuracy](<../ly_thuyet/Ảnh info grafic/Phần 7/7.6. Hình per-class accuracy.png>)

Các hình này giúp chứng minh mô hình không chỉ đạt accuracy tổng thể cao, mà còn hoạt động ổn định trên từng class.

Với bản v7 có SE Attention, nên trình bày thêm một nhóm hình so sánh trực tiếp với v6. Mục tiêu không chỉ là chứng minh v7 có validation cao hơn, mà còn kiểm tra xem SE có giúp các class khó, ảnh mờ, ảnh nghiêng hoặc ảnh có nền phức tạp tốt hơn không.

Các hình cần so sánh:

- `training_validation_diagnostics.png` của v6 và v7.
- `overall_metrics.png` của v6 và v7.
- `confusion_matrix_normalized.png` của v6 và v7.
- `per_class_metric_overview.png` của v6 và v7.
- `class_confidence_errors.png` của v6 và v7.
- Một vài `class_xx_samples.png` ở các class có lỗi hoặc class cải thiện rõ.

---

# 12. Nhận xét, hạn chế và hướng cải thiện

## 12.1. Ưu điểm của MobileNetV2 trong đề tài

- Nhẹ hơn nhiều mô hình CNN lớn.
- Tốc độ train và inference tốt.
- Phù hợp ảnh biển báo crop.
- Có cơ sở kiến trúc rõ ràng: depthwise separable convolution, inverted residual, linear bottleneck.
- Phiên bản v7 bổ sung SE Attention để mô hình tự học mức độ quan trọng của từng channel đặc trưng.
- Dễ triển khai bằng PyTorch từ đầu.
- Kết quả validation vượt xa mốc yêu cầu 85%.

## 12.2. Hạn chế

MobileNetV2 trong notebook là classifier, không phải detector. Vì vậy nó không tự tìm vị trí biển báo trong ảnh nguyên cảnh.

Ngoài ra:

- Nếu validation quá giống train, accuracy có thể cao hơn thực tế.
- Nếu ảnh ngoài crop sai, model dễ dự đoán sai.
- Nếu ảnh ngoài crop sai hoặc bị mờ mạnh, các chi tiết nhỏ vẫn có thể bị mất dù input là 224x224.
- Nếu data quá dễ, mô hình đạt gần 100% không nhất thiết chứng minh khả năng tổng quát ngoài thực tế.

## 12.3. Cách làm kết quả đáng tin hơn

- Dùng test set khác nguồn với train.
- Đảm bảo split trước augmentation.
- Không để biến thể của cùng ảnh gốc nằm ở nhiều split.
- Bổ sung ảnh khó: mờ, tối, xa, nghiêng, che khuất.
- Kiểm tra confusion matrix và per-class accuracy.
- Test thêm ảnh ngoài tự crop.

## 12.4. Hướng phát triển kiến trúc

Có thể cải thiện từ MobileNetV2 theo các hướng:

- MobileNetV3: thêm SE block, hard-swish, kiến trúc tối ưu bằng NAS.
- EfficientNet-B0: compound scaling depth/width/resolution.
- Đã thử hướng thêm attention nhẹ bằng SE trong phiên bản v7; bước tiếp theo là so sánh định lượng với v6 trên test set và ảnh ngoài.
- Có thể thử CBAM hoặc Coordinate Attention nếu SE chưa cải thiện rõ trên ảnh có background.
- Knowledge distillation từ mô hình lớn hơn.
- Quantization-aware training để triển khai trên thiết bị nhúng.
- Kết hợp detector để xử lý ảnh nguyên cảnh.

---

# 13. Danh sách hình cần bổ sung cho phiên bản SE Attention

Vì phiên bản v7 mới thêm SE Attention, báo cáo nên bổ sung các hình dưới đây sau khi train/test v7 hoàn tất. Các hình này giúp phần so sánh có căn cứ thay vì chỉ dựa vào validation log.

## 13.1. Hình kiến trúc SE trong MobileNetV2

Cần bổ sung:

- Sơ đồ `Expansion -> Depthwise -> SE -> Projection -> Skip`.
- Sơ đồ hoạt động SE: `Global Average Pooling -> C/reduction -> C -> Sigmoid -> nhân lại feature map`.

Vị trí nên đặt:

- Sau mục `5.11. SE Attention trong phiên bản v7`.

## 13.2. Hình so sánh training v6 và v7

Cần bổ sung:

- Training/validation loss của v6 và v7.
- Training/validation accuracy của v6 và v7.
- Accuracy gap và loss gap của v7.
- Learning rate và epoch time của v7.

File gợi ý từ notebook:

```text
training_validation_diagnostics.png
```

Vị trí nên đặt:

- Trong mục `10.3. Kết quả validation hiện tại`.

## 13.3. Hình đánh giá test set của v7

Cần bổ sung:

- `overall_metrics.png`.
- `confusion_matrix.png`.
- `confusion_matrix_normalized.png`.
- `per_class_metric_overview.png`.
- `test_confidence_distribution.png`.
- `class_confidence_errors.png`.

Vị trí nên đặt:

- Trong mục `10.5. Đánh giá cuối`.

## 13.4. Hình phân tích class sau khi thêm SE

Cần bổ sung:

- Các hình `class_XX_metrics.png` cho class có lỗi.
- Các hình `class_XX_samples.png` cho class cải thiện hoặc class còn nhầm.
- Ảnh dự đoán sai trước/sau nếu có so sánh v6 và v7.

Vị trí nên đặt:

- Sau phần confusion matrix hoặc trong phụ lục kết quả thực nghiệm.

## 13.5. Bảng so sánh cuối v6 và v7

Cần bổ sung bảng:

| Tiêu chí | v6 MobileNetV2 | v7 MobileNetV2-SE | Nhận xét |
|---|---:|---:|---|
| Best Val Acc | cần bổ sung | cần bổ sung | - |
| Test Acc | cần bổ sung | cần bổ sung | - |
| Test Precision | cần bổ sung | cần bổ sung | - |
| Test Recall | cần bổ sung | cần bổ sung | - |
| Test F1 | cần bổ sung | cần bổ sung | - |
| Params | cần bổ sung | cần bổ sung | v7 tăng nhẹ |
| Model size | cần bổ sung | cần bổ sung | v7 tăng nhẹ |
| Time/epoch | cần bổ sung | cần bổ sung | v7 có thể chậm hơn |

Vị trí nên đặt:

- Cuối mục `10.5. Đánh giá cuối` hoặc ngay trước kết luận.

---

# 14. Kết luận

MobileNetV2 là kiến trúc phù hợp cho bài toán phân loại biển báo giao thông đã crop vì nó cân bằng tốt giữa độ chính xác và hiệu quả tính toán. Trọng tâm của mô hình nằm ở inverted residual block với cấu trúc `narrow -> wide -> narrow`, trong đó expansion 1x1 convolution mở rộng không gian biểu diễn, depthwise 3x3 convolution xử lý đặc trưng không gian với chi phí thấp, và projection 1x1 convolution tuyến tính nén tensor về bottleneck hẹp mà không làm mất thông tin do ReLU.

So với convolution thường, depthwise separable convolution giảm mạnh số phép tính. So với residual block truyền thống, inverted residual nối shortcut giữa các tensor hẹp, giúp giảm bộ nhớ và hỗ trợ gradient truyền qua mạng sâu. Linear bottleneck là điểm khác biệt quan trọng vì nó tránh phá hủy thông tin ở không gian channel thấp.

Trong notebook, MobileNetV2 được triển khai từ đầu bằng PyTorch, không dùng pretrained. Kiến trúc gồm `ConvBNReLU6`, `InvertedResidual`, các stage bottleneck, adaptive average pooling và classifier cuối. Phiên bản v7 bổ sung `SEBlock` vào sau depthwise convolution để mô hình học trọng số quan trọng cho từng channel đặc trưng trước khi projection nén về bottleneck. Với dữ liệu hiện tại, cả baseline v6 và v7-SE đều học nhanh và vượt mốc yêu cầu 85% validation accuracy. Tuy nhiên, cần đánh giá thêm bằng test set độc lập và ảnh ngoài khó hơn để kết luận chắc chắn về khả năng tổng quát.

Tóm lại, phần đóng góp chính của MobileNetV2 trong đề tài không chỉ nằm ở accuracy cao, mà ở thiết kế kiến trúc hiệu quả: nhẹ, có cơ sở lý thuyết, dễ triển khai và phù hợp với ứng dụng nhận dạng biển báo trên thiết bị hạn chế tài nguyên. Phiên bản MobileNetV2-SE là bước mở rộng hợp lý vì giữ được tính nhẹ của MobileNetV2 nhưng thêm khả năng chú ý theo channel, từ đó có tiềm năng cải thiện các class khó hoặc ảnh có nhiễu nền.

---

# 15. Tài liệu tham khảo

[1] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018. CVF Open Access: https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html. arXiv HTML dùng cho hình minh họa: https://ar5iv.labs.arxiv.org/html/1801.04381.

[2] A. G. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," arXiv:1704.04861, 2017. https://arxiv.org/abs/1704.04861.

[3] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," CVPR, 2016.

[4] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," ICML, 2015.

[5] TorchVision documentation, `torchvision.models.mobilenet_v2`: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html.

[6] TorchVision source, `torchvision.models.mobilenetv2`: https://docs.pytorch.org/vision/main/_modules/torchvision/models/mobilenetv2.html.

[7] J. Hu, L. Shen, and G. Sun, "Squeeze-and-Excitation Networks," CVPR, 2018. https://arxiv.org/abs/1709.01507.

---

# Phụ lục A. Những điểm nên nói khi bảo vệ

- MobileNetV2 không phải chỉ là "CNN nhẹ"; nó nhẹ nhờ depthwise separable convolution.
- Điểm mới của MobileNetV2 so với MobileNetV1 là inverted residual và linear bottleneck.
- Inverted residual nối shortcut giữa tensor hẹp, khác ResNet nối giữa tensor rộng.
- Projection layer cuối block không dùng ReLU để tránh mất thông tin ở bottleneck.
- SE Attention trong v7 học trọng số cho từng channel bằng `Global Average Pooling -> C/reduction -> C -> Sigmoid`.
- SE được đặt sau depthwise convolution và trước projection để chọn lọc channel trước khi nén bottleneck.
- `attention_reduction = 4` nghĩa là channel trong SE được nén theo tỉ lệ `C -> C/4 -> C`.
- Trong notebook, model được xây dựng từ đầu, không dùng pretrained.
- Output của model là logits; softmax chỉ dùng khi hiển thị xác suất.
- Validation accuracy cao rất nhanh cho thấy dữ liệu dễ hoặc validation gần train; cần test độc lập để kết luận tổng quát.
