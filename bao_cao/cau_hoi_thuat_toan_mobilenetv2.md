# Câu hỏi thuật toán MobileNetV2

Tài liệu này tập trung vào cách thuật toán hoạt động trong mô hình MobileNetV2 + SE Attention. Nội dung không chỉ hỏi khái niệm, mà hỏi theo hướng luồng xử lý, bước tính toán, điều kiện, shape tensor, training và inference.

---

## 1. Nếu mô tả thuật toán MobileNetV2-SE bằng các bước thì gồm những bước nào?

Thuật toán có thể mô tả theo pipeline:

```text
Input image 224x224x3
-> Normalize
-> Conv đầu
-> Các Inverted Residual Block
-> SE Attention trong từng block
-> Conv cuối 1x1
-> Global Average Pooling
-> Dropout
-> Linear classifier
-> Logits 12 lớp
-> Softmax khi dự đoán
-> Chọn class có xác suất cao nhất
```

Trong training, logits được đưa vào `CrossEntropyLoss`. Trong inference, logits được đưa qua `softmax` để lấy xác suất dự đoán.

---

## 2. Pseudocode forward của mô hình là gì?

```text
function forward(x):
    x = first_conv(x)

    for block in inverted_residual_blocks:
        x = block(x)

    x = last_1x1_conv(x)
    x = global_average_pooling(x)
    x = flatten(x)
    x = dropout(x)
    logits = linear(x)

    return logits
```

Với input `224x224`, output cuối là vector logits có `12` phần tử, tương ứng với `12` class biển báo.

---

## 3. Một Inverted Residual Block xử lý input như thế nào?

Pseudocode:

```text
function inverted_residual_block(x):
    identity = x

    if expand_ratio != 1:
        x = conv1x1_expand(x)
        x = batch_norm(x)
        x = relu6(x)

    x = depthwise_conv3x3(x)
    x = batch_norm(x)
    x = relu6(x)

    x = se_attention(x)

    x = conv1x1_project(x)
    x = batch_norm(x)

    if stride == 1 and input_channels == output_channels:
        x = x + identity

    return x
```

Điểm quan trọng là projection cuối không dùng ReLU, vì đây là linear bottleneck.

---

## 4. Khi nào block dùng skip connection?

Block dùng skip connection khi:

```text
stride == 1
input_channels == output_channels
```

Nếu hai điều kiện này đúng:

```text
output = input + conv(input)
```

Nếu `stride = 2`, spatial size bị giảm nên không cộng được. Nếu số channel thay đổi, shape cũng khác nên không cộng trực tiếp được.

---

## 5. Nếu input block là C channel và expansion ratio là 6 thì hidden channel tính thế nào?

Hidden channel được tính:

```text
hidden_dim = C x expansion_ratio
```

Ví dụ input có `24` channel:

```text
hidden_dim = 24 x 6 = 144
```

Sau đó depthwise convolution chạy trên `144` channel. Cuối block, projection `1x1` nén lại về số output channel theo stage.

---

## 6. Depthwise convolution trong thuật toán chạy khác gì convolution thường?

Convolution thường dùng filter có shape:

```text
K x K x input_channels
```

để tạo ra mỗi output channel.

Depthwise convolution dùng một kernel riêng cho từng input channel:

```text
mỗi channel -> một kernel K x K
```

Do đó depthwise không trộn thông tin giữa các channel. Nó chỉ học đặc trưng không gian trên từng channel. Việc trộn channel được thực hiện sau đó bằng pointwise convolution `1x1`.

---

## 7. Pointwise convolution 1x1 làm gì trong thuật toán?

Pointwise convolution `1x1` có nhiệm vụ trộn thông tin giữa các channel.

Trong MobileNetV2, `1x1 conv` xuất hiện ở hai vị trí:

```text
1. Expansion: tăng channel
2. Projection: nén channel
```

Expansion giúp mô hình có không gian biểu diễn rộng hơn. Projection nén đặc trưng về số channel cần thiết để giữ mô hình nhẹ.

---

## 8. Padding ảnh hưởng gì đến kích thước feature map?

Trong model, convolution `3x3` dùng:

```python
padding = (kernel_size - 1) // 2
```

Với kernel `3x3`:

```text
padding = 1
```

Công thức output size:

```text
out = floor((in + 2*padding - kernel_size) / stride) + 1
```

Nếu `input = 224`, `kernel = 3`, `padding = 1`, `stride = 1`:

```text
out = floor((224 + 2 - 3) / 1) + 1 = 224
```

Nếu `stride = 2`:

```text
out = floor((224 + 2 - 3) / 2) + 1 = 112
```

Vì vậy stride `1` giữ nguyên kích thước, stride `2` giảm kích thước khoảng một nửa.

---

## 9. Shape tensor thay đổi thế nào qua toàn bộ mô hình?

Với input `224x224x3`:

```text
Input                  : 3 x 224 x 224
First conv             : 32 x 112 x 112
Stage 1                : 16 x 112 x 112
Stage 2                : 24 x 56 x 56
Stage 3                : 32 x 28 x 28
Stage 4                : 64 x 14 x 14
Stage 5                : 96 x 14 x 14
Stage 6                : 160 x 7 x 7
Stage 7                : 320 x 7 x 7
Last conv              : 1280 x 7 x 7
Global average pooling : 1280 x 1 x 1
Flatten                : 1280
Linear                 : 12
```

Spatial size giảm dần, còn số channel tăng dần. Đây là cách CNN chuyển từ đặc trưng cục bộ sang đặc trưng ngữ nghĩa.

---

## 10. SE Attention trong thuật toán nhận input và trả output như thế nào?

SE nhận feature map:

```text
x: B x C x H x W
```

Bước 1, squeeze:

```text
s = GlobalAveragePool(x)
s: B x C x 1 x 1
```

Bước 2, excitation:

```text
a = Conv1x1(C -> C/reduction)(s)
a = ReLU(a)
a = Conv1x1(C/reduction -> C)(a)
a = Sigmoid(a)
```

Lúc này:

```text
a: B x C x 1 x 1
```

Bước 3, reweight:

```text
output = x * a
```

Nhờ broadcasting, mỗi channel của `x` được nhân với một trọng số attention riêng.

---

## 11. Pseudocode SE Attention là gì?

```text
function SE(x):
    s = average_pool(x)       # B x C x 1 x 1
    a = conv1x1_reduce(s)     # C -> C/reduction
    a = relu(a)
    a = conv1x1_expand(a)     # C/reduction -> C
    a = sigmoid(a)
    return x * a
```

`sigmoid` đưa trọng số về khoảng `0` đến `1`. Channel có trọng số cao được giữ mạnh hơn, channel có trọng số thấp bị giảm ảnh hưởng.

---

## 12. Trong thuật toán, SE học bằng cách nào?

SE có các tham số riêng trong hai lớp convolution `1x1`. Các tham số này được học bằng backpropagation giống các layer khác.

Quá trình:

```text
1. Forward tạo attention weight
2. Attention weight nhân vào feature map
3. Model dự đoán logits
4. CrossEntropyLoss tính lỗi
5. Backpropagation tính gradient
6. Optimizer cập nhật cả weight của SE và weight của MobileNetV2
```

SE không cần nhãn riêng cho attention. Nó học gián tiếp từ loss phân loại cuối cùng.

---

## 13. Thuật toán training một batch diễn ra thế nào?

Pseudocode:

```text
for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    logits = model(images)
    loss = CrossEntropyLoss(logits, labels)

    loss.backward()
    optimizer.step()
```

Trong notebook, có thể có thêm mixed precision, gradient clipping hoặc scheduler, nhưng lõi thuật toán training vẫn là forward, tính loss, backpropagation, cập nhật weight.

---

## 14. Backpropagation cập nhật những gì trong mô hình?

Backpropagation tính gradient cho tất cả tham số có `requires_grad=True`.

Trong mô hình này, các tham số được cập nhật gồm:

```text
Conv đầu
Expansion conv
Depthwise conv
Projection conv
BatchNorm parameters
SE Attention parameters
Classifier linear layer
```

Vì model train from scratch nên toàn bộ tham số đều được học từ dữ liệu biển báo.

---

## 15. Loss Cross Entropy được tính như thế nào?

Với một ảnh, model trả về logits:

```text
z = [z1, z2, ..., zC]
```

Trong đó `C = 12` là số class. Đầu tiên softmax biến logits thành xác suất:

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

Nếu nhãn đúng là class `k`, Cross Entropy cơ bản là:

```text
L = -log(p_k)
```

Nếu model gán xác suất cao cho class đúng thì `p_k` gần `1`, loss nhỏ. Nếu model gán xác suất thấp cho class đúng thì loss lớn.

Trong PyTorch:

```python
loss = nn.CrossEntropyLoss()(logits, labels)
```

không cần tự softmax trước, vì `CrossEntropyLoss` đã gộp `LogSoftmax + NLLLoss`.

---

## 16. Gradient của Softmax + Cross Entropy là gì?

Với softmax và cross entropy, gradient theo logit có công thức gọn:

```text
dL/dz_i = p_i - y_i
```

Trong đó:

```text
p_i: xác suất dự đoán của class i
y_i: target one-hot của class i
```

Nếu class đúng là `k`:

```text
dL/dz_k = p_k - 1
```

Nếu model chưa tự tin vào class đúng, `p_k` thấp nên gradient âm lớn. Khi optimizer cập nhật theo hướng ngược gradient, logit class đúng sẽ được tăng lên.

Với class sai:

```text
dL/dz_i = p_i
```

Class sai nào đang có xác suất cao sẽ có gradient dương lớn, nên sau cập nhật logit của class đó bị giảm xuống.

---

## 17. Backward qua classifier cuối tính như thế nào?

Classifier cuối là linear layer:

```text
z = W h + b
```

Trong đó:

```text
h: vector đặc trưng 1280 chiều sau global average pooling
W: ma trận weight của Linear(1280, 12)
b: bias
z: logits 12 chiều
```

Sau khi có:

```text
dL/dz = p - y
```

gradient của classifier là:

```text
dL/dW = (dL/dz) * h^T
dL/db = dL/dz
dL/dh = W^T * (dL/dz)
```

Ý nghĩa:

```text
dL/dW: dùng để cập nhật weight classifier
dL/db: dùng để cập nhật bias classifier
dL/dh: truyền lỗi ngược về backbone MobileNetV2-SE
```

---

## 18. Backward qua Global Average Pooling như thế nào?

Global Average Pooling biến feature map:

```text
X: C x H x W
```

thành vector:

```text
h_c = average(X_c)
```

Cụ thể:

```text
h_c = (1 / (H*W)) * sum_i sum_j X_c,i,j
```

Khi backward, gradient từ `h_c` được chia đều về mọi vị trí không gian trong channel đó:

```text
dL/dX_c,i,j = dL/dh_c * (1 / (H*W))
```

Với feature map cuối `7x7`, mỗi vị trí nhận `1/49` gradient của channel tương ứng.

---

## 19. Backward qua SE Attention như thế nào?

SE Attention có dạng:

```text
Y = X * A
```

Trong đó:

```text
X: feature map đầu vào
A: attention weight theo channel, shape C x 1 x 1
Y: feature map sau attention
```

Khi backward, gradient đi theo hai nhánh:

```text
1. Gradient trực tiếp về feature map X
2. Gradient về attention A, rồi đi ngược qua sigmoid, conv1x1, ReLU, conv1x1 và global average pooling của SE
```

Với phép nhân:

```text
Y = X * A
```

ta có:

```text
dL/dX = dL/dY * A
dL/dA = sum(dL/dY * X)
```

Điều này nghĩa là SE không chỉ nhân trọng số khi forward, mà khi backward nó cũng học được channel nào nên tăng hoặc giảm ảnh hưởng để làm loss nhỏ hơn.

---

## 20. Backward qua skip connection hoạt động thế nào?

Với block có skip connection:

```text
Y = F(X) + X
```

Khi backward:

```text
dL/dX = dL/dY * dF/dX + dL/dY
```

Gradient có hai đường:

```text
1. Đi qua nhánh convolution F(X)
2. Đi thẳng qua nhánh identity X
```

Nhánh identity giúp gradient truyền ngược dễ hơn, giảm nguy cơ mất gradient khi mạng sâu. Đây là lý do skip connection giúp training ổn định.

---

## 21. Backward qua convolution học filter bằng cách nào?

Trong convolution, output được tính từ input và kernel:

```text
Y = Conv(X, W)
```

Backward tính hai loại gradient chính:

```text
dL/dW: gradient theo kernel để cập nhật filter
dL/dX: gradient theo input để truyền lỗi về layer trước
```

Trực giác:

```text
dL/dW đo mỗi filter đã góp phần làm loss tăng/giảm như thế nào
dL/dX cho biết lỗi nên truyền về vùng ảnh/feature nào ở layer trước
```

Trong depthwise convolution, mỗi kernel chỉ nhận gradient từ channel tương ứng. Trong pointwise `1x1 convolution`, gradient học cách trộn thông tin giữa các channel.

---

## 22. Thuật toán backward toàn bộ một batch có thể viết thế nào?

Pseudocode:

```text
Input: batch images X, labels y

1. Forward:
   logits = model(X)

2. Loss:
   loss = CrossEntropyLoss(logits, y)

3. Gradient tại output:
   dL/dlogits = softmax(logits) - one_hot(y)

4. Backward từ cuối về đầu:
   classifier -> global average pooling -> last conv
   -> inverted residual blocks -> first conv

5. Mỗi layer tính:
   gradient theo weight của chính nó
   gradient theo input để truyền tiếp về layer trước

6. Optimizer cập nhật:
   weight = weight - learning_rate * gradient
```

Trong code PyTorch, các bước 3 đến 5 được tự động thực hiện bằng:

```python
loss.backward()
```

PyTorch biết công thức đạo hàm của từng operation trong graph nên không cần tự viết tay backward cho từng layer.

---

## 23. Vì sao phải gọi `optimizer.zero_grad()` trước backward?

PyTorch mặc định cộng dồn gradient:

```text
grad_moi = grad_cu + grad_batch_hien_tai
```

Nếu không xóa gradient cũ, gradient của nhiều batch sẽ bị cộng lẫn không chủ đích, làm update sai.

Vì vậy mỗi batch cần:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Thứ tự này nghĩa là:

```text
xóa gradient cũ -> tính gradient batch hiện tại -> cập nhật weight
```

---

## 24. Gradient clipping nằm ở đâu trong thuật toán?

Sau backward, trước optimizer step:

```python
loss.backward()
clip_grad_norm_(model.parameters(), max_norm)
optimizer.step()
```

Nếu norm gradient quá lớn:

```text
||g|| > max_norm
```

gradient được scale nhỏ lại:

```text
g_new = g * max_norm / ||g||
```

Clipping giữ hướng gradient nhưng giảm độ lớn, giúp tránh cập nhật quá mạnh khi gradient bùng nổ.

---

## 25. SGD cập nhật weight theo công thức nào?

Công thức SGD cơ bản:

```text
w_new = w_old - lr * gradient
```

Trong đó:

```text
w_old: trọng số hiện tại
lr: learning rate
gradient: đạo hàm của loss theo trọng số
```

Trong notebook, SGD có thêm momentum và Nesterov nên cập nhật không chỉ dựa trên gradient hiện tại mà còn dựa trên hướng cập nhật trước đó.

---

## 26. Momentum ảnh hưởng đến thuật toán cập nhật thế nào?

Momentum lưu lại hướng cập nhật từ các bước trước.

Dạng đơn giản:

```text
v = momentum * v - lr * gradient
w = w + v
```

Nếu nhiều batch liên tiếp cho gradient cùng hướng, momentum giúp cập nhật ổn định và nhanh hơn theo hướng đó. Nếu gradient nhiễu, momentum giúp giảm dao động.

Trong đề tài:

```text
momentum = 0.9
```

---

## 27. Nesterov momentum khác momentum thường ở đâu?

Momentum thường cập nhật dựa trên gradient tại vị trí hiện tại.

Nesterov momentum nhìn trước một bước theo hướng momentum rồi hiệu chỉnh bằng gradient ở vị trí nhìn trước đó.

Viết trực giác:

```text
momentum thường:
  dùng gradient tại w hiện tại

Nesterov:
  nhìn trước theo vận tốc cũ
  tính gradient hiệu chỉnh
  cập nhật weight
```

Trong PyTorch:

```python
optim.SGD(..., momentum=0.9, nesterov=True)
```

Nesterov thường giúp cập nhật ổn định hơn vì optimizer không chỉ lao theo hướng cũ mà có bước kiểm tra trước.

---

## 28. Weight decay tác động vào gradient như thế nào?

Weight decay thêm phạt trọng số lớn:

```text
L_total = L_data + lambda * ||W||^2
```

Khi lấy gradient:

```text
dL_total/dW = dL_data/dW + 2 * lambda * W
```

Nghĩa là ngoài gradient từ dữ liệu, weight còn bị kéo nhỏ lại. Tác dụng là giảm overfit và tránh weight quá lớn.

Trong notebook:

```text
weight_decay = 1e-4
```

---

## 29. Scheduler ảnh hưởng đến thuật toán training thế nào?

Scheduler thay đổi learning rate theo epoch.

Trong mô hình:

```text
Epoch 1 -> 5: warmup, lr tăng dần
Epoch 6 -> 20: cosine, lr giảm dần
```

Warmup:

```text
lr = base_lr * current_epoch / warmup_epochs
```

Cosine:

```text
lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
```

Scheduler không thay đổi kiến trúc model. Nó chỉ thay đổi kích thước bước cập nhật weight.

---

## 30. Thuật toán inference một ảnh diễn ra thế nào?

Pseudocode:

```text
function predict(image):
    image = crop(image) nếu người dùng chọn vùng biển báo
    image = resize(image, 224x224)
    tensor = to_tensor(image)
    tensor = normalize(tensor)

    logits = model(tensor)
    probs = softmax(logits)
    predicted_class = argmax(probs)

    return predicted_class, confidence
```

Trong web landing page, người dùng upload ảnh, crop vùng biển báo, sau đó backend đưa crop vào model để dự đoán.

---

## 31. Vì sao inference cần softmax nhưng training forward không cần softmax?

Trong training, `CrossEntropyLoss` của PyTorch nhận logits trực tiếp và tự xử lý log-softmax bên trong.

Trong inference, ta cần hiển thị xác suất cho người dùng, nên phải dùng:

```text
probs = softmax(logits)
```

Sau đó chọn class:

```text
class = argmax(probs)
```

Vì vậy:

```text
Training: logits -> CrossEntropyLoss
Inference: logits -> softmax -> argmax
```

---

## 32. Nếu người dùng crop sai thì thuật toán dự đoán có thể sai vì sao?

Mô hình là classifier, không phải detector.

Classifier giả định input chính là vùng biển báo. Nếu crop sai, input có thể chứa nhiều background hoặc thiếu phần quan trọng của biển báo. Khi đó feature map học được sẽ không còn đúng với dữ liệu train.

Ví dụ:

```text
crop thiếu chữ số
crop thiếu viền biển báo
crop dính quá nhiều nền
crop biển báo quá nhỏ
```

Các trường hợp này làm đặc trưng đầu vào bị lệch, nên model có thể dự đoán sai.

---

## 33. Thuật toán chọn top-5 prediction như thế nào?

Sau khi có xác suất softmax:

```text
probs = softmax(logits)
```

Thuật toán sắp xếp hoặc lấy `topk`:

```text
top5 = topk(probs, k=5)
```

Kết quả gồm 5 class có xác suất cao nhất. Class đứng đầu là dự đoán chính.

---

## 34. Vì sao output classifier là 12?

Vì bài toán có `12` lớp biển báo.

Tầng cuối:

```text
Linear(1280, 12)
```

Nghĩa là từ vector đặc trưng `1280` chiều, model tạo ra `12` logits. Mỗi logit tương ứng với một class.

---

## 35. Nếu số class thay đổi thì thuật toán cần sửa gì?

Nếu số class thay đổi, cần sửa tầng classifier cuối:

```text
Linear(1280, num_classes)
```

Ngoài ra cần cập nhật danh sách class name đúng thứ tự với mapping dữ liệu.

Nếu dùng checkpoint cũ có `12` class mà đổi sang số class khác, không thể load trực tiếp classifier cũ vì shape không khớp.

---

## 36. Vì sao classifier chỉ có ít tham số?

Sau global average pooling, tensor chỉ còn vector `1280` chiều. Classifier là:

```text
Linear(1280, 12)
```

Số tham số:

```text
1280 * 12 + 12 = 15,372
```

Phần lớn tham số nằm ở `features`, tức backbone MobileNetV2-SE. Classifier chỉ ánh xạ đặc trưng cuối sang class nên rất nhỏ.

---

## 37. Thuật toán đánh giá accuracy tính như thế nào?

Với mỗi ảnh test:

```text
logits = model(image)
pred = argmax(logits)
```

Nếu:

```text
pred == label
```

thì tính là đúng.

Accuracy:

```text
accuracy = số dự đoán đúng / tổng số ảnh
```

Nếu accuracy test là `99.4%`, nghĩa là khoảng `99.4%` ảnh trong tập test được dự đoán đúng.

---

## 38. Thuật toán tạo confusion matrix như thế nào?

Với mỗi ảnh test, lưu:

```text
true_label
predicted_label
```

Sau đó cập nhật matrix:

```text
confusion_matrix[true_label][predicted_label] += 1
```

Nếu dự đoán đúng, giá trị nằm trên đường chéo chính. Nếu dự đoán sai, giá trị nằm ngoài đường chéo.

Normalized confusion matrix chia từng hàng cho tổng số ảnh thật của class đó để ra phần trăm.

---

## 39. Thuật toán tính precision và recall theo class như thế nào?

Với một class:

```text
TP: dự đoán đúng class đó
FP: dự đoán là class đó nhưng thật ra class khác
FN: thật là class đó nhưng dự đoán sang class khác
```

Precision:

```text
precision = TP / (TP + FP)
```

Recall:

```text
recall = TP / (TP + FN)
```

F1-score:

```text
F1 = 2 * precision * recall / (precision + recall)
```

---

## 40. Nếu thầy hỏi độ phức tạp thuật toán nằm ở đâu thì trả lời sao?

Chi phí chính nằm ở các convolution, đặc biệt là các pointwise convolution `1x1`.

MobileNetV2 giảm chi phí bằng cách thay convolution thường bằng depthwise separable convolution:

```text
Conv thường: H * W * M * N * K * K
Depthwise separable: H * W * M * K * K + H * W * M * N
```

Tuy nhiên, vì pointwise convolution trộn channel nên vẫn chiếm phần lớn tính toán trong MobileNetV2. SE Attention tăng thêm tham số và phép tính, nhưng không quá lớn so với backbone.

---

## 41. Nếu hỏi thuật toán có phải object detection không?

Không.

Thuật toán hiện tại là:

```text
image classification
```

Nó phân loại ảnh đã crop thành một trong 12 class. Nó không tìm bounding box của biển báo trong ảnh.

Nếu muốn object detection:

```text
Ảnh nguyên cảnh -> detector tìm biển báo -> crop -> MobileNetV2 phân loại
```

---

## 42. Trả lời ngắn khi thầy hỏi thuật toán hoạt động thế nào

Mô hình nhận ảnh RGB `224x224`, normalize rồi đưa qua MobileNetV2-SE. Đầu tiên ảnh qua convolution đầu để giảm kích thước còn `112x112`. Sau đó đi qua các inverted residual block. Trong mỗi block, mô hình mở rộng channel bằng `1x1 conv`, học đặc trưng không gian bằng depthwise `3x3 conv`, dùng SE Attention để tính trọng số cho từng channel, rồi nén lại bằng projection `1x1 conv`. Nếu shape không đổi thì cộng skip connection. Cuối cùng, feature map `1280 x 7 x 7` được đưa qua global average pooling thành vector `1280`, rồi classifier tạo `12` logits. Khi train, logits được dùng với CrossEntropyLoss; khi dự đoán, logits qua softmax và chọn class có xác suất cao nhất.
