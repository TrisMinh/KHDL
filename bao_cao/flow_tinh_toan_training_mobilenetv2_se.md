# Flow tính toán training MobileNetV2-SE từ đầu đến cuối

Tài liệu này trình bày thuật toán huấn luyện theo đúng flow tính toán: dữ liệu vào, forward, logits, loss, gradient, backward, mixed precision, gradient clipping, weight decay, momentum, Nesterov, scheduler và cập nhật trọng số.

Mục tiêu là trả lời câu hỏi: mô hình học bằng thuật toán gì, từng biến được tính như thế nào, và weight được cập nhật ra sao.

---

## 1. Ký hiệu dùng trong thuật toán

Với một mini-batch:

```text
B: batch size
C: số class
H, W: chiều cao và rộng ảnh
X: batch ảnh đầu vào
y: nhãn thật dạng class index
theta: toàn bộ tham số trainable của model
z: logits đầu ra của model
p: xác suất sau softmax
L: loss
g: gradient
eta: learning rate hiện tại
lambda: weight decay
mu: momentum
epsilon: label smoothing
```

Theo bản MobileNetV2-SE Attention đang dùng:

```text
B = 32
C = 12
H = W = 224
eta_base = 0.003
mu = 0.9
lambda = 1e-4
epsilon = 0.05
grad_clip = 5.0
warmup_epochs = 5
epochs = 20
optimizer = SGD + momentum + Nesterov
loss = CrossEntropyLoss(label_smoothing=0.05)
```

---

## 2. Input một batch

DataLoader lấy một batch:

```text
images: [B, 3, 224, 224]
labels: [B]
```

Ví dụ:

```text
images: [32, 3, 224, 224]
labels: [32]
```

`labels` không phải one-hot, mà là class index:

```text
labels = [0, 5, 2, 11, ...]
```

Sau đó đưa dữ liệu lên GPU:

```python
images = images.to(device)
labels = labels.to(device)
```

---

## 3. Xóa gradient cũ

Trước khi tính gradient cho batch hiện tại:

```python
optimizer.zero_grad()
```

Lý do: PyTorch mặc định cộng dồn gradient vào `.grad`.

Nếu không xóa:

```text
grad_batch_2 = grad_batch_1 + grad_batch_2
```

Điều này làm update sai vì mỗi batch cần gradient riêng.

---

## 4. Forward pass qua MobileNetV2-SE

Model nhận:

```text
X: [B, 3, 224, 224]
```

Forward tổng quát:

```text
X
-> first conv
-> inverted residual blocks
-> SE attention trong các block
-> last 1x1 conv
-> global average pooling
-> dropout
-> linear classifier
-> logits z
```

Output:

```text
z = model(X)
z: [B, C] = [32, 12]
```

Mỗi dòng của `z` là logits của một ảnh:

```text
z_n = [z_n1, z_n2, ..., z_n12]
```

Logits chưa phải xác suất. Logits có thể âm, dương, lớn hoặc nhỏ bất kỳ.

---

## 5. Tính softmax từ logits

Với ảnh thứ `n`, softmax tính:

```text
p_ni = exp(z_ni) / sum_j exp(z_nj)
```

Trong đó:

```text
i: class đang xét
j: chạy qua toàn bộ C class
```

Kết quả:

```text
p_n: [C]
sum_i p_ni = 1
0 <= p_ni <= 1
```

Ví dụ nếu có 3 class:

```text
z = [2.0, 1.0, 0.5]
p = softmax(z) = [0.629, 0.231, 0.140]
```

Trong code training không gọi softmax thủ công vì `CrossEntropyLoss` tự xử lý `log_softmax` bên trong.

---

## 6. Tạo target với label smoothing

Nếu không label smoothing, nhãn đúng được biểu diễn one-hot:

```text
y = [0, 0, 1, 0, ..., 0]
```

Nếu class đúng là `k`:

```text
y_k = 1
y_i = 0 với i != k
```

Với label smoothing:

```text
y'_i = (1 - epsilon) * y_i + epsilon / C
```

Với:

```text
epsilon = 0.05
C = 12
```

Class đúng:

```text
y'_k = 1 - 0.05 + 0.05/12
     = 0.9541667
```

Class sai:

```text
y'_i = 0.05/12
     = 0.0041667
```

Ý nghĩa: model không bị ép phải tin tuyệt đối 100% vào class đúng. Điều này giảm overconfidence và giảm overfit.

---

## 7. Tính Cross Entropy Loss cho từng ảnh

Với một ảnh `n`:

```text
L_n = - sum_i y'_ni * log(p_ni)
```

Nếu không label smoothing, công thức rút gọn thành:

```text
L_n = -log(p_nk)
```

với `k` là class đúng.

Với cả batch, loss là trung bình:

```text
L_batch = (1/B) * sum_n L_n
```

Trong code:

```python
loss = criterion(outputs, labels)
```

Trong đó:

```text
outputs = z = logits [B, C]
labels = y = class index [B]
```

---

## 8. Gradient tại logits

Với softmax + cross entropy, gradient theo logit có dạng:

```text
dL/dz_ni = (p_ni - y'_ni) / B
```

Nếu bỏ qua hệ số trung bình batch, phần lõi là:

```text
dL/dz_i = p_i - y'_i
```

Ý nghĩa:

```text
Nếu class đúng có p thấp:
  p_k - y'_k âm lớn
  optimizer sẽ tăng logit class đúng

Nếu class sai có p cao:
  p_i - y'_i dương lớn
  optimizer sẽ giảm logit class sai
```

Đây là tín hiệu lỗi đầu tiên để backward truyền ngược về toàn bộ mạng.

---

## 9. Backward qua classifier cuối

Classifier cuối là linear layer:

```text
z = hW^T + b
```

Trong đó:

```text
h: feature vector sau global average pooling, shape [B, 1280]
W: weight classifier, shape [C, 1280]
b: bias classifier, shape [C]
z: logits, shape [B, C]
```

Sau khi có:

```text
G_z = dL/dz
G_z: [B, C]
```

Gradient của classifier:

```text
dL/dW = G_z^T h
dL/db = sum_n G_z_n
dL/dh = G_z W
```

Trong đó:

```text
dL/dW: dùng để cập nhật weight classifier
dL/db: dùng để cập nhật bias classifier
dL/dh: truyền lỗi ngược về backbone
```

---

## 10. Backward qua dropout

Trong training, dropout tạo mask:

```text
m_i = 0 hoặc 1
```

Forward:

```text
h_drop = h * m / keep_prob
```

Backward:

```text
dL/dh = dL/dh_drop * m / keep_prob
```

Neuron nào bị dropout ở forward thì gradient của nó cũng bằng 0 ở backward trong batch đó.

Trong validation hoặc inference, dropout tắt nên không có mask ngẫu nhiên.

---

## 11. Backward qua Global Average Pooling

Trước global average pooling:

```text
F: [B, 1280, 7, 7]
```

Sau pooling:

```text
h: [B, 1280]
```

Forward:

```text
h_nc = (1 / (H*W)) * sum_i sum_j F_ncij
```

Với `H = W = 7`:

```text
h_nc = (1 / 49) * sum_i sum_j F_ncij
```

Backward:

```text
dL/dF_ncij = dL/dh_nc * (1 / 49)
```

Gradient của mỗi channel được chia đều về 49 vị trí không gian.

---

## 12. Backward qua convolution

Một convolution có dạng:

```text
Y = Conv(X, W) + b
```

Backward tính:

```text
dL/dW: gradient theo kernel
dL/db: gradient theo bias nếu có
dL/dX: gradient truyền về layer trước
```

Ý nghĩa:

```text
dL/dW cho biết filter cần thay đổi thế nào để loss giảm
dL/dX cho biết lỗi truyền về feature map trước đó như thế nào
```

Với pointwise convolution `1x1`:

```text
Nhiệm vụ forward: trộn thông tin giữa channel
Nhiệm vụ backward: học cách trộn channel sao cho loss giảm
```

Với depthwise convolution:

```text
Mỗi channel có kernel riêng
Gradient của kernel channel nào chủ yếu đến từ channel đó
```

---

## 13. Backward qua BatchNorm

BatchNorm trong training chuẩn hóa activation:

```text
x_hat = (x - mean_batch) / sqrt(var_batch + eps_bn)
y = gamma * x_hat + beta
```

Tham số học được:

```text
gamma
beta
```

Backward tính:

```text
dL/dgamma
dL/dbeta
dL/dx
```

`dL/dgamma` và `dL/dbeta` được optimizer cập nhật như các weight khác. `dL/dx` truyền lỗi về convolution phía trước.

---

## 14. Backward qua ReLU6

ReLU6:

```text
y = min(max(x, 0), 6)
```

Gradient:

```text
dy/dx = 1 nếu 0 < x < 6
dy/dx = 0 nếu x <= 0 hoặc x >= 6
```

Nghĩa là gradient chỉ đi qua vùng activation đang hoạt động. Vùng bị chặn dưới 0 hoặc chặn trên 6 không truyền gradient.

---

## 15. Backward qua SE Attention

SE nhận feature map:

```text
X: [B, C, H, W]
```

Forward:

```text
s = GlobalAveragePool(X)          # [B, C, 1, 1]
a = Conv1x1_reduce(s)             # [B, C/r, 1, 1]
a = ReLU(a)
a = Conv1x1_expand(a)             # [B, C, 1, 1]
a = Sigmoid(a)                    # attention weight
Y = X * a
```

Phép nhân cuối:

```text
Y_bcij = X_bcij * a_bc
```

Backward tại phép nhân:

```text
dL/dX_bcij = dL/dY_bcij * a_bc
dL/da_bc = sum_i sum_j dL/dY_bcij * X_bcij
```

Sau đó `dL/da` tiếp tục đi ngược qua:

```text
Sigmoid -> Conv1x1_expand -> ReLU -> Conv1x1_reduce -> GlobalAveragePool
```

Vì vậy SE học được channel nào nên tăng trọng số, channel nào nên giảm trọng số, dựa trên loss phân loại cuối cùng.

---

## 16. Backward qua skip connection

Trong inverted residual block có skip connection khi shape khớp:

```text
Y = F(X) + X
```

Backward:

```text
dL/dX = dL/dY * dF/dX + dL/dY
```

Gradient có hai đường:

```text
1. Đi qua nhánh convolution F(X)
2. Đi thẳng qua nhánh identity X
```

Đường identity giúp gradient truyền ngược dễ hơn qua mạng sâu, làm training ổn định hơn.

---

## 17. `loss.backward()` thực hiện gì?

Trong code:

```python
scaler.scale(loss).backward()
```

Nếu bỏ AMP, ý nghĩa tương đương:

```python
loss.backward()
```

PyTorch duyệt computation graph từ loss về đầu vào:

```text
loss
-> logits
-> classifier
-> pooling
-> convolution blocks
-> first conv
```

Với mỗi parameter `theta_i`, PyTorch tính:

```text
theta_i.grad = dL/dtheta_i
```

Các tham số có gradient gồm:

```text
classifier weight/bias
conv weight
BatchNorm gamma/beta
SE conv weight/bias
```

---

## 18. Mixed precision và GradScaler

Notebook dùng:

```python
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
```

`autocast` cho phép một số phép tính dùng FP16 để nhanh hơn và tiết kiệm VRAM, nhưng các phép nhạy cảm vẫn giữ FP32.

Vấn đề của FP16: gradient nhỏ có thể bị underflow thành 0.

Do đó dùng:

```python
scaled_loss = scaler.scale(loss)
scaled_loss.backward()
```

Nếu scale factor là `S`:

```text
L_scaled = S * L
dL_scaled/dtheta = S * dL/dtheta
```

Gradient đang bị nhân `S`, nên trước khi clip và update phải unscale:

```python
scaler.unscale_(optimizer)
```

Sau unscale:

```text
theta.grad = dL/dtheta
```

---

## 19. Gradient clipping

Sau backward và unscale, code gọi:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

Gọi:

```text
g_i = theta_i.grad
```

Tổng norm toàn bộ gradient:

```text
total_norm = sqrt(sum_i ||g_i||^2)
```

Nếu:

```text
total_norm <= 5.0
```

thì giữ nguyên gradient.

Nếu:

```text
total_norm > 5.0
```

thì scale toàn bộ gradient:

```text
clip_coef = 5.0 / (total_norm + small_eps)
g_i = g_i * clip_coef
```

Clipping giữ hướng gradient tổng thể nhưng giảm độ lớn. Mục tiêu là tránh bước cập nhật quá mạnh khi gradient bùng nổ.

Điểm quan trọng theo code hiện tại:

```text
Gradient clipping diễn ra trước optimizer.step().
Weight decay của PyTorch SGD được cộng trong optimizer.step().
```

Nghĩa là clipping đang áp dụng lên gradient từ loss, trước khi optimizer cộng thêm thành phần weight decay.

---

## 20. Weight decay được tính trong optimizer step

Optimizer dùng:

```python
optim.SGD(
    model.parameters(),
    lr=eta,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)
```

Với mỗi parameter `theta`, sau clipping ta có:

```text
g = theta.grad
```

PyTorch SGD cộng weight decay:

```text
g_decay = g + lambda * theta
```

Trong đó:

```text
lambda = 1e-4
```

Ý nghĩa: nếu weight lớn, thành phần `lambda * theta` kéo weight nhỏ lại. Đây là regularization để giảm overfit.

Nếu viết theo loss hiệu dụng, có thể hiểu gần như:

```text
L_total = L_data + (lambda/2) * ||theta||^2
```

Khi lấy đạo hàm:

```text
dL_total/dtheta = dL_data/dtheta + lambda * theta
```

---

## 21. Momentum buffer

SGD thường:

```text
theta_new = theta_old - eta * g
```

SGD có momentum lưu thêm vận tốc:

```text
v_t = mu * v_{t-1} + g_decay
```

Trong đó:

```text
mu = 0.9
v_t: momentum buffer ở step hiện tại
v_{t-1}: momentum buffer từ step trước
```

Nếu nhiều batch liên tiếp có gradient cùng hướng, momentum làm bước cập nhật mạnh và ổn định hơn. Nếu gradient nhiễu, momentum làm đường đi bớt dao động.

---

## 22. Nesterov momentum

Vì optimizer đặt:

```text
nesterov = True
```

PyTorch không dùng trực tiếp `v_t` để update, mà dùng:

```text
g_update = g_decay + mu * v_t
```

Sau đó cập nhật:

```text
theta_new = theta_old - eta * g_update
```

Tóm lại với PyTorch SGD + Nesterov:

```text
g = clipped_gradient
g_decay = g + lambda * theta
v_t = mu * v_{t-1} + g_decay
g_update = g_decay + mu * v_t
theta_new = theta_old - eta * g_update
```

Đây là công thức cập nhật weight thực tế trong optimizer.

---

## 23. Learning rate scheduler

Learning rate không cố định. Đầu mỗi epoch:

```python
scheduler.step()
eta = scheduler.get_lr()
```

Trong warmup:

```text
eta_epoch = eta_base * epoch / warmup_epochs
```

Với:

```text
eta_base = 0.003
warmup_epochs = 5
```

thì:

```text
epoch 1: eta = 0.0006
epoch 2: eta = 0.0012
epoch 3: eta = 0.0018
epoch 4: eta = 0.0024
epoch 5: eta = 0.0030
```

Sau warmup, dùng cosine annealing:

```text
progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
eta_epoch = eta_min + (eta_base - eta_min) * 0.5 * (1 + cos(pi * progress))
```

Trong code:

```text
eta_min = 1e-6
```

Scheduler không tự tính gradient. Nó chỉ quyết định bước cập nhật weight ở epoch đó lớn hay nhỏ.

---

## 24. Toàn bộ flow của một batch train

Pseudocode đầy đủ:

```text
Input:
  images X: [B, 3, 224, 224]
  labels y: [B]
  parameters theta
  momentum buffers v

Step 1: zero gradient
  theta.grad = 0

Step 2: forward
  z = model(X; theta)
  z: [B, 12]

Step 3: softmax trong CrossEntropyLoss
  p_ni = exp(z_ni) / sum_j exp(z_nj)

Step 4: label smoothing
  y'_ni = (1 - epsilon) * one_hot(y)_ni + epsilon / C

Step 5: loss
  L_n = - sum_i y'_ni * log(p_ni)
  L_batch = (1/B) * sum_n L_n

Step 6: AMP scale loss
  L_scaled = S * L_batch

Step 7: backward
  theta.grad = dL_scaled/dtheta = S * dL_batch/dtheta

Step 8: unscale gradient
  theta.grad = theta.grad / S

Step 9: gradient clipping
  total_norm = sqrt(sum_i ||theta_i.grad||^2)
  if total_norm > 5.0:
      theta_i.grad = theta_i.grad * 5.0 / total_norm

Step 10: weight decay trong optimizer
  g_i = theta_i.grad + lambda * theta_i

Step 11: momentum
  v_i = mu * v_i + g_i

Step 12: Nesterov
  g_update_i = g_i + mu * v_i

Step 13: update weight
  theta_i = theta_i - eta * g_update_i

Step 14: GradScaler update
  nếu gradient ổn: giữ hoặc tăng scale
  nếu inf/nan: giảm scale và bỏ qua update lỗi
```

Trong code PyTorch, flow này tương ứng:

```python
optimizer.zero_grad()

with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
scaler.step(optimizer)
scaler.update()
```

---

## 25. Flow của một epoch

Một epoch gồm toàn bộ batch trong `train_loader`:

```text
for each batch:
    chạy flow train một batch
    cộng dồn loss
    cộng dồn số dự đoán đúng
```

Train loss:

```text
train_loss = total_loss / total_samples
```

Trong code, vì `loss.item()` là loss trung bình của batch:

```text
running_loss += loss.item() * batch_size
train_loss = running_loss / total
```

Train accuracy:

```text
pred = argmax(logits, dim=1)
correct += count(pred == labels)
train_acc = correct / total
```

---

## 26. Validation khác training ở đâu?

Validation dùng:

```python
model.eval()
with torch.no_grad():
    outputs = model(images)
    loss = criterion(outputs, labels)
```

Khác training:

```text
Không dropout ngẫu nhiên
BatchNorm dùng running mean/var
Không lưu computation graph
Không backward
Không gradient clipping
Không optimizer.step
Không cập nhật weight
```

Validation chỉ đo:

```text
val_loss
val_acc
```

---

## 27. Chọn best checkpoint

Sau mỗi epoch:

```text
nếu val_acc > best_val_acc:
    lưu best_model.pth
    best_val_acc = val_acc
    patience_counter = 0
ngược lại:
    patience_counter += 1
```

Best model được chọn theo validation accuracy, không chọn theo train accuracy. Lý do là train accuracy chỉ đo khả năng học trên tập train, còn validation accuracy đo khả năng tổng quát hóa.

---

## 28. Thuật toán tổng quát từ epoch 1 đến hết

```text
Khởi tạo model parameters theta
Khởi tạo optimizer SGD(momentum=0.9, weight_decay=1e-4, nesterov=True)
Khởi tạo scheduler warmup + cosine
Khởi tạo GradScaler

for epoch = 1 to epochs:
    scheduler.step()
    eta = scheduler.get_lr()

    model.train()
    for batch in train_loader:
        tính forward
        tính loss
        backward
        unscale gradient
        clip gradient
        optimizer cập nhật theta bằng weight decay + momentum + Nesterov
        scaler.update()

    model.eval()
    validate trên val_loader

    nếu val_acc tốt nhất:
        lưu best_model.pth

    lưu checkpoint_latest.pth

    nếu nhiều epoch không cải thiện:
        early stopping
```

---

## 29. Trả lời ngắn khi thầy hỏi thuật toán học thế nào

Mỗi batch ảnh được đưa qua MobileNetV2-SE để tạo logits 12 lớp. `CrossEntropyLoss` biến logits thành log-softmax, so với nhãn có label smoothing để tạo loss trung bình trên batch. Backpropagation tính gradient của loss theo mọi tham số, bắt đầu từ công thức lõi `dL/dz = p - y'` rồi truyền ngược qua classifier, pooling, SE attention, skip connection và các convolution. Vì dùng mixed precision, loss được scale trước backward rồi gradient được unscale lại. Sau đó gradient được clipping theo norm tối đa 5.0 để tránh cập nhật quá mạnh. Trong `optimizer.step`, PyTorch SGD cộng weight decay vào gradient, cập nhật momentum buffer, áp dụng Nesterov momentum và cuối cùng cập nhật weight theo learning rate hiện tại. Learning rate được scheduler điều chỉnh theo warmup và cosine annealing qua từng epoch.

---

## 30. Công thức cần nhớ

```text
Softmax:
p_i = exp(z_i) / sum_j exp(z_j)

Label smoothing:
y'_i = (1 - epsilon) * y_i + epsilon / C

Cross entropy:
L = - sum_i y'_i log(p_i)

Gradient logits:
dL/dz_i = p_i - y'_i

Gradient clipping:
g = g * max_norm / ||g|| nếu ||g|| > max_norm

Weight decay:
g_decay = g + lambda * theta

Momentum:
v_t = mu * v_{t-1} + g_decay

Nesterov update gradient:
g_update = g_decay + mu * v_t

Weight update:
theta_new = theta_old - eta * g_update

Warmup learning rate:
eta = eta_base * epoch / warmup_epochs

Cosine learning rate:
eta = eta_min + (eta_base - eta_min) * 0.5 * (1 + cos(pi * progress))
```

