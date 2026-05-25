# Phần 7: Loss function, optimizer và learning rate scheduler

## 1. Mục tiêu

Phần này giải thích Cell 7 trong notebook:

```text
LOSS FUNCTION, OPTIMIZER, SCHEDULER
```

Các thành phần chính:

```python
criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])

optimizer = optim.SGD(
    model.parameters(),
    lr=CONFIG['lr'],
    momentum=CONFIG['momentum'],
    weight_decay=CONFIG['weight_decay'],
    nesterov=True
)

scheduler = WarmupCosineScheduler(...)
```

Đây là phần quyết định model học như thế nào sau khi đã có architecture và data.

## 2. Loss function là gì?

Loss function đo model sai bao nhiêu.

Trong classification:

```text
model output -> so với label thật -> loss
```

Loss càng thấp nghĩa là model dự đoán càng phù hợp với label.

Training cố gắng tối thiểu hóa loss bằng cách cập nhật trọng số.

## 3. Logits là gì?

Output trực tiếp của model là logits:

```text
z = [z1, z2, ..., zC]
```

Logits chưa phải xác suất. Chúng có thể âm, dương, lớn, nhỏ bất kỳ.

Ví dụ:

```text
[2.1, -0.4, 0.7, 5.2]
```

Muốn chuyển thành xác suất, dùng softmax.

## 4. Softmax

Softmax:

```text
p_i = exp(z_i) / Σ_j exp(z_j)
```

Tính chất:

```text
0 <= p_i <= 1
Σ_i p_i = 1
```

Class có logit cao hơn sẽ có xác suất cao hơn.

Ví dụ:

```text
logits = [1, 2, 5]
softmax ≈ [0.017, 0.047, 0.936]
```

## 5. Cross entropy

Cross entropy cho classification:

```text
L = -Σ_i y_i log(p_i)
```

Với one-hot label, nếu class đúng là `k`:

```text
L = -log(p_k)
```

Nếu model dự đoán đúng với xác suất cao:

```text
p_k gần 1 -> loss gần 0
```

Nếu model dự đoán sai hoặc không chắc:

```text
p_k thấp -> loss lớn
```

## 6. Vì sao dùng `CrossEntropyLoss` trực tiếp với logits?

PyTorch:

```python
nn.CrossEntropyLoss()
```

đã gộp:

```text
LogSoftmax + Negative Log Likelihood
```

Vì vậy không cần tự softmax trước loss.

Sai:

```python
probs = torch.softmax(outputs, dim=1)
loss = criterion(probs, labels)
```

Đúng:

```python
loss = criterion(outputs, labels)
```

Softmax chỉ dùng khi cần đọc xác suất lúc evaluate/predict.

## 7. Gradient của cross entropy

Với softmax + cross entropy, gradient theo logit:

```text
∂L/∂z_i = p_i - y_i
```

Ý nghĩa:

- Class đúng: `y_i = 1`, nếu `p_i` thấp thì gradient âm lớn, optimizer tăng logit class đúng.
- Class sai: `y_i = 0`, gradient bằng `p_i`, optimizer giảm logit class sai.

Đây là lý do cross entropy rất phù hợp cho classification.

## 8. Label smoothing

Notebook dùng:

```python
label_smoothing = 0.1
```

Thay vì target one-hot:

```text
class đúng = 1.0
class sai = 0.0
```

target được làm mềm:

```text
class đúng ≈ 0.9
class sai nhận một phần nhỏ
```

## 9. Vì sao label smoothing hữu ích?

Nếu không smoothing, model có thể học quá tự tin:

```text
class đúng xác suất 0.9999
class khác gần 0
```

Quá tự tin có thể làm:

```text
overfit
calibration kém
loss nhạy với label noise
```

Label smoothing giúp:

```text
model bớt chắc chắn tuyệt đối
generalize tốt hơn
chịu label noise nhẹ tốt hơn
```

## 10. Accuracy cao nhưng loss chưa thấp có lạ không?

Không.

Accuracy chỉ xét:

```text
dự đoán đúng class hay không
```

Loss xét cả:

```text
model tự tin bao nhiêu
```

Nếu model dự đoán đúng nhưng chỉ 60% confidence, accuracy vẫn đúng nhưng loss còn cao.

Label smoothing cũng làm loss khó về 0 tuyệt đối.

## 11. Optimizer là gì?

Optimizer quyết định cách cập nhật trọng số dựa trên gradient.

Quy trình:

```text
forward -> loss -> backward -> optimizer.step()
```

Sau backward, mỗi weight có gradient:

```text
∂loss/∂weight
```

Optimizer dùng gradient đó để cập nhật weight.

## 12. SGD

Notebook dùng:

```python
optim.SGD(...)
```

SGD cập nhật:

```text
W_new = W_old - lr * gradient
```

Trong đó:

```text
lr = learning rate
```

SGD đơn giản, ổn định, thường generalize tốt cho CNN.

## 13. Momentum

Notebook:

```python
momentum = 0.9
```

Momentum thêm quán tính:

```text
v_t = momentum * v_{t-1} + gradient
W = W - lr * v_t
```

Trực giác:

```text
gradient cùng hướng nhiều bước -> đi nhanh hơn
gradient nhiễu -> đường đi mượt hơn
```

Momentum giống quả bóng lăn xuống dốc, có quán tính thay vì mỗi bước đổi hướng hoàn toàn.

## 14. Nesterov momentum

Notebook dùng:

```python
nesterov=True
```

Nesterov momentum nhìn trước theo hướng momentum rồi mới tính gradient hiệu chỉnh.

Trực giác:

```text
momentum thường: nhìn gradient tại vị trí hiện tại
nesterov: nhìn gradient tại vị trí sắp tới
```

Nesterov thường giúp cập nhật chính xác hơn và hội tụ tốt hơn.

## 15. Weight decay

Notebook:

```python
weight_decay = 1e-4
```

Weight decay phạt trọng số lớn.

Loss hiệu dụng:

```text
L_total = L_data + λ ||W||²
```

Trong đó:

```text
λ = 1e-4
```

Tác dụng:

```text
giảm overfit
khuyến khích weight nhỏ
làm model mượt hơn
```

Nếu quá lớn:

```text
model underfit
train acc thấp
```

## 16. Learning rate scheduler là gì?

Learning rate không nhất thiết cố định.

Scheduler thay đổi learning rate theo epoch.

Notebook dùng:

```text
linear warmup + cosine annealing
```

Mục tiêu:

```text
đầu train ổn định
giữa train học nhanh
cuối train tinh chỉnh nhẹ
```

## 17. Warmup

Trong warmup:

```python
lr = base_lr * (current_epoch / warmup_epochs)
```

Nếu:

```text
base_lr = 0.01
warmup_epochs = 5
```

thì:

```text
epoch 1: 0.002
epoch 2: 0.004
epoch 3: 0.006
epoch 4: 0.008
epoch 5: 0.010
```

Warmup giúp tránh cập nhật quá mạnh khi weight còn random.

## 18. Cosine annealing

Sau warmup:

```python
progress = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
```

Khi `progress = 0`:

```text
cos(0) = 1
lr ≈ base_lr
```

Khi `progress = 1`:

```text
cos(pi) = -1
lr ≈ min_lr
```

Learning rate giảm mượt.

## 19. Vì sao dùng cosine?

Cosine decay có ưu điểm:

```text
giảm learning rate mượt
không giảm đột ngột
cuối training bước nhỏ giúp fine-tune
```

Nó thường dùng trong computer vision.

## 20. Thứ tự scheduler trong notebook

Trong training loop:

```python
scheduler.step()
current_lr = scheduler.get_lr()
train_one_epoch(...)
validate(...)
```

Tức scheduler cập nhật LR ở đầu mỗi epoch.

Epoch 1 dùng LR sau step đầu tiên.

## 21. Khi nào warmup quá dài?

Nếu model học rất nhanh và chỉ train ít epoch, warmup 5 có thể hơi dài.

Ví dụ:

```text
epochs = 10
warmup_epochs = 5
```

nửa quá trình là warmup.

Với dataset này, có thể cân nhắc:

```python
warmup_epochs = 2
epochs = 15 hoặc 20
```

## 22. Khi nào learning rate quá cao?

Dấu hiệu:

```text
train loss dao động mạnh
val loss rất bất ổn
accuracy không tăng
loss NaN
```

Cách xử lý:

```python
lr = 0.005
```

hoặc tăng warmup.

## 23. Khi nào learning rate quá thấp?

Dấu hiệu:

```text
loss giảm rất chậm
accuracy tăng rất chậm
sau nhiều epoch vẫn thấp
```

Cách xử lý:

```python
lr = 0.01
```

hoặc train nhiều epoch hơn.

## 24. Quan hệ giữa optimizer và scheduler

Optimizer giữ learning rate trong:

```python
optimizer.param_groups[0]['lr']
```

Scheduler sửa giá trị đó mỗi epoch.

Optimizer cập nhật weight bằng learning rate hiện tại.

Vì vậy nếu scheduler sai, optimizer vẫn chạy nhưng LR có thể không như mong muốn.

## 25. `state_dict` của scheduler

Notebook có:

```python
def state_dict(self):
    return {'current_epoch': self.current_epoch}
```

Khi lưu checkpoint, scheduler lưu epoch hiện tại.

Khi resume:

```python
scheduler.load_state_dict(...)
```

LR schedule tiếp tục đúng vị trí, không reset về epoch 1.

## 26. Vì sao cần checkpoint optimizer?

Checkpoint lưu:

```python
optimizer_state_dict
```

Vì optimizer SGD momentum có trạng thái vận tốc.

Nếu chỉ load model weight mà không load optimizer, training vẫn tiếp tục được nhưng momentum bị mất, quỹ đạo training thay đổi.

## 27. Câu hỏi phản biện thường gặp

### 27.1. Vì sao dùng SGD mà không dùng Adam?

Adam thường hội tụ nhanh, nhưng SGD momentum thường generalize tốt cho CNN vision truyền thống. Với dataset đã crop sạch, SGD là lựa chọn hợp lý.

### 27.2. Vì sao cần label smoothing?

Để giảm overconfidence và tăng khả năng generalize, nhất là khi dữ liệu có thể có label noise hoặc nhiều ảnh rất giống nhau.

### 27.3. Vì sao warmup khi model học nhanh?

Warmup vẫn giúp ổn định lúc đầu. Tuy nhiên nếu model học rất nhanh, có thể giảm warmup từ 5 xuống 2.

### 27.4. Vì sao loss chưa bằng 0 dù accuracy 100%?

Do loss xét confidence, và label smoothing làm target mềm. Accuracy chỉ xét đúng/sai.

## 28. Cách trình bày trong báo cáo

Có thể viết:

```text
Hàm mất mát được sử dụng là CrossEntropyLoss với label smoothing 0.1 nhằm giảm overconfidence. Mô hình được tối ưu bằng SGD với Nesterov momentum 0.9 và weight decay 1e-4 để cải thiện khả năng tổng quát hóa. Learning rate ban đầu là 0.01, được điều chỉnh bởi scheduler gồm giai đoạn warmup tuyến tính và cosine annealing, giúp quá trình huấn luyện ổn định ở giai đoạn đầu và tinh chỉnh tốt hơn ở giai đoạn cuối.
```

## 29. Checklist phần loss/optimizer/scheduler

```text
[ ] Model output logits, không softmax trước loss
[ ] CrossEntropyLoss dùng đúng labels dạng index
[ ] Label smoothing hợp lý
[ ] SGD momentum/Nesterov được cấu hình
[ ] Weight decay không quá lớn
[ ] LR schedule hiển thị đúng
[ ] Scheduler state được checkpoint
[ ] Resume load optimizer và scheduler
```

## 30. Kết luận phần 7

Loss, optimizer và scheduler quyết định cách model học.

Tóm tắt:

```text
CrossEntropyLoss đo sai số phân loại.
Label smoothing giảm tự tin quá mức.
SGD momentum cập nhật weight ổn định.
Nesterov giúp nhìn trước hướng cập nhật.
Weight decay giảm overfit.
Warmup giúp đầu train ổn định.
Cosine annealing giúp cuối train tinh chỉnh mượt.
```

Hiểu phần này giúp giải thích vì sao loss giảm, vì sao LR tăng ở epoch đầu, vì sao accuracy cao nhưng loss chưa về 0, và vì sao checkpoint cần lưu cả optimizer/scheduler.
