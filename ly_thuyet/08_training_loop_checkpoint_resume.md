# Phần 8: Training loop, checkpoint và resume

## 1. Mục tiêu

Phần này giải thích Cell 8 trong notebook:

```text
TRAINING LOOP + LOGGING + CHECKPOINT
```

Các nội dung chính:

```text
model.train và model.eval
forward pass
loss
backward pass
optimizer step
mixed precision
gradient clipping
validation
training_log
best checkpoint
latest checkpoint
resume training
early stopping
```

Đây là phần mô hình thật sự học.

## 2. Training loop là gì?

Training loop là vòng lặp lặp lại nhiều epoch.

Mỗi epoch:

```text
1. Train qua toàn bộ train_loader
2. Validate trên val_loader
3. Ghi log
4. Lưu checkpoint nếu cần
5. Kiểm tra early stopping
```

Trong notebook:

```python
for epoch in range(start_epoch + 1, CONFIG['epochs'] + 1):
    scheduler.step()
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = validate(...)
```

## 3. Một batch train gồm những bước nào?

Trong `train_one_epoch`:

```python
for images, labels in loader:
    images, labels = images.to(device), labels.to(device)

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

Từng bước:

```text
1. Đưa ảnh và label lên GPU
2. Xóa gradient cũ
3. Forward pass
4. Tính loss
5. Backward pass
6. Clip gradient
7. Optimizer cập nhật weight
8. Cập nhật scaler cho mixed precision
```

## 4. `model.train()`

Trong train:

```python
model.train()
```

Chế độ train ảnh hưởng tới:

```text
Dropout: bật
BatchNorm: dùng thống kê batch hiện tại và cập nhật running stats
```

Nếu quên `model.train()`, dropout/BatchNorm có thể chạy ở chế độ eval, làm training không đúng.

## 5. `model.eval()`

Trong validation:

```python
model.eval()
```

Chế độ eval:

```text
Dropout: tắt
BatchNorm: dùng running mean/var đã học
```

Validation phải dùng eval để metric ổn định.

## 6. `torch.no_grad()`

Trong validate:

```python
@torch.no_grad()
def validate(...):
```

Không cần tính gradient khi validation.

Tác dụng:

```text
tiết kiệm VRAM
chạy nhanh hơn
không làm thay đổi gradient
```

## 7. Forward pass

Forward:

```python
outputs = model(images)
```

Input:

```text
images shape = [batch_size, 3, 224, 224]
```

Output:

```text
outputs shape = [batch_size, num_classes]
```

Nếu `num_classes = 12`:

```text
[32, 12]
```

Output là logits.

## 8. Loss

```python
loss = criterion(outputs, labels)
```

`labels` có shape:

```text
[batch_size]
```

Ví dụ:

```text
[0, 3, 11, 5, ...]
```

Mỗi số là index class.

CrossEntropyLoss so logits với label index.

## 9. `optimizer.zero_grad()`

PyTorch mặc định cộng dồn gradient qua các lần backward.

Nếu không gọi:

```python
optimizer.zero_grad()
```

gradient batch sau sẽ cộng với batch trước, làm update sai.

Vì vậy mỗi batch cần xóa gradient cũ trước khi backward.

## 10. Backward pass

```python
scaler.scale(loss).backward()
```

Backward tính gradient:

```text
∂loss/∂weight
```

cho mọi parameter trainable.

Gradient cho biết nếu thay đổi weight thì loss thay đổi thế nào.

## 11. Optimizer step

```python
scaler.step(optimizer)
```

Optimizer dùng gradient để cập nhật weight.

Với SGD:

```text
W_new = W_old - lr * gradient
```

có thêm momentum, Nesterov, weight decay như phần 7.

## 12. Mixed precision training

Notebook dùng:

```python
from torch.cuda.amp import GradScaler, autocast
```

Trong forward:

```python
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
```

Mixed precision cho phép một số phép toán dùng FP16 thay vì FP32.

Ưu điểm:

```text
giảm VRAM
tăng tốc trên GPU hỗ trợ
```

## 13. Vì sao cần `GradScaler`

FP16 có miền giá trị nhỏ hơn FP32. Gradient nhỏ có thể bị underflow thành 0.

`GradScaler` nhân loss lên trước backward:

```text
scaled_loss = loss * scale
```

sau đó unscale gradient trước khi optimizer step.

Mục tiêu:

```text
tránh gradient quá nhỏ bị mất trong FP16
```

## 14. Gradient clipping

Notebook:

```python
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
```

Phải unscale trước clip vì gradient đang bị scale.

Clipping giới hạn norm gradient:

```text
||gradient|| <= grad_clip
```

Nếu gradient bùng nổ, clipping giúp training không nhảy quá mạnh.

## 15. Tính train loss và train accuracy

Trong train:

```python
running_loss += loss.item() * images.size(0)
_, predicted = outputs.max(1)
total += labels.size(0)
correct += predicted.eq(labels).sum().item()
```

Loss trung bình:

```text
running_loss / total
```

Accuracy:

```text
correct / total * 100
```

Nhân `loss.item()` với `images.size(0)` vì loss là trung bình batch. Cộng theo số mẫu giúp tính trung bình đúng nếu batch cuối nhỏ hơn.

## 16. Validation loop

Validation gần giống train nhưng:

```text
không zero_grad
không backward
không optimizer step
không gradient clipping
```

Nó chỉ tính:

```text
val_loss
val_acc
```

Validation dùng để đánh giá model sau mỗi epoch.

## 17. Scheduler step mỗi epoch

Notebook:

```python
scheduler.step()
current_lr = scheduler.get_lr()
```

LR được cập nhật theo epoch.

Log in ra `current_lr` để biết epoch đó đang dùng learning rate nào.

## 18. `training_log`

Notebook tạo:

```python
training_log = {
    'epoch': [],
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'lr': [],
    'epoch_time': []
}
```

Mỗi epoch append một dòng.

Log này dùng cho:

```text
vẽ training history
lưu CSV
resume training
phân tích overfit
```

## 19. Đọc dòng log epoch

Ví dụ:

```text
2 | 0.6061 | 98.56% | 0.5616 | 99.73% | 0.004000 | 248.6s | BEST
```

Ý nghĩa:

```text
epoch = 2
train_loss = 0.6061
train_acc = 98.56%
val_loss = 0.5616
val_acc = 99.73%
lr = 0.004
time = 248.6s
status = BEST
```

`BEST` nghĩa là val accuracy tốt nhất đến thời điểm đó.

## 20. Checkpoint là gì?

Checkpoint là file lưu trạng thái training.

Notebook lưu:

```python
torch.save(state, filepath)
```

Một checkpoint chứa:

```text
epoch
model_state_dict
optimizer_state_dict
scheduler_state_dict
best_val_acc
training_log
config
```

## 21. `model_state_dict`

Chứa trọng số model:

```text
conv weights
batchnorm weights
linear weights
```

Nếu có file này, có thể load model để test/predict.

## 22. `optimizer_state_dict`

Chứa trạng thái optimizer:

```text
momentum buffer
learning rate hiện tại
```

Cần nếu muốn resume training đúng nghĩa.

Nếu chỉ inference, không cần optimizer state.

## 23. `scheduler_state_dict`

Chứa:

```python
current_epoch
```

Nếu resume, scheduler biết đang ở epoch nào để tiếp tục LR schedule.

## 24. `best_model.pth`

Khi:

```python
val_acc > best_val_acc
```

notebook lưu:

```text
best_model.pth
```

Đây là model tốt nhất theo validation accuracy.

Khi đánh giá test, notebook load file này:

```python
best_ckpt = os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth')
```

## 25. `checkpoint_latest.pth`

Notebook lưu sau mỗi epoch:

```text
checkpoint_latest.pth
```

File này dùng để resume nếu bị ngắt.

Khác với `best_model.pth`:

```text
best_model: tốt nhất theo val
latest: epoch mới nhất
```

## 26. Checkpoint định kỳ

Theo:

```python
save_every = 10
```

notebook lưu:

```text
checkpoint_epoch_10.pth
checkpoint_epoch_20.pth
```

Dùng để quay lại mốc cụ thể nếu cần.

## 27. Resume training

Nếu:

```python
'resume': True
```

và có `checkpoint_latest.pth`, notebook load:

```python
model.load_state_dict(...)
optimizer.load_state_dict(...)
scheduler.load_state_dict(...)
```

rồi train tiếp từ epoch đã lưu.

Nếu:

```python
'resume': False
```

notebook xóa checkpoint/log cũ và train mới.

## 28. Dừng cell train có sao không?

Nếu dừng giữa epoch:

```text
epoch đó chưa hoàn tất
checkpoint mới nhất có thể là epoch trước
training_log epoch đang chạy chưa ghi
```

Nếu dừng sau khi một dòng epoch đã in xong:

```text
checkpoint_latest thường đã lưu
best_model lưu nếu status BEST
```

Muốn train tiếp:

```python
'resume': True
```

## 29. Early stopping

Notebook dùng:

```python
patience_counter
CONFIG['patience']
```

Nếu val accuracy không cải thiện:

```text
patience_counter tăng
```

Nếu cải thiện:

```text
patience_counter reset về 0
```

Khi:

```text
patience_counter >= patience
```

training dừng.

Mục đích:

```text
tránh train quá lâu
giảm overfit
tiết kiệm GPU
```

## 30. Vì sao val_acc được dùng để chọn best?

Validation accuracy trực tiếp đo tỷ lệ đúng trên val.

Trong bài classification, đây là chỉ số dễ hiểu.

Tuy nhiên có thể cân nhắc chọn theo:

```text
val_loss thấp nhất
val_f1 cao nhất
```

nếu dataset imbalance hoặc accuracy bão hòa.

Notebook hiện chọn theo `val_acc`.

## 31. Khi nào nghi ngờ overfit trong log?

Dấu hiệu:

```text
train_loss giảm
train_acc tăng
val_loss tăng nhiều epoch
val_acc giảm hoặc đứng yên
```

Nếu cả train và val đều tốt:

```text
chưa kết luận overfit
```

Nếu test nội bộ tốt nhưng ảnh ngoài kém:

```text
có thể model overfit theo style dataset hoặc có distribution shift
```

## 32. Vì sao epoch 1 có thể cao?

Epoch 1 không phải model mới xem vài ảnh. Epoch 1 đã đi qua toàn bộ train set.

Nếu:

```text
train = 8578
batch = 32
```

thì epoch 1 có khoảng:

```text
269 optimizer steps
```

Với ảnh crop sạch, 12 class, model có thể học nhanh.

## 33. OOM trong training loop

Nếu OOM trong training:

```text
CUDA out of memory
```

Cách xử lý:

```text
1. Restart runtime để giải phóng GPU
2. Giảm batch_size
3. Đảm bảo first_stride=2 với ảnh 224
4. Tắt augment nếu augment quá nặng
```

Nếu dummy batch 1 cũng OOM, thường là GPU memory đang bị kẹt từ lần lỗi trước, cần restart runtime.

## 34. Cách trình bày trong báo cáo

Có thể viết:

```text
Quá trình huấn luyện được thực hiện theo từng epoch. Trong mỗi batch, ảnh được đưa qua mô hình để tính logits, sau đó CrossEntropyLoss được tính với nhãn thật. Gradient được lan truyền ngược, được clipping để tránh bùng nổ gradient, rồi optimizer SGD cập nhật trọng số. Notebook sử dụng mixed precision nhằm giảm VRAM và tăng tốc trên GPU. Sau mỗi epoch, mô hình được đánh giá trên validation set, ghi log và lưu checkpoint. Checkpoint tốt nhất được chọn dựa trên validation accuracy.
```

## 35. Câu hỏi phản biện thường gặp

### 35.1. Vì sao phải validation mỗi epoch?

Để theo dõi model có generalize không và lưu best checkpoint.

### 35.2. Vì sao lưu cả latest và best?

`latest` để resume training. `best` để đánh giá/predict model tốt nhất.

### 35.3. Vì sao batch cuối có thể nhỏ hơn?

Nếu số ảnh không chia hết cho batch size, batch cuối chứa phần còn lại. Vì vậy loss phải nhân với batch size rồi chia tổng số mẫu.

### 35.4. Vì sao cần restart sau OOM?

Sau OOM, PyTorch/Colab có thể vẫn giữ một phần memory. Restart runtime giải phóng GPU sạch.

## 36. Checklist training

```text
[ ] model.train khi train
[ ] model.eval khi validate
[ ] torch.no_grad khi validate
[ ] zero_grad trước backward
[ ] loss dùng logits
[ ] scaler dùng đúng với autocast
[ ] unscale trước gradient clipping
[ ] optimizer.step sau backward
[ ] scheduler step mỗi epoch
[ ] log train/val đầy đủ
[ ] best_model lưu khi val cải thiện
[ ] latest checkpoint lưu mỗi epoch
[ ] resume load model/optimizer/scheduler
```

## 37. Kết luận phần 8

Training loop là nơi dữ liệu, model, loss và optimizer kết hợp lại.

Tóm tắt:

```text
Forward tạo logits.
Loss đo sai số.
Backward tính gradient.
Optimizer cập nhật weight.
Validation đo khả năng tổng quát.
Checkpoint bảo vệ kết quả train.
Resume giúp train tiếp khi bị ngắt.
Early stopping tránh train quá lâu.
```

Hiểu training loop giúp đọc log đúng, xử lý OOM đúng, dừng/resume đúng, và giải thích được model thật sự học như thế nào.
