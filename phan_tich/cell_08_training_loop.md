# CELL 8: TRAINING LOOP + CHECKPOINT

## 1. Tổng quan

Cell dài nhất và phức tạp nhất — chứa toàn bộ logic training:
1. Resume/Reset logic
2. Mixed Precision Training (AMP)
3. Training loop (forward → loss → backward → update)
4. Validation loop
5. Logging & Checkpointing
6. Early Stopping

## 2. Resume/Reset Logic

```python
if not CONFIG['resume']:
    shutil.rmtree(CONFIG['checkpoint_dir'])  # Xóa checkpoint cũ
    print('🗑️ Đã xóa checkpoint + logs cũ!')

if CONFIG['resume'] and os.path.exists(latest_ckpt):
    start_epoch, best_val_acc, saved_log = load_checkpoint(...)
else:
    print("🆕 Bắt đầu training mới...")
```

**resume=False:** Xóa sạch → train từ epoch 1
**resume=True:** Load checkpoint → tiếp tục từ epoch đã dừng

## 3. Mixed Precision Training (AMP)

```python
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

for images, labels in train_loader:
    with autocast():                    # Forward pass bằng float16
        outputs = model(images)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()       # Scale loss lên, backward
    scaler.unscale_(optimizer)           # Unscale gradient
    clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])  # Gradient clipping
    scaler.step(optimizer)               # Update weights
    scaler.update()                      # Cập nhật scale factor
```

### 3.1 autocast()

- Tự động chuyển operations sang **float16** khi có lợi (matmul, conv)
- Giữ float32 cho operations nhạy cảm (softmax, loss, BN)
- GPU Tensor Cores xử lý float16 nhanh gấp 2× so với float32

### 3.2 GradScaler

**Vấn đề:** float16 có range nhỏ (±65,504). Gradient nhỏ (1e-7) bị round thành 0 (underflow).

**Giải pháp:**
1. `scaler.scale(loss)`: Nhân loss × scale_factor (ví dụ ×1024)
2. Gradient cũng lớn hơn ×1024 → không bị underflow
3. `scaler.unscale_()`: Chia gradient cho scale_factor trước khi update weights
4. `scaler.update()`: Nếu gradient inf/nan → tăng/giảm scale_factor tự động

### 3.3 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

- Nếu ||gradient|| > 5.0 → scale gradient xuống sao cho ||gradient|| = 5.0
- Giữ **hướng** gradient, chỉ giảm **độ lớn**
- Ngăn exploding gradient (thường xảy ra đầu training hoặc khi LR lớn)

## 4. Training Loop

```python
for epoch in range(start_epoch + 1, CONFIG['epochs'] + 1):
    model.train()                    # Bật training mode (BN, Dropout active)
    for images, labels in train_loader:
        optimizer.zero_grad()        # Reset gradient
        with autocast():
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Tính loss
        scaler.scale(loss).backward()  # Backward pass
        ...
        scaler.step(optimizer)       # Update weights
```

**model.train() vs model.eval():**

| | train() | eval() |
|:---|:---|:---|
| BatchNorm | Dùng batch μ, σ | Dùng running μ, σ |
| Dropout | Active (tắt 20% neurons) | Inactive (dùng hết neurons) |
| Gradient | Tính gradient | Thường dùng với no_grad() |

## 5. Validation Loop

```python
model.eval()                          # Tắt dropout, BN dùng running stats
with torch.no_grad():                 # Không tính gradient (tiết kiệm memory)
    for images, labels in val_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
```

**torch.no_grad():** Không lưu computation graph → tiết kiệm ~50% memory → cho phép batch_size lớn hơn khi eval.

## 6. Checkpoint System

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_acc': best_val_acc,
    'training_log': training_log,
}
torch.save(checkpoint, latest_ckpt)
```

**Lưu gì và tại sao:**

| Component | Tại sao lưu |
|:---|:---|
| model_state_dict | Weights đã train → resume predict |
| optimizer_state_dict | Momentum buffer → resume training mượt |
| scheduler_state_dict | LR hiện tại → resume đúng LR |
| best_val_acc | Biết acc tốt nhất → so sánh |
| training_log | History → vẽ biểu đồ |

**3 loại checkpoint:**
1. `best_model.pth` — khi val_acc cải thiện → dùng cho evaluation cuối
2. `checkpoint_epoch_N.pth` — mỗi 10 epoch → backup
3. `checkpoint_latest.pth` — mỗi epoch → resume khi Colab disconnect

## 7. Early Stopping

```python
if val_acc > best_val_acc:
    best_val_acc = val_acc
    patience_counter = 0
    torch.save(..., best_model_path)
else:
    patience_counter += 1
    if patience_counter >= CONFIG['patience']:  # 10
        break
```

10 epoch không cải thiện → dừng. Tránh train thừa gây overfit và lãng phí thời gian.

## 8. Tác dụng tổng thể
Cell này là **engine** chạy toàn bộ quá trình training. Kết hợp AMP (nhanh 2×), gradient clipping (ổn định), checkpoint (an toàn), early stopping (hiệu quả).
