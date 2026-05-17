# CELL 3: CẤU HÌNH HYPERPARAMETERS

## 1. Mã nguồn

```python
CONFIG = {
    'img_size': 96,
    'batch_size': 128,
    'epochs': 50,
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'num_classes': 43,
    'width_mult': 1.0,
    'warmup_epochs': 5,
    'label_smoothing': 0.1,
    'dropout': 0.2,
    'grad_clip': 5.0,
    'patience': 10,
    'save_every': 10,
    'seed': 42,
    'resume': False,
    'checkpoint_dir': '/content/drive/MyDrive/mobilenetv2_gtsrb/checkpoints',
    'log_dir': '/content/drive/MyDrive/mobilenetv2_gtsrb/logs',
}
```

## 2. Phân tích từng Hyperparameter

### 2.1 img_size = 96

**Khái niệm:** Kích thước ảnh input đưa vào model (resize tất cả ảnh về 96×96 pixels).

**Tại sao 96 mà không phải kích thước khác?**

| Size | Ưu điểm | Nhược điểm | Thời gian/epoch |
|:---|:---|:---|:---|
| 32×32 | Rất nhanh | Mất quá nhiều chi tiết | ~10s |
| 48×48 | Nhanh | Mất chi tiết nhỏ (số trong biển) | ~30s |
| **96×96** | **Cân bằng tốt** | **Hơi chậm** | **~90s** |
| 224×224 | Chi tiết cao | Rất chậm, tốn VRAM, ảnh gốc bị mờ khi phóng to | ~300s |

Ảnh GTSRB gốc phần lớn 30-120px. Resize lên 96×96 giữ đủ chi tiết (đọc được số, thấy biểu tượng) mà không quá tốn tài nguyên.

### 2.2 batch_size = 128

**Khái niệm:** Số lượng ảnh được xử lý đồng thời trong 1 lần forward/backward pass.

**Ảnh hưởng:**
- **Lớn hơn (256, 512):** Gradient ổn định hơn (trung bình nhiều samples), tận dụng GPU parallelism tốt hơn, nhưng tốn VRAM và có thể generalize kém hơn
- **Nhỏ hơn (32, 64):** Gradient noisy hơn (có tác dụng regularization), ít VRAM, nhưng chậm hơn

**Tính VRAM:**
```
1 ảnh 96×96×3 (float32) = 96 × 96 × 3 × 4 bytes = 110 KB
128 ảnh = ~14 MB (chỉ input, chưa tính intermediate activations)
```
Trên GPU T4 (16GB VRAM), batch_size=128 với 96×96 là phù hợp.

### 2.3 epochs = 50

**Khái niệm:** 1 epoch = model nhìn qua TOÀN BỘ training set 1 lần.

50 epochs × 39,209 ảnh = ~1,960,450 lần model nhìn ảnh (có augmentation nên mỗi lần ảnh khác nhau).

Kết hợp early stopping (patience=10): nếu model không cải thiện trong 10 epoch → dừng sớm.

### 2.4 lr = 0.01 (Learning Rate)

**Khái niệm:** Tốc độ cập nhật weights mỗi bước.

```
w_new = w_old - lr × gradient
```

| LR | Hành vi |
|:---|:---|
| 0.1 | Quá lớn → loss dao động, có thể diverge |
| **0.01** | **Chuẩn cho SGD+Momentum trên classification** |
| 0.001 | Hội tụ chậm, nhưng ổn định hơn |
| 0.0001 | Rất chậm, thường dùng cho fine-tuning |

0.01 là giá trị được chứng minh hiệu quả qua nhiều paper (ImageNet training, CIFAR-10).

### 2.5 momentum = 0.9

**Khái niệm:** Tỷ lệ "quán tính" từ gradient trước đó.

```
velocity = 0.9 × velocity_old + gradient_new
w = w - lr × velocity
```

Giống như quả bóng lăn xuống dốc — momentum giúp nó vượt qua các "hố nhỏ" (local minima) để tìm "hố lớn" (global minimum).

### 2.6 weight_decay = 1e-4

**Khái niệm:** L2 Regularization — phạt weights có giá trị lớn.

```
Loss_total = Loss_CE + λ × Σ(w²)
```

- λ = 1e-4: Phạt nhẹ → weights không quá lớn → model đơn giản hơn → ít overfit
- Quá lớn (1e-2): Phạt quá mạnh → underfit
- Quá nhỏ (1e-6): Gần như không có tác dụng

### 2.7 width_mult = 1.0

**Khái niệm:** Hệ số scale số channels của model.

```
actual_channels = original_channels × width_mult
```

| width_mult | Channels (ví dụ stage 2) | Params | Dùng khi |
|:---|:---|:---|:---|
| 0.5 | 12 (thay vì 24) | ~0.7M | Mobile cực nhẹ |
| 0.75 | 18 | ~1.4M | Mobile nhẹ |
| **1.0** | **24** | **~2.2M** | **Chuẩn** |
| 1.4 | 34 | ~4.3M | Cần accuracy cao hơn |

### 2.8 warmup_epochs = 5

**Khái niệm:** Số epoch đầu tiên LR tăng dần từ nhỏ đến `lr`.

Epoch 1: LR = 0.002 → Epoch 5: LR = 0.01

**Tại sao cần?** Weights ban đầu random → gradient hỗn loạn. Nếu dùng LR lớn ngay → model "sốc", loss tăng vọt. Warmup cho model thích nghi trước.

### 2.9 label_smoothing = 0.1

**Khái niệm:** Làm "mềm" one-hot label.

```
Hard:  [0, 0, 0, 1, 0, ...]     → 100% chắc chắn
Soft:  [0.0023, 0.0023, 0.0023, 0.907, 0.0023, ...]  → 90.7% chắc chắn
```

Ngăn model quá tự tin → cải thiện generalization 1-2%.

### 2.10 dropout = 0.2

**Khái niệm:** Xác suất "tắt" mỗi neuron trong FC layer khi training.

Mỗi lần forward, 20% neurons khác nhau bị tắt → model không thể phụ thuộc vào neuron cụ thể → buộc phải học features phân tán, robust.

### 2.11 grad_clip = 5.0

**Khái niệm:** Giới hạn norm tối đa của gradient vector.

```
if ||gradient|| > 5.0:
    gradient = gradient × 5.0 / ||gradient||
```

Ngăn exploding gradient — gradient quá lớn làm weights "nhảy" xa → training bất ổn.

### 2.12 patience = 10

**Khái niệm:** Số epoch chờ đợi val_acc cải thiện trước khi dừng training.

Nếu 10 epoch liên tiếp val_acc không tăng → model đã hội tụ hoặc bắt đầu overfit → dừng sớm tiết kiệm thời gian.

### 2.13 seed = 42

**Khái niệm:** Số seed cho random number generator → kết quả reproducible.

```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

Cố định seed = cùng data split, cùng weight initialization, cùng augmentation → chạy lại cho cùng kết quả.

### 2.14 resume = False

- `False`: Xóa checkpoint cũ, train lại từ epoch 1
- `True`: Load checkpoint, tiếp tục từ epoch đã dừng

## 3. Tác dụng tổng thể
Cell này đóng vai trò **bảng điều khiển trung tâm** — thay đổi 1 giá trị ở đây ảnh hưởng toàn bộ pipeline. Thiết kế tập trung giúp dễ thực nghiệm (chỉ sửa CONFIG, không sửa code).
