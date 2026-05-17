# CELL 7: LOSS FUNCTION, OPTIMIZER, SCHEDULER

## 1. CrossEntropyLoss + Label Smoothing

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 1.1 CrossEntropyLoss là gì?

Kết hợp **LogSoftmax + NLLLoss** trong 1 hàm:

```
Softmax:  p_i = exp(z_i) / Σ exp(z_j)     → chuyển logits thành xác suất [0,1]
CE Loss:  L = -Σ y_i × log(p_i)            → loss nhỏ khi p đúng class cao
```

**Ví dụ:** Model output [2.0, 1.0, 0.5] cho 3 classes, true = class 0:
```
Softmax: [0.659, 0.242, 0.099]
Loss: -1 × log(0.659) = 0.417
```
Model dự đoán đúng (class 0 cao nhất) nhưng chưa chắc chắn lắm → loss = 0.417.

### 1.2 Label Smoothing (ε = 0.1)

```
Trước: y = [1, 0, 0, ..., 0]   → "100% chắc chắn là class 0"
Sau:   y = [0.907, 0.0023, 0.0023, ..., 0.0023]  → "90.7% là class 0"
```

**Công thức:** `y_smooth = y × (1 - ε) + ε / K` (K = 43 classes)

**Tại sao cần?**
- Không smoothing: Model cố gắng output xác suất = 1.0 cho true class → logits phải tiến tới +∞ → weights cực lớn → overfit
- Có smoothing: Model chỉ cần output ~0.9 → weights vừa phải → generalize tốt hơn

**Kết quả thực nghiệm:** Label smoothing cải thiện 1-2% accuracy trên GTSRB.

---

## 2. SGD + Nesterov Momentum

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                      weight_decay=1e-4, nesterov=True)
```

### 2.1 SGD (Stochastic Gradient Descent)

**Cơ bản:** `w = w - lr × ∇L(w)`
- Tính gradient trên mini-batch (128 ảnh) thay vì toàn bộ dataset
- "Stochastic" = ngẫu nhiên (mini-batch ngẫu nhiên mỗi step)

### 2.2 Momentum

```
v_t = β × v_{t-1} + ∇L(w)        β = 0.9
w = w - lr × v_t
```

- `v` = velocity (tích lũy gradient trước đó)
- 90% momentum cũ + gradient mới → di chuyển mượt, vượt local minima
- Giống quả bóng lăn: tích lũy "đà" để vượt qua đồi nhỏ

### 2.3 Nesterov Momentum

Khác momentum thường: tính gradient tại **vị trí dự đoán** thay vì vị trí hiện tại.

```
w_look_ahead = w - lr × β × v_{t-1}     # "Nhìn trước"
v_t = β × v_{t-1} + ∇L(w_look_ahead)    # Tính gradient ở vị trí nhìn trước
w = w - lr × v_t
```

**Ưu điểm:** Nhanh hơn momentum thường 2-3% vì "sửa sai" trước khi đến → hội tụ mượt hơn.

### 2.4 Tại sao SGD mà không phải Adam?

| | SGD+Momentum | Adam |
|:---|:---|:---|
| Hội tụ | Chậm hơn ban đầu | Nhanh hơn ban đầu |
| Final accuracy | **Cao hơn** | Thấp hơn 0.5-1% |
| Generalization | **Tốt hơn** | Kém hơn |
| Cần tune LR | Có (Warmup + Cosine) | Ít hơn |

SGD+Momentum được ưu tiên cho image classification vì final accuracy cao hơn.

---

## 3. Warmup + Cosine Annealing Scheduler

```python
class WarmupCosineScheduler:
    def step(self):
        if epoch <= warmup_epochs:
            lr = base_lr × (epoch / warmup_epochs)    # Linear warmup
        else:
            progress = (epoch - warmup) / (total - warmup)
            lr = min_lr + (base_lr - min_lr) × 0.5 × (1 + cos(π × progress))
```

### 3.1 Warmup Phase (Epoch 1-5)

```
Epoch 1: lr = 0.002  |████░░░░░░|
Epoch 2: lr = 0.004  |████████░░|
Epoch 3: lr = 0.006  |██████████████░░|
Epoch 4: lr = 0.008  |████████████████████░░|
Epoch 5: lr = 0.010  |████████████████████████████|  ← Peak
```

**Tại sao warmup?**
- Epoch 1: weights random → gradient lớn, hướng hỗn loạn
- Nếu LR = 0.01 ngay: weights "nhảy" xa → loss tăng → diverge
- LR nhỏ (0.002) → bước nhỏ → model ổn định dần → tăng LR khi đã ổn

### 3.2 Cosine Decay Phase (Epoch 6-50)

```
lr = min_lr + (base_lr - min_lr) × 0.5 × (1 + cos(π × progress))
```

```
Epoch 6:  lr ≈ 0.010  |████████████████████████████|
Epoch 15: lr ≈ 0.008  |██████████████████████░░░░░░|
Epoch 25: lr ≈ 0.005  |████████████████░░░░░░░░░░░░|
Epoch 35: lr ≈ 0.002  |████████░░░░░░░░░░░░░░░░░░░░|
Epoch 50: lr ≈ 0.000  |░░░░░░░░░░░░░░░░░░░░░░░░░░░░|
```

**Tại sao cosine mà không giảm tuyến tính?**
- Cosine giảm **chậm** ở đầu (LR cao → học features chính) và **nhanh** ở cuối (LR thấp → fine-tune)
- Linear giảm đều → lãng phí thời gian ở LR trung bình
- Cosine decay cho accuracy cao hơn StepLR 0.5-1% (nhiều paper chứng minh)

## 4. Tác dụng tổng thể
Cell này thiết lập **bộ ba tối ưu**: Loss function (mục tiêu), Optimizer (cách cập nhật), Scheduler (tốc độ cập nhật). Ba thành phần phối hợp để training ổn định và hiệu quả.
