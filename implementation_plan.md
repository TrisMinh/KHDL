# MobileNetV2 Phân Loại Biển Báo Giao Thông — Implementation Plan

## Mục tiêu
Xây dựng **MobileNetV2 từ đầu** (không dùng pretrained/model có sẵn) bằng PyTorch để phân loại biển báo giao thông. Toàn bộ nằm trong **1 file Jupyter Notebook (.ipynb)** chạy trên Google Colab.

---

## 1. Dataset — GTSRB (German Traffic Sign Recognition Benchmark)

> [!IMPORTANT]
> **Lý do chọn GTSRB:**
> - Dataset chuẩn học thuật, phổ biến nhất cho bài toán phân loại biển báo
> - **43 lớp** biển báo giao thông thực tế
> - **~51,839 ảnh** (39,209 train + 12,630 test)
> - Ảnh đa dạng: nhiều góc chụp, ánh sáng, điều kiện thời tiết khác nhau
> - Có sẵn trên Kaggle, dễ tải về bằng Kaggle API

**Nguồn:** `kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`

### 43 Lớp biển báo
Bao gồm: Giới hạn tốc độ (20, 30, 50, 60, 70, 80, 100, 120 km/h), Cấm vượt, Ưu tiên, Nhường đường, Dừng lại, Cấm đi, Nguy hiểm, Đường cong, Trơn trượt, Vòng xuyến, v.v.

---

## 2. Kiến Trúc MobileNetV2 — Custom Implementation

### 2.1 Core Concepts
- **Depthwise Separable Convolution**: Tách convolution thành depthwise (3×3) + pointwise (1×1) → giảm params
- **Inverted Residual Block**: Narrow → Wide → Narrow (expand → depthwise → project)
- **Linear Bottleneck**: Không dùng ReLU ở lớp projection cuối cùng
- **ReLU6**: Activation function `min(max(0, x), 6)`

### 2.2 Architecture Table (theo paper gốc, điều chỉnh cho ảnh 48×48)

| Stage | Operator | t | c | n | s |
|:---:|:---|:---:|:---:|:---:|:---:|
| 0 | Conv2d 3×3 | - | 32 | 1 | 2 |
| 1 | Bottleneck | 1 | 16 | 1 | 1 |
| 2 | Bottleneck | 6 | 24 | 2 | 2 |
| 3 | Bottleneck | 6 | 32 | 3 | 2 |
| 4 | Bottleneck | 6 | 64 | 4 | 2 |
| 5 | Bottleneck | 6 | 96 | 3 | 1 |
| 6 | Bottleneck | 6 | 160 | 3 | 2 |
| 7 | Bottleneck | 6 | 320 | 1 | 1 |
| 8 | Conv2d 1×1 | - | 1280 | 1 | 1 |
| 9 | AvgPool + FC | - | 43 | 1 | - |

> [!NOTE]
> Input sẽ resize về **48×48** thay vì 224×224 gốc vì ảnh GTSRB nhỏ (nhiều ảnh chỉ ~30×30). Stride ở stage đầu có thể điều chỉnh từ 2→1 để giữ feature map lớn hơn khi input nhỏ.

### 2.3 Các Module cần implement

```
├── ConvBNReLU6        # Conv2d + BatchNorm + ReLU6
├── InvertedResidual   # Inverted Residual Block (expand → depthwise → project)
└── MobileNetV2        # Full network assembly
```

---

## 3. Cấu Trúc Notebook (.ipynb)

### Cell 1: Setup & Cài đặt
```
- Mount Google Drive
- Cài Kaggle API, tải dataset
- Import thư viện (torch, torchvision, matplotlib, sklearn, ...)
- Thiết lập device (CUDA/CPU), seed
```

### Cell 2: Cấu hình Hyperparameters
```python
CONFIG = {
    'img_size': 48,
    'batch_size': 128,
    'epochs': 50,
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'num_classes': 43,
    'width_mult': 1.0,
    'checkpoint_dir': '/content/drive/MyDrive/mobilenetv2_gtsrb/checkpoints/',
    'log_dir': '/content/drive/MyDrive/mobilenetv2_gtsrb/logs/',
}
```

### Cell 3: Data Loading & Augmentation
```
- Load ảnh từ folder GTSRB
- Train transforms: Resize, RandomRotation, RandomAffine, ColorJitter, 
  RandomHorizontalFlip (cẩn thận với biển báo), Normalize
- Test transforms: Resize, Normalize
- Split train → train/val (80/20)
- DataLoader với num_workers, pin_memory
- Visualize sample images + class distribution
```

### Cell 4: Định nghĩa MobileNetV2 từ đầu
```
- class ConvBNReLU6(nn.Sequential)
- class InvertedResidual(nn.Module)
    - Expansion (1×1 conv) → Depthwise (3×3 conv, groups) → Projection (1×1 conv, linear)
    - Skip connection khi stride=1 và in_channels == out_channels
- class MobileNetV2(nn.Module)
    - Build từ architecture table
    - Dropout trước FC
    - Kaiming initialization
- In model summary + tổng params
```

### Cell 5: Loss Function, Optimizer, Scheduler
```
- Loss: CrossEntropyLoss với label smoothing (0.1)
  hoặc Focal Loss (custom) để xử lý class imbalance
- Optimizer: SGD + Nesterov Momentum
- Scheduler: CosineAnnealingLR hoặc Warmup + Cosine Decay
```

### Cell 6: Training Loop với Logging & Checkpoint

```python
# Mỗi epoch log:
training_log = {
    'epoch': [],
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'lr': [],
    'epoch_time': [],
}

# Checkpoint mỗi N epoch + best model:
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_acc': best_val_acc,
    'training_log': training_log,
}

# Resume training từ checkpoint nếu có
```

**Chi tiết Training Loop:**
1. Forward pass → compute loss
2. Backward pass → gradient clipping (optional)
3. Update weights
4. Log metrics mỗi epoch
5. Validate trên val set
6. Save checkpoint (periodic + best)
7. Early stopping (patience = 10)
8. Print progress table mỗi epoch

### Cell 7: Training History Visualization
```
- Plot Loss curves (train vs val) 
- Plot Accuracy curves (train vs val)
- Plot Learning Rate schedule
- Plot epoch training time
```

### Cell 8: Evaluation trên Test Set
```
- Load best model từ checkpoint
- Đánh giá trên test set
- Tính: Accuracy, Precision, Recall, F1-score (macro + weighted)
- Classification Report (sklearn)
```

### Cell 9: Visualization Kết Quả
```
- Confusion Matrix (heatmap 43×43)
- Per-class Accuracy bar chart
- Top-5 lớp tốt nhất & kém nhất
- Sample predictions (đúng + sai) với ảnh gốc
```

### Cell 10: Model Analysis
```
- Tổng số parameters
- FLOPs estimation
- So sánh kích thước model
- Export model (save .pth)
```

---

## 4. Kỹ Thuật Training Nâng Cao

| Kỹ thuật | Chi tiết |
|:---|:---|
| **Label Smoothing** | ε = 0.1 để tránh overconfident |
| **Warmup** | 5 epochs linear warmup từ lr/10 → lr |
| **Cosine Decay** | Sau warmup, LR giảm theo cosine |
| **Data Augmentation** | RandomRotation(15°), ColorJitter, RandomAffine |
| **Gradient Clipping** | max_norm = 5.0 |
| **Weight Decay** | 1e-4 (L2 regularization) |
| **Dropout** | 0.2 trước FC layer |
| **Early Stopping** | Patience = 10 epochs |
| **Mixed Precision** | torch.cuda.amp (nếu GPU hỗ trợ) |

---

## 5. Checkpoint System

```
/content/drive/MyDrive/mobilenetv2_gtsrb/
├── checkpoints/
│   ├── best_model.pth          # Model tốt nhất (val_acc cao nhất)
│   ├── checkpoint_epoch_10.pth # Checkpoint mỗi 10 epoch
│   ├── checkpoint_epoch_20.pth
│   ├── checkpoint_latest.pth   # Checkpoint mới nhất (để resume)
│   └── training_log.json       # Full training history
└── logs/
    └── training_history.csv    # CSV log cho phân tích
```

> [!TIP]
> Checkpoint lưu trên Google Drive để không bị mất khi Colab disconnect. Hỗ trợ resume training từ epoch cuối.

---

## 6. Target Performance

| Metric | Target |
|:---|:---|
| **Test Accuracy** | ≥ 95% |
| **Training Time** | ~30-45 phút (T4 GPU) |
| **Model Size** | ~8-10 MB |
| **Total Parameters** | ~2.2M |

> [!NOTE]
> GTSRB là dataset tương đối "dễ" cho deep learning, MobileNetV2 custom nên đạt được >95% accuracy.

---

## 7. Output File

### [NEW] [mobilenetv2_gtsrb.ipynb](file:///c:/Users/minht/OneDrive/Desktop/angra_mobilenetv2/mobilenetv2_gtsrb.ipynb)

Notebook duy nhất, self-contained, bao gồm tất cả:
- Model architecture definition
- Data loading & preprocessing
- Training loop với logging
- Checkpoint save/load
- Evaluation & visualization
- Chạy hoàn chỉnh trên Google Colab

---

## Verification Plan

### Automated Tests
- Kiểm tra model có thể forward pass với input shape `(1, 3, 48, 48)`
- Kiểm tra output shape = `(1, 43)`
- Kiểm tra checkpoint save/load hoạt động đúng

### Manual Verification
- Upload notebook lên Google Colab
- Train và kiểm tra loss giảm, accuracy tăng
- Kiểm tra checkpoint lưu trên Google Drive
- Kiểm tra resume training từ checkpoint
- Đánh giá accuracy trên test set ≥ 95%

---

## Open Questions

> [!IMPORTANT]
> 1. **Input size**: Bạn muốn resize ảnh về **48×48** hay **64×64**? (48×48 nhanh hơn, 64×64 có thể accuracy cao hơn)
> 2. **Framework**: Bạn muốn dùng **PyTorch** hay **TensorFlow/Keras**?
> 3. **Thời gian train**: Bạn muốn train bao nhiêu epochs? (đề xuất 50 epochs)
> 4. **Google Drive**: Bạn có muốn checkpoint lưu trên Google Drive không? 
> 5. **Ngôn ngữ**: Comment/markdown trong notebook viết bằng **tiếng Việt** hay **tiếng Anh**?
