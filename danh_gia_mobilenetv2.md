# 📋 Đánh Giá Chi Tiết: MobileNetV2 Phân Loại Biển Báo Giao Thông

---

## 1. TỔNG QUAN

### Tại sao chọn MobileNetV2?

| Tiêu chí | MobileNetV2 | VGG16 | ResNet50 |
|:---|:---:|:---:|:---:|
| Parameters | **~2.2M** | 138M | 25.6M |
| Model size | **~9 MB** | 528 MB | 98 MB |
| Deploy mobile | **✅** | ❌ | ❌ |

**Ý nghĩa:** Biển báo cần nhận diện **real-time** trên thiết bị mobile. MobileNetV2 nhẹ gấp 60× so với VGG16 nhưng accuracy tương đương.

### Luồng dữ liệu

```
Input (3×96×96) → Conv 3×3 (32ch) → 7 stages Inverted Residual → Conv 1×1 (1280ch) → AvgPool → Dropout → FC (43 classes)
```

---

## 2. PHÂN TÍCH TỪNG THÀNH PHẦN

### 2.1 Conv2d (Convolution 2D)
Trượt filter qua ảnh để phát hiện đặc trưng.
- **Lớp đầu:** Cạnh ngang, dọc, góc, đường cong
- **Lớp giữa:** Hình tròn (biển cấm), tam giác (cảnh báo)
- **Lớp sâu:** Ngữ nghĩa — "biển Stop", "tốc độ 50km/h"

### 2.2 Depthwise Separable Convolution

| | Conv thường | Depthwise Separable |
|:---|:---:|:---:|
| Cách làm | 1 bước | 2 bước (Depthwise + Pointwise) |
| Params (32→64, 3×3) | **18,432** | **2,336** (giảm 8×) |

- **Depthwise** (groups=channels): Mỗi channel tự học spatial — channel đỏ học cạnh đỏ
- **Pointwise** (1×1): Kết hợp channels — "cạnh đỏ + hình tròn = biển cấm"

### 2.3 Inverted Residual Block — Khối chính

```
Input (narrow, 24ch)
  → [1×1 Expand]     144ch + BN + ReLU6    ← Mở rộng "không gian suy nghĩ"
  → [3×3 Depthwise]  144ch + BN + ReLU6    ← Xử lý vị trí, hình dạng
  → [1×1 Project]    32ch  + BN (LINEAR!)  ← Nén giữ thông tin quan trọng
  + Skip Connection (nếu stride=1, in==out)
```

**"Inverted":** Narrow→Wide→Narrow (ngược ResNet: Wide→Narrow→Wide). Skip ở lớp narrow → tiết kiệm memory.

### 2.4 ReLU6
```
ReLU6(x) = min(max(0, x), 6)
```
Giới hạn output ≤ 6 → phù hợp **quantization** (float32→int8) cho deploy mobile.

### 2.5 Linear Bottleneck (Không ReLU ở projection)
ReLU phá hủy thông tin ở chiều thấp. Projection output ít channels → dùng linear giữ nguyên thông tin.

### 2.6 Skip Connection
```python
return x + self.conv(x)  # Cộng input vào output
```
Giải quyết **vanishing gradient** — gradient "chảy thẳng" qua các lớp. Model chỉ cần học **phần khác biệt** F(x)-x.

### 2.7 Batch Normalization
Chuẩn hóa output mỗi lớp (mean=0, std=1) → train nhanh, ổn định, cho phép LR lớn hơn.

### 2.8 Adaptive Average Pooling
Lấy trung bình toàn bộ feature map → 1 số/channel. Biển Stop ở góc trái hay phải đều cho cùng kết quả.

### 2.9 Dropout (p=0.2)
Tắt ngẫu nhiên 20% neurons → buộc mỗi neuron tự học features hữu ích → chống overfit.

---

## 3. BẢNG KIẾN TRÚC

| Stage | Operator | t | c | n | s | Output |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 0 | Conv2d 3×3 | - | 32 | 1 | 1 | 96×96 |
| 1 | Inverted Residual | 1 | 16 | 1 | 1 | 96×96 |
| 2 | Inverted Residual | 6 | 24 | 2 | 2 | 48×48 |
| 3 | Inverted Residual | 6 | 32 | 3 | 2 | 24×24 |
| 4 | Inverted Residual | 6 | 64 | 4 | 2 | 12×12 |
| 5 | Inverted Residual | 6 | 96 | 3 | 1 | 12×12 |
| 6 | Inverted Residual | 6 | 160 | 3 | 2 | 6×6 |
| 7 | Inverted Residual | 6 | 320 | 1 | 1 | 6×6 |
| 8 | Conv2d 1×1 | - | 1280 | 1 | 1 | 6×6 |
| 9 | AvgPool + FC | - | 43 | - | - | 1×1 |

> **t** = expansion | **c** = output channels | **n** = repeat | **s** = stride

---

## 4. KỸ THUẬT TRAINING

| Kỹ thuật | Chi tiết | Tác dụng |
|:---|:---|:---|
| **Label Smoothing** (ε=0.1) | Target [0,0,0.91,0,0] thay vì [0,0,1,0,0] | Ngăn overconfident |
| **Warmup** (5 epochs) | LR: 0.002 → 0.01 tăng dần | Ổn định đầu training |
| **Cosine Decay** | LR giảm mượt theo cosine | Fine-tune cuối training |
| **SGD Nesterov** | "Nhìn trước" rồi cập nhật | Hội tụ nhanh hơn |
| **Weight Decay** (1e-4) | Phạt weights lớn | Chống overfit |
| **Gradient Clipping** (5.0) | Giới hạn gradient | Ngăn exploding gradient |
| **Mixed Precision** (AMP) | float16 thay float32 | Nhanh 2×, tiết kiệm VRAM |
| **Data Augmentation** | Rotation, Perspective, Blur, Erasing | Generalize tốt hơn |

---

## 5. CÂU HỎI VẤN ĐÁP BẢO VỆ ĐỀ TÀI

### Q1: "Depthwise Separable Conv khác gì Conv thường?"
> Tách thành 2 bước: Depthwise (spatial riêng mỗi channel) + Pointwise (kết hợp channels). Giảm ~8× params mà accuracy gần bằng.

### Q2: "Inverted Residual hoạt động thế nào?"
> Expand channels ×6 → Depthwise xử lý spatial → Project nén lại. Skip connection nối input/output giúp gradient chảy dễ. "Inverted" vì narrow→wide→narrow.

### Q3: "Tại sao không ReLU ở projection?"
> ReLU phá hủy thông tin ở không gian chiều thấp. Lớp projection ít channels → dùng linear giữ thông tin.

### Q4: "Label Smoothing tác dụng gì?"
> Ngăn model quá tự tin (100% 1 class). Dùng soft label → generalize tốt hơn trên dữ liệu mới.

### Q5: "Tại sao Warmup rồi Cosine?"
> Weights random → gradient không ổn định. Warmup tăng LR từ từ cho ổn, Cosine giảm dần để fine-tune chi tiết.

### Q6: "Val 99% có đáng tin?"
> Không hoàn toàn. GTSRB chụp từ video → frames giống nhau ở cả train/val. **Test accuracy** (video khác) mới là kết quả thực (~95-97%).

### Q7: "Triển khai thực tế thế nào?"
> Cần thêm bước Detection (YOLO) tìm vị trí biển → Crop → MobileNetV2 phân loại. Đây là pipeline chuẩn xe tự lái.

### Q8: "Tại sao input 96×96?"
> Ảnh GTSRB gốc nhỏ (~30-250px). 224×224 gây mờ do phóng to 7×. 96×96 cân bằng chi tiết và hiệu suất.

### Q9: "ReLU6 khác ReLU thường?"
> Giới hạn output ≤ 6. Phù hợp quantization INT8 cho mobile. ReLU thường output có thể rất lớn → khó biểu diễn fixed-point.

### Q10: "Mixed Precision có ảnh hưởng accuracy?"
> Không. Forward/backward dùng float16 (nhanh 2×), nhưng GradScaler giữ master weights ở float32 → accuracy như nhau.

---

## 6. ĐÁNH GIÁ TỔNG THỂ

### ✅ Điểm mạnh
- Model tự xây từ đầu, không dùng pretrained
- Checkpoint + Resume trên Google Drive
- Augmentation đa dạng, training techniques đầy đủ
- Visualization: loss, accuracy, confusion matrix, per-class

### ⚠️ Hạn chế
- Train trên GTSRB (biển Đức), chưa test biển Việt Nam
- Val accuracy cao ảo do đặc thù video GTSRB
- Chỉ phân loại, chưa có detection

### 🚀 Hướng phát triển
- Thêm YOLOv8 detection → pipeline hoàn chỉnh
- Fine-tune trên biển báo Việt Nam
- Quantization INT8 → deploy Raspberry Pi/mobile
- So sánh MobileNetV3, EfficientNet
