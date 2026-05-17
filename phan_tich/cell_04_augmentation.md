# CELL 4: DATA LOADING & AUGMENTATION

## 1. Mã nguồn chính

```python
train_transform = transforms.Compose([
    transforms.Resize((104, 104)),
    transforms.RandomCrop((96, 96)),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    transforms.Normalize(MEAN, STD),
])
```

## 2. Data Augmentation — Khái niệm

**Định nghĩa:** Kỹ thuật tạo ra các biến thể của ảnh training bằng cách áp dụng biến đổi ngẫu nhiên, nhằm tăng tính đa dạng của dữ liệu mà không cần thu thập thêm.

**Tại sao cần?**
- Dataset có giới hạn (39,209 ảnh) nhưng thực tế vô hạn tình huống
- Không augment → model "học thuộc" ảnh training → overfit
- Có augment → mỗi epoch model thấy ảnh khác nhau → học bản chất, không phải pixel

## 3. Phân tích từng kỹ thuật Augmentation

### 3.1 Resize(104) + RandomCrop(96)

```python
transforms.Resize((104, 104)),         # Resize lên 104×104
transforms.RandomCrop((96, 96)),       # Crop ngẫu nhiên 96×96
```

**Kỹ thuật:** Random Translation (dịch chuyển ngẫu nhiên)
- Resize 104 rồi crop 96 → dịch tối đa ±4 pixel theo mỗi hướng
- **Mô phỏng thực tế:** Biển báo không luôn nằm chính giữa ảnh. Camera có thể lệch, biển ở góc khung hình
- **Tác dụng:** Model học nhận diện biển ở bất kỳ vị trí nào, không chỉ chính giữa

### 3.2 RandomRotation(20)

```python
transforms.RandomRotation(20)  # Xoay ±20°
```

**Kỹ thuật:** Random Rotation
- Xoay ảnh ngẫu nhiên trong khoảng [-20°, +20°]
- **Mô phỏng thực tế:** Biển báo có thể bị nghiêng do gió, lắp đặt không chuẩn, camera nghiêng
- **Tác dụng:** Model bất biến với góc nghiêng nhẹ
- **Tại sao chỉ ±20°?** Biển báo hiếm khi nghiêng >20°. Xoay quá nhiều sẽ tạo ảnh phi thực tế

### 3.3 RandomAffine

```python
transforms.RandomAffine(
    degrees=0,              # Không xoay thêm (đã có RandomRotation)
    translate=(0.15, 0.15), # Dịch chuyển tối đa 15% theo x và y
    scale=(0.8, 1.2),       # Scale 80%-120%
    shear=10                # Biến dạng cắt ±10°
)
```

**Kỹ thuật:** Affine Transformation (biến đổi affine)
- Kết hợp: dịch chuyển + phóng to/thu nhỏ + shear (méo nghiêng)
- **translate:** Biển không ở tâm → dịch 15%
- **scale:** Biển ở xa (nhỏ, 80%) hoặc gần (lớn, 120%)
- **shear:** Nhìn biển từ góc xiên → méo hình bình hành
- **Tác dụng:** Model robust với khoảng cách và góc nhìn đa dạng

### 3.4 RandomPerspective

```python
transforms.RandomPerspective(distortion_scale=0.3, p=0.5)
```

**Kỹ thuật:** Perspective Transformation (biến đổi phối cảnh)
- Biến đổi 3D perspective — như nhìn biển từ vị trí khác nhau
- `distortion_scale=0.3`: Mức biến dạng 30%
- `p=0.5`: Chỉ áp dụng 50% ảnh
- **Mô phỏng thực tế:** Camera trên xe nhìn biển từ góc khác nhau (bên trái, bên phải, từ dưới lên)
- **Tác dụng:** Model học được biển báo ở mọi góc nhìn 3D

### 3.5 ColorJitter

```python
transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
```

**Kỹ thuật:** Color Augmentation
- **brightness=0.5:** Thay đổi độ sáng ±50% → nắng gắt hoặc ban đêm
- **contrast=0.5:** Thay đổi độ tương phản ±50% → sương mù hoặc rõ nét
- **saturation=0.5:** Thay đổi độ bão hòa màu ±50% → phai màu hoặc rực rỡ
- **hue=0.1:** Thay đổi tông màu ±10% → ánh sáng vàng buổi chiều, xanh bóng cây
- **Tác dụng:** Model nhận diện được biển trong mọi điều kiện ánh sáng

### 3.6 RandomGrayscale

```python
transforms.RandomGrayscale(p=0.05)  # 5% ảnh chuyển xám
```

**Kỹ thuật:** Grayscale Conversion
- 5% ảnh training bị chuyển thành grayscale (đen trắng)
- **Tác dụng:** Buộc model không CHỈ dựa vào màu sắc, mà phải học cả hình dạng
- **Ví dụ:** Biển Stop = bát giác. Nếu chỉ học "màu đỏ = Stop" → sai khi gặp ảnh đen trắng. Nếu học "bát giác + text STOP" → đúng cả khi không có màu

### 3.7 GaussianBlur

```python
transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
```

**Kỹ thuật:** Gaussian Blur (làm mờ Gaussian)
- Áp dụng bộ lọc Gaussian với kernel 3×3, sigma ngẫu nhiên 0.1-2.0
- **Mô phỏng thực tế:** Camera bị mờ do xe rung, mưa, lấy nét sai
- **Tác dụng:** Model robust với ảnh mờ, không chỉ ảnh nét

### 3.8 ToTensor + RandomErasing

```python
transforms.ToTensor(),                        # PIL Image → Tensor [0,1]
transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # Xóa 2-15% diện tích
```

**ToTensor:**
- Chuyển PIL Image (H×W×C, uint8 [0,255]) → PyTorch Tensor (C×H×W, float [0,1])
- Phải đặt TRƯỚC RandomErasing vì Erasing yêu cầu Tensor

**RandomErasing (Cutout):**
- 20% ảnh bị xóa một vùng nhỏ (2-15% diện tích) thay bằng random pixels
- **Mô phỏng thực tế:** Biển bị che một phần bởi cây, sticker, bùn, tuyết
- **Tác dụng:** Model học nhận diện biển ngay cả khi bị che 1 phần (occlusion)

### 3.9 Normalize

```python
transforms.Normalize(MEAN=[0.3403, 0.3121, 0.3214], STD=[0.2724, 0.2608, 0.2669])
```

**Kỹ thuật:** Channel-wise Normalization
- `pixel_normalized = (pixel - mean) / std` cho mỗi channel RGB
- MEAN và STD tính từ toàn bộ GTSRB training set
- **Tác dụng:** Input có mean≈0, std≈1 → gradient ổn định, training nhanh hơn

## 4. Val/Test Transform

```python
val_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
```

**Không augment** — đánh giá model trên ảnh gốc, không biến đổi. Nếu augment cả val/test → kết quả không phản ánh đúng năng lực model.

## 5. Train/Val Split

```python
val_size = int(0.15 * len(train_full))   # 15% cho validation
train_dataset, val_dataset = random_split(train_full, [train_size, val_size])
```

**Tại sao tách val từ train?**
- Không được dùng test set để chọn model → data leakage
- Val set: chọn best model, early stopping, monitor overfit
- Test set: đánh giá cuối cùng, chỉ dùng 1 lần

## 6. DataLoader

```python
DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
```

| Tham số | Giá trị | Ý nghĩa |
|:---|:---|:---|
| `shuffle` | True (train) / False (val/test) | Xáo trộn thứ tự → tránh model học thuộc pattern |
| `num_workers` | 2 | 2 CPU threads load data song song với GPU compute |
| `pin_memory` | True | Pinned memory → transfer CPU→GPU nhanh hơn |

## 7. Tác dụng tổng thể
Cell này biến raw images thành **training-ready data pipeline** với augmentation mạnh. Đây là cell quan trọng nhất cho việc model generalize tốt trên ảnh thực tế.
