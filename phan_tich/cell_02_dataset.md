# CELL 2: TẢI DATASET GTSRB

## 1. Mã nguồn

```python
import torchvision

DATA_DIR = '/content/gtsrb_data'
os.makedirs(DATA_DIR, exist_ok=True)

raw_train = torchvision.datasets.GTSRB(root=DATA_DIR, split='train', download=True)
raw_test = torchvision.datasets.GTSRB(root=DATA_DIR, split='test', download=True)
```

## 2. Dataset GTSRB — Phân tích chi tiết

### 2.1 Thông tin tổng quan

| Thuộc tính | Chi tiết |
|:---|:---|
| **Tên đầy đủ** | German Traffic Sign Recognition Benchmark |
| **Năm công bố** | 2011 |
| **Tổ chức** | Institut für Neuroinformatik, Ruhr-Universität Bochum, Đức |
| **Số lớp** | 43 loại biển báo giao thông |
| **Train set** | 39,209 ảnh |
| **Test set** | 12,630 ảnh |
| **Kích thước ảnh** | 15×15 đến 250×250 pixels (không đồng nhất) |
| **Định dạng gốc** | PPM (Portable Pixmap) |
| **Dung lượng** | ~600 MB (sau giải nén) |

### 2.2 Cách thu thập dữ liệu

1. **Gắn camera** trên xe chạy trên đường phố Đức
2. **Quay video** liên tục khi xe di chuyển
3. **Phát hiện biển báo** trong video (bounding box)
4. **Crop** từng biển báo thành ảnh riêng
5. Mỗi biển báo có **~30 frames** liên tiếp (từ xa đến gần)

### 2.3 Phân loại 43 lớp

**Nhóm 1: Biển cấm (Prohibitory) — Hình tròn, viền đỏ**
| Class | Tên | Đặc điểm |
|:---:|:---|:---|
| 0-8 | Speed limit (20-120 km/h) | Số trên nền trắng |
| 9-10 | No passing | Xe đỏ+đen |
| 15-17 | No vehicles, No entry | Hình đặc trưng |

**Nhóm 2: Biển cảnh báo (Warning) — Hình tam giác, viền đỏ**
| Class | Tên | Đặc điểm |
|:---:|:---|:---|
| 18-31 | Caution, Curves, Road work... | Biểu tượng bên trong tam giác |

**Nhóm 3: Biển bắt buộc (Mandatory) — Hình tròn, nền xanh**
| Class | Tên | Đặc điểm |
|:---:|:---|:---|
| 33-40 | Turn, Keep, Roundabout... | Mũi tên trắng trên nền xanh |

**Nhóm 4: Biển đặc biệt**
| Class | Tên |
|:---:|:---|
| 12 | Priority road (hình thoi vàng) |
| 13 | Yield (tam giác ngược) |
| 14 | Stop (bát giác đỏ) |

### 2.4 Đặc điểm quan trọng của GTSRB

**Class Imbalance (Mất cân bằng lớp):**
- Class nhiều nhất: ~2,250 ảnh (Speed limit 50)
- Class ít nhất: ~210 ảnh (Speed limit 20, Dangerous curve left)
- Tỷ lệ chênh: ~10:1
- Tác động: Model có thể bias về class nhiều ảnh

**Video Correlation (Tương quan video):**
- Frames liên tiếp từ cùng 1 biển gần giống nhau
- Nếu random split → "data leakage" giữa train/val
- Test set lấy từ video KHÁC → đánh giá công bằng hơn

### 2.5 `torchvision.datasets.GTSRB`

```python
torchvision.datasets.GTSRB(root, split, download, transform)
```

| Tham số | Giá trị | Ý nghĩa |
|:---|:---|:---|
| `root` | `/content/gtsrb_data` | Thư mục lưu data |
| `split` | `'train'` / `'test'` | Chọn tập train hoặc test |
| `download` | `True` | Tự động download nếu chưa có |
| `transform` | `None` (ở cell này) | Transform áp dụng khi load ảnh |

**Cơ chế download:**
1. Kiểm tra `root` đã có data chưa
2. Nếu chưa → download từ server INI Bochum (~600MB zip)
3. Giải nén vào `root/gtsrb/`
4. Lần sau: phát hiện data đã tồn tại → bỏ qua download

## 3. Tác dụng
Cell này cung cấp **nguyên liệu thô** cho toàn bộ pipeline — 39,209 ảnh train và 12,630 ảnh test đã sẵn sàng để tiền xử lý ở cell tiếp theo.
