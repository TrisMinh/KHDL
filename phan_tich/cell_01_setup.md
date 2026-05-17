# CELL 1: SETUP & CÀI ĐẶT

## 1. Mã nguồn

```python
import os, sys, time, json, csv, random, math, warnings
from pathlib import Path
import numpy as np
from PIL import Image
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')
```

## 2. Phân tích từng thành phần

### 2.1 Import thư viện chuẩn Python

| Thư viện | Kiểu | Vai trò trong project |
|:---|:---|:---|
| `os` | Standard | Tạo thư mục checkpoint, kiểm tra file tồn tại, ghép đường dẫn |
| `sys` | Standard | Truy cập thông tin hệ thống (Python version, platform) |
| `time` | Standard | Đo thời gian training mỗi epoch (`time.time()`) |
| `json` | Standard | Serialize/deserialize training log (dict → JSON file) |
| `csv` | Standard | Ghi training history ra file CSV cho phân tích sau |
| `random` | Standard | Đặt random seed cho reproducibility |
| `math` | Standard | Hàm `math.cos()` tính cosine LR schedule, `math.pi` |
| `warnings` | Standard | Tắt FutureWarning, DeprecationWarning gây nhiễu output |
| `pathlib.Path` | Standard | Xử lý đường dẫn file cross-platform |

### 2.2 Import thư viện bên ngoài

| Thư viện | Vai trò |
|:---|:---|
| `numpy` | Tính toán ma trận: confusion matrix, thống kê, normalize |
| `PIL.Image` | Đọc/ghi ảnh, convert format (RGB, resize) |

### 2.3 Tắt warnings

```python
warnings.filterwarnings('ignore')
```

**Tại sao tắt?** Torchvision và PyTorch thường in ra FutureWarning về API sẽ thay đổi trong phiên bản tương lai. Những warning này không ảnh hưởng chức năng nhưng gây rối output → tắt để output sạch hơn.

**Lưu ý:** Trong production code, KHÔNG nên tắt warning vì có thể bỏ lỡ cảnh báo quan trọng. Ở đây tắt vì đây là notebook thực nghiệm.

### 2.4 Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cơ chế hoạt động:**
1. `drive.mount()` gọi OAuth2 authentication → hiện popup yêu cầu đăng nhập Google
2. Sau khi xác thực, Google Drive được mount như ổ đĩa tại `/content/drive/`
3. Mọi file đọc/ghi tại `/content/drive/MyDrive/...` sẽ tự động đồng bộ với Drive

**Tại sao cần Mount Drive?**

| Vấn đề | Không có Drive | Có Drive |
|:---|:---|:---|
| Colab timeout (90 phút) | **Mất tất cả** training progress | Checkpoint trên Drive, **resume được** |
| Colab crash | Mất model, log, kết quả | Model lưu trên Drive, an toàn |
| Chuyển máy | Phải train lại từ đầu | Download checkpoint, tiếp tục train |

**Cấu trúc thư mục trên Drive:**
```
/content/drive/MyDrive/mobilenetv2_gtsrb/
├── checkpoints/
│   ├── best_model.pth           ← Model tốt nhất (val acc cao nhất)
│   ├── checkpoint_latest.pth    ← Checkpoint mới nhất (để resume)
│   ├── checkpoint_epoch_10.pth  ← Backup mỗi 10 epoch
│   └── training_log.json       ← Log toàn bộ training
└── logs/
    ├── training_history.csv     ← History dạng CSV
    └── *.png                    ← Biểu đồ visualization
```

## 3. Khái niệm liên quan

### Reproducibility (Tái lập kết quả)
Trong nghiên cứu khoa học, kết quả phải **tái lập được** — chạy lại code cho cùng kết quả. Cell này đặt nền tảng bằng việc import `random` (sẽ set seed ở cell sau).

### Persistence (Lưu trữ bền vững)
Mount Drive tạo kênh persistence — dữ liệu sống sót qua các session. Đây là pattern chuẩn khi dùng Colab cho training dài.

## 4. Tác dụng tổng thể
Cell này thiết lập **môi trường làm việc**: import công cụ cần thiết, kết nối lưu trữ. Giống như chuẩn bị bàn làm việc trước khi bắt đầu thí nghiệm.
