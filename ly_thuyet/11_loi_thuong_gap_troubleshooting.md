# Phần 11 - Lỗi Thường Gặp Và Cách Sửa

## 1. Mục tiêu phần này

Phần này gom các lỗi rất dễ gặp khi chạy notebook train MobileNetV2 trên Colab:

- lỗi load dữ liệu từ Google Drive;
- lỗi giải nén `.rar`, `.zip`, `.tar`;
- lỗi sai đường dẫn;
- lỗi CUDA out of memory;
- lỗi runtime còn giữ VRAM sau khi crash;
- lỗi batch size quá lớn;
- lỗi resize/stride làm tốn bộ nhớ;
- lỗi normalize/test ảnh ngoài;
- lỗi class mapping;
- lỗi kết quả cao bất thường;
- lỗi tiếng Việt bị hỏng encoding.

Đây là phần nên đọc khi notebook báo lỗi hoặc kết quả nhìn "kỳ kỳ".

---

## 2. Drive đã mount nhưng vẫn chậm

Thông báo:

```text
Drive already mounted at /content/drive
```

Không phải lỗi. Nó chỉ nói Google Drive đã được mount rồi.

Drive chậm vì khi đọc nhiều ảnh nhỏ trực tiếp từ Drive, Colab phải gọi file qua mount nhiều lần. Việc này chậm hơn đọc từ ổ local `/content`.

Cách tốt hơn:

```text
1. Để file nén trên Drive.
2. Copy file nén về /content.
3. Giải nén vào /content.
4. Train từ /content.
```

Lý do:

```text
copy một file nén lớn nhanh hơn đọc hàng chục nghìn file ảnh nhỏ từ Drive
```

---

## 3. Lỗi `archive_path is not defined`

Lỗi từng gặp:

```text
NameError: name 'archive_path' is not defined
```

Nguyên nhân:

```python
LOCAL_ARCHIVE_PATH = f'/content/{archive_path.name}'
```

được chạy trước khi biến `archive_path` được tạo.

Code đúng phải có:

```python
archive_path = Path(DRIVE_ARCHIVE_PATH)
LOCAL_ARCHIVE_PATH = f'/content/{archive_path.name}'
```

Quy tắc:

```text
biến phải được gán trước khi dùng
```

Nếu gặp lỗi này trong notebook, chạy lại cell load data từ đầu sau khi đã sửa thứ tự dòng.

---

## 4. Lỗi sai đường dẫn file nén

Ví dụ:

```python
DRIVE_ARCHIVE_PATH = '/content/drive/MyDrive/data_bien_bao.rar'
```

Nếu file thật nằm chỗ khác, notebook sẽ báo không tìm thấy file.

Cách kiểm tra trong Colab:

```python
from pathlib import Path

path = Path(DRIVE_ARCHIVE_PATH)
print(path.exists())
print(path)
```

Nếu `False`, đường dẫn sai.

Có thể dùng:

```python
!ls "/content/drive/MyDrive"
```

hoặc mở thanh file bên trái Colab để xem tên file thật.

Chú ý:

- tên file có dấu cách phải đặt trong dấu nháy nếu dùng shell;
- Python `Path` xử lý dấu cách tốt hơn;
- phân biệt `.rar`, `.zip`, `.tar.gz`.

---

## 5. Lỗi giải nén `.rar`

Colab thường chưa có sẵn công cụ giải nén rar.

Cần cài:

```python
!apt-get install -y unrar
```

Sau đó giải nén:

```python
!unrar x -o+ "/content/data_bien_bao.rar" "/content/data_bien_bao/"
```

Trong notebook, phần load data đã hỗ trợ `.rar` bằng cách gọi `unrar`.

Nếu báo:

```text
unrar: command not found
```

thì cell cài `unrar` chưa chạy hoặc chạy lỗi.

---

## 6. Lỗi file nén bị hỏng

Dấu hiệu:

```text
Unexpected end of archive
Cannot open file
CRC failed
```

Nguyên nhân có thể:

- upload lên Drive chưa xong;
- file nén bị lỗi;
- copy từ Drive về `/content` bị ngắt;
- file quá lớn và runtime bị reset giữa chừng.

Cách kiểm tra:

```python
print(Path(LOCAL_ARCHIVE_PATH).stat().st_size)
```

So sánh dung lượng với file trên máy.

Nếu nghi file hỏng:

- upload lại;
- nén lại;
- dùng `.zip` nếu không bắt buộc rar;
- kiểm tra giải nén được trên máy trước.

---

## 7. Lỗi không tìm thấy `SplitData`

Notebook cần folder kiểu:

```text
SplitData/
  train/
  val/
  test/
```

Nếu giải nén xong mà folder lồng nhiều tầng:

```text
/content/data_bien_bao/DataFinal/SplitData/train
```

thì notebook phải auto-detect hoặc mình phải trỏ đúng `DATA_DIR`.

Cách kiểm tra cây folder:

```python
from pathlib import Path

root = Path('/content/data_bien_bao')
for p in root.rglob('*'):
    if p.name == 'SplitData':
        print(p)
```

Nếu không in ra gì, file nén chưa chứa `SplitData` hoặc giải nén sai.

---

## 8. Lỗi class folder không đúng

`ImageFolder` yêu cầu cấu trúc:

```text
train/
  no_entry/
    image1.jpg
  stop_sign/
    image2.jpg
```

Nếu ảnh nằm trực tiếp trong `train`:

```text
train/
  image1.jpg
  image2.jpg
```

thì `ImageFolder` không hiểu class.

Với `ImageFolder`, tên folder con chính là tên class.

---

## 9. Lỗi số class không khớp

Nếu model tạo với:

```python
num_classes = 43
```

nhưng dataset có 12 class, output model là 43 logits trong khi label chỉ 0-11.

Training vẫn có thể chạy nhưng không đúng mục tiêu, hoặc inference mapping sai.

Notebook nên cập nhật:

```python
CONFIG['num_classes'] = len(train_dataset.classes)
```

Nếu dataset hiện tại có 12 class, model classifier cuối phải output 12.

---

## 10. Lỗi CUDA out of memory

Lỗi:

```text
OutOfMemoryError: CUDA out of memory
```

Nghĩa là GPU không đủ VRAM cho batch/model/input hiện tại.

Nguyên nhân thường gặp:

- batch size quá lớn;
- ảnh input quá lớn;
- model giữ feature map lớn do stride nhỏ;
- runtime còn giữ VRAM sau lỗi trước;
- chạy nhiều model/cell tạo model nhiều lần;
- không dùng mixed precision;
- DataLoader hoặc biến tensor lớn chưa được giải phóng.

---

## 11. Sau OOM cần restart runtime

Khi PyTorch OOM, nhiều khi GPU memory vẫn bị giữ.

Dù sửa batch size rồi, chạy lại vẫn OOM:

```text
GPU còn trống rất ít
```

Cách chắc nhất:

```text
Runtime -> Restart runtime
```

Sau đó chạy lại notebook từ đầu theo thứ tự.

Có thể thử:

```python
import gc, torch
gc.collect()
torch.cuda.empty_cache()
```

Nhưng nếu bộ nhớ đã rối sau OOM nặng, restart vẫn sạch nhất.

---

## 12. Batch size nên chỉnh ở đâu?

Trong Cell CONFIG:

```python
'batch_size': 32
```

Nếu OOM, giảm:

```text
32 -> 16 -> 8
```

Batch size nhỏ hơn:

- ít tốn VRAM hơn;
- mỗi epoch có nhiều step hơn;
- gradient noisy hơn một chút;
- train có thể chậm hơn về thời gian epoch;
- đôi khi cần learning rate nhỏ hơn.

Nếu dùng Colab T4 16GB, với MobileNetV2 224x224, batch size 32 thường an toàn hơn 128.

---

## 13. Vì sao stride ảnh hưởng VRAM?

Stride ở layer đầu quyết định feature map giảm kích thước nhanh hay chậm.

Nếu input 224x224:

### 13.1. Stride 1

```text
224x224 -> 224x224
```

Feature map vẫn lớn.

Các layer sau phải lưu activation lớn để backprop.

Tốn VRAM.

### 13.2. Stride 2

```text
224x224 -> 112x112
```

Feature map giảm 4 lần về số vị trí không gian:

```text
224 * 224 = 50176
112 * 112 = 12544
```

Nhỏ hơn 4 lần.

Vì vậy stride 2 giúp tiết kiệm VRAM rõ rệt.

---

## 14. Vì sao nhìn code toàn stride 1 nhưng vẫn OOM?

MobileNetV2 có nhiều block.

Không phải block nào cũng stride 2.

Nếu layer đầu để stride 1, feature map lớn kéo dài qua nhiều layer đầu. Dù sau đó có block stride 2, bộ nhớ activation ban đầu vẫn lớn.

Ngoài ra VRAM không chỉ do stride:

- batch size;
- số channel;
- expansion ratio;
- ảnh input;
- optimizer state;
- gradient;
- mixed precision bật/tắt.

Vì vậy thấy nhiều stride 1 không có nghĩa model nhẹ về VRAM.

---

## 15. Lỗi OOM ngay ở test forward dummy

Lỗi:

```python
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
out = model(dummy)
```

mà vẫn OOM, thường là GPU memory đã gần đầy từ lần chạy trước.

Vì batch 1 mà OOM thì khả năng cao:

- model cũ vẫn nằm trong VRAM;
- cell trước crash giữ memory;
- runtime chưa restart sau OOM;
- notebook đã tạo nhiều model/tensor.

Cách xử lý:

```text
Restart runtime
```

rồi chạy lại từ đầu.

---

## 16. Lỗi resume/checkpoint nhầm

Nếu `resume = True`, notebook có thể load checkpoint cũ.

Nếu muốn train mới hoàn toàn:

```python
'resume': False
```

và có thể xóa checkpoint/log cũ nếu notebook có cell hỗ trợ.

Nếu muốn train tiếp:

```python
'resume': True
```

Điều kiện:

- checkpoint tồn tại;
- kiến trúc model giống;
- số class giống;
- optimizer/scheduler tương thích.

Nếu đổi số class từ 43 sang 12 mà load checkpoint 43 class, sẽ lỗi shape ở layer cuối.

---

## 17. Dừng cell train giữa chừng có chạy cell sau được không?

Tùy.

Nếu đã có checkpoint best từ epoch trước, cell test có thể load best checkpoint và chạy.

Nhưng nếu dừng trước khi notebook lưu checkpoint nào, cell sau có thể không có model tốt để test.

Nếu dừng giữa epoch:

- epoch đó chưa hoàn tất;
- checkpoint cuối epoch đó thường chưa được lưu;
- best checkpoint trước đó vẫn còn nếu đã lưu.

An toàn nhất:

```text
chờ hết epoch hiện tại rồi dừng
```

Nếu bắt buộc dừng, kiểm tra folder checkpoint có file best không.

---

## 18. Lỗi accuracy cao bất thường

Ví dụ:

```text
epoch 1 val acc 98%
epoch 2 val acc 99%
```

Không tự động là lỗi.

Nhưng cần kiểm tra:

- dataset có quá dễ không;
- ảnh đã crop sạch không;
- có dùng pretrained không;
- train/val/test có cùng nguồn quá giống không;
- có leakage từ ảnh augment không;
- test có ảnh trùng train không;
- số class chỉ 12, không phải 43 không.

Nếu data đã crop sạch và class rõ, accuracy cao là có thể.

Nhưng cần test ảnh ngoài để kiểm chứng tính thực tế.

---

## 19. Lỗi train/val/test chia không đúng 70/15/15

Nếu thấy train nhiều bất thường, val/test ít, có thể do:

- train đã chứa ảnh augment tạo sẵn;
- split không chạy từ dữ liệu gốc;
- file CSV cũ;
- folder cũ chưa xóa trước khi split.

Sau khi chạy lại split từ `RGBData` gốc, số liệu hiện tại là:

```text
train: 8578
val:   1839
test:  1835
total: 12252
```

Tỉ lệ xấp xỉ:

```text
70% / 15% / 15%
```

Nếu train lại tăng lên hơn 40 nghìn, khả năng đã dùng lại data augment sẵn.

---

## 20. Lỗi augmentation không thấy khác

Nếu `augment_enabled = 0`, ảnh augment example sẽ giống ảnh gốc hoặc chỉ hiển thị transform cơ bản.

Muốn bật online augmentation:

```python
'augment_enabled': 1
```

trong Cell CONFIG.

Chú ý:

- augmentation online không tạo thêm file ảnh mới trong folder;
- mỗi lần lấy ảnh train, transform random có thể biến đổi ảnh khác nhau;
- số ảnh trên disk không tăng;
- số sample mỗi epoch vẫn bằng số ảnh train.

Nếu muốn thấy khác rõ trong visualization, bật augment và chạy lại cell dataset/visualization.

---

## 21. Lỗi hiểu nhầm "mỗi ảnh augment bao nhiêu"

Với online augmentation:

```text
không có số ảnh augment cố định được lưu ra disk
```

Mỗi epoch, mỗi ảnh có thể được biến đổi ngẫu nhiên một lần khi được lấy vào batch.

Nếu train 50 epoch, cùng một ảnh gốc có thể xuất hiện dưới nhiều biến thể khác nhau qua các epoch.

Nhưng trong một epoch, số sample vẫn là số ảnh train gốc.

Ví dụ:

```text
train có 8578 ảnh
batch size 32
1 epoch vẫn đi qua 8578 lần lấy mẫu
```

Không phải tự nhân thành 8578 * 8 nếu chỉ dùng online augmentation.

---

## 22. Lỗi normalize khi test ngoài

Nếu ảnh ngoài dự đoán sai hàng loạt, kiểm tra normalize.

Train dùng:

```python
Normalize(mean, std)
```

Inference cũng phải dùng đúng `mean`, `std`.

Nếu không normalize hoặc dùng mean/std khác, input distribution lệch.

Đặc biệt với pretrained ImageNet, mean/std thường là:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Nhưng notebook có thể tính mean/std từ train set. Phải dùng đúng cái notebook đã dùng.

---

## 23. Lỗi nhầm min-max với z-score

`ToTensor()` đã đưa pixel từ `[0, 255]` về `[0, 1]`.

`Normalize(mean, std)` là chuẩn hóa kiểu z-score theo kênh:

```text
x' = (x - mean) / std
```

Đây không phải min-max scaling thông thường.

Min-max:

```text
x' = (x - min) / (max - min)
```

Trong pipeline ảnh PyTorch:

```text
ToTensor -> Normalize
```

là cách rất phổ biến.

---

## 24. Lỗi dùng chung normalize cho mọi model

Không phải lúc nào cũng nên dùng chung.

Nếu model train from scratch:

```text
có thể dùng mean/std tính từ dataset của mình
```

Nếu model pretrained ImageNet:

```text
nên dùng mean/std ImageNet
```

Lý do:

```text
pretrained weights đã học trên input được normalize theo ImageNet
```

Nếu dùng normalize khác, phân phối input vào các layer pretrained lệch so với lúc pretrain.

Với MobileNetV2/ResNet nếu đều dùng pretrained ImageNet, cả hai thường dùng cùng ImageNet normalize. Nhưng nếu mỗi model hoặc codebase định nghĩa preprocessing riêng, cần theo đúng tài liệu/weight.

---

## 25. Lỗi ảnh ngoài không thuộc 12 class

Classifier luôn chọn một trong các class đã biết.

Nếu đưa ảnh biển báo không nằm trong 12 class:

```text
model vẫn bắt buộc chọn class gần nhất
```

Confidence cao không đảm bảo ảnh thuộc class đó.

Muốn xử lý unknown cần thiết kế thêm:

- class unknown;
- threshold;
- dữ liệu ngoài class;
- open-set recognition.

---

## 26. Lỗi class mapping sai

Nếu `ImageFolder` tạo class theo thứ tự alphabet:

```text
0 -> no_entry
1 -> no_stopping
2 -> no_vehicles
...
```

nhưng inference tự viết list khác thứ tự, tên class hiển thị sẽ sai.

Cách tốt:

```python
class_names = train_dataset.classes
```

và lưu vào checkpoint.

Khi load checkpoint:

```python
class_names = checkpoint['class_names']
```

---

## 27. Lỗi tiếng Việt bị hỏng dấu

Nếu file `.md`, `.py`, `.ipynb` bị lỗi dấu tiếng Việt, thường do encoding.

Nên dùng UTF-8.

Trong VS Code:

```text
góc dưới phải -> encoding -> Save with Encoding -> UTF-8
```

Trong Python đọc file:

```python
open(path, encoding='utf-8')
```

Trong notebook markdown, Colab thường hỗ trợ UTF-8 tốt.

Tránh copy qua môi trường dùng ANSI/Windows-1258 nếu không cần.

---

## 28. Lỗi cell chạy không theo thứ tự

Notebook phụ thuộc biến từ các cell trước.

Ví dụ cell train cần:

- `train_loader`;
- `val_loader`;
- `model`;
- `criterion`;
- `optimizer`;
- `device`.

Nếu chạy cell train trước cell tạo DataLoader/model, sẽ lỗi `NameError`.

Cách sửa:

```text
Runtime -> Restart runtime
Run all từ đầu theo thứ tự
```

Hoặc chạy lại các cell từ CONFIG đến trước cell bị lỗi.

---

## 29. Lỗi chỉnh CONFIG nhưng không có tác dụng

Nếu chỉnh:

```python
'batch_size': 16
```

nhưng không chạy lại cell tạo DataLoader, batch size cũ vẫn đang được dùng.

Quy tắc:

```text
chỉnh CONFIG -> phải chạy lại các cell phụ thuộc CONFIG
```

Ví dụ:

- chỉnh `batch_size` -> chạy lại cell DataLoader;
- chỉnh `img_size` -> chạy lại transform, dataset, model;
- chỉnh `augment_enabled` -> chạy lại transform/dataset;
- chỉnh `num_classes` -> chạy lại model;
- chỉnh `lr` -> chạy lại optimizer/scheduler.

---

## 30. Lỗi val/test vẫn dùng augment

Val/test không nên dùng random augmentation.

Nếu val/test dùng random transform, kết quả có thể:

- thay đổi mỗi lần chạy;
- không đo đúng dữ liệu gốc;
- bị khó hơn hoặc dễ hơn tùy random.

Chỉ train mới nên dùng augmentation.

Val/test nên deterministic:

```text
Resize nếu cần
ToTensor
Normalize
```

---

## 31. Lỗi report sai phạm vi bài toán

Không nên viết:

```text
Hệ thống phát hiện và phân loại biển báo giao thông.
```

Nếu notebook chỉ classification.

Nên viết:

```text
Mô hình phân loại biển báo giao thông từ ảnh đã được crop.
```

Nếu demo ảnh ngoài:

```text
Ảnh ngoài được crop quanh biển báo trước khi đưa vào mô hình.
```

Nếu muốn nói hướng phát triển:

```text
Có thể tích hợp thêm mô hình detection để tự động xác định vị trí biển báo trên ảnh nguyên cảnh.
```

---

## 32. Quy trình debug nhanh

Khi gặp lỗi, đi theo thứ tự:

1. Đọc dòng lỗi cuối cùng.
2. Xác định lỗi thuộc data, model, GPU, checkpoint hay inference.
3. Kiểm tra cell CONFIG đã đúng chưa.
4. Kiểm tra biến/cell trước đã chạy chưa.
5. In shape dữ liệu.
6. In số class.
7. In vài đường dẫn ảnh.
8. Nếu OOM, giảm batch size và restart runtime.
9. Nếu kết quả lạ, kiểm tra split và leakage.
10. Nếu test ngoài sai, kiểm tra crop/normalize/RGB/mapping.

---

## 33. Bảng lỗi nhanh

| Hiện tượng | Khả năng cao | Cách sửa |
|---|---|---|
| `archive_path is not defined` | Dùng biến trước khi gán | Tạo `archive_path = Path(...)` trước |
| Không tìm thấy file nén | Sai `DRIVE_ARCHIVE_PATH` | Kiểm tra path trên Drive |
| `unrar command not found` | Chưa cài unrar | `apt-get install -y unrar` |
| Không thấy `SplitData` | Giải nén sai tầng hoặc thiếu folder | Auto-detect hoặc trỏ đúng `DATA_DIR` |
| CUDA OOM | Batch/model/input quá lớn | Giảm batch, restart runtime |
| Batch 1 vẫn OOM | VRAM bị giữ sau crash | Restart runtime |
| Accuracy rất cao | Dữ liệu dễ hoặc leakage | Kiểm tra split, test ngoài |
| Test ngoài sai | Crop/preprocess sai | Crop lại, RGB, normalize đúng |
| Tên class sai | Mapping lệch | Dùng `train_dataset.classes` |
| Chỉnh config không đổi | Chưa chạy lại cell phụ thuộc | Run lại từ CONFIG trở xuống |

---

## 34. Kết luận phần 11

Hầu hết lỗi trong notebook không phải do MobileNetV2 khó, mà do pipeline:

```text
đường dẫn -> giải nén -> split -> transform -> model -> train -> checkpoint -> inference
```

Chỉ cần một bước lệch, kết quả có thể sai hoặc notebook crash.

Khi debug, đừng đoán quá nhanh. In ra:

- path;
- số ảnh;
- số class;
- shape tensor;
- batch size;
- device;
- checkpoint path;
- class mapping;
- mean/std.

Những thông tin này thường chỉ ra lỗi rất nhanh.
