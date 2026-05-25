# Phần 2: Cấu trúc dữ liệu và pipeline dữ liệu

## 1. Mục tiêu của phần này

Phần này giải thích sâu về dữ liệu trước khi đi vào model:

```text
Dữ liệu nằm ở đâu?
Dữ liệu đã đi qua những bước tiền xử lý nào?
Notebook đọc label bằng cách nào?
Vì sao phải split train/val/test?
Vì sao augment phải làm đúng thời điểm?
Làm sao kiểm tra data có bị sai hoặc rò rỉ không?
```

Trong machine learning, model tốt chưa đủ. Nếu data bị tổ chức sai, split sai, label sai hoặc rò rỉ dữ liệu, kết quả có thể rất cao nhưng không đáng tin.

## 2. Tổng quan pipeline dữ liệu

Theo folder hiện tại và tài liệu hướng dẫn, pipeline dữ liệu có dạng:

```text
Ảnh ban đầu
-> lọc ảnh lỗi/không đạt
-> crop vùng biển báo
-> resize về 224x224
-> chuyển RGB 3 kênh
-> split train/val/test
-> train MobileNetV2 classifier
```

Folder tương ứng:

```text
DataFinal/
  Data/
  FilterData/
  CropData/
  ResizeData/
  RGBData/
  metaData/
  SplitData/
```

Ý nghĩa ngắn:

| Folder | Vai trò |
|---|---|
| `Data` | Ảnh gốc ban đầu |
| `FilterData` | Ảnh bị loại hoặc không đạt |
| `CropData` | Ảnh đã crop vùng biển báo |
| `ResizeData` | Ảnh đã resize |
| `RGBData` | Ảnh đã chuyển về RGB 3 kênh |
| `metaData` | CSV chứa tọa độ crop |
| `SplitData` | Dữ liệu cuối để train/val/test |

Notebook ver3 chủ yếu dùng:

```text
DataFinal/SplitData
```

Không train trực tiếp từ `Data`, `CropData`, `ResizeData`, hay `metaData`.

## 3. Vì sao cần nhiều bước tiền xử lý?

### 3.1. Lọc ảnh

Ảnh ban đầu có thể có lỗi:

```text
ảnh quá mờ
ảnh không có biển báo
ảnh sai class
ảnh bị hỏng file
ảnh quá tối hoặc không đủ thông tin
```

Nếu đưa ảnh lỗi vào train, model học nhiễu. Label nói một đằng, ảnh thể hiện một nẻo, gradient sẽ kéo model theo hướng sai.

Ví dụ:

```text
label = stop_sign
ảnh = biển parking
```

Model bị phạt nếu dự đoán parking, dù dự đoán đó đúng theo ảnh. Đây gọi là label noise.

### 3.2. Crop

Crop biến bài toán từ ảnh nguyên cảnh thành ảnh chứa chủ thể chính:

```text
ảnh gốc lớn -> vùng biển báo
```

Điều này phù hợp với bài toán classification đã nói ở phần 1.

### 3.3. Resize

CNN cần batch tensor có cùng kích thước:

```text
batch shape = [batch_size, channels, height, width]
```

Nếu ảnh có kích thước khác nhau, PyTorch không thể gom thành một batch tensor bình thường.

Resize về `224x224` giúp:

```text
mọi ảnh cùng kích thước
phù hợp với nhiều kiến trúc CNN phổ biến
dễ tính batch
dễ tính mean/std
```

### 3.4. RGB

Ảnh có thể ở nhiều mode:

```text
RGB
RGBA
grayscale
palette
```

Model MobileNetV2 trong notebook nhận input 3 kênh:

```text
[3, H, W]
```

Nếu ảnh grayscale chỉ có 1 kênh hoặc RGBA có 4 kênh, model sẽ lỗi shape. Chuyển RGB đảm bảo mọi ảnh có 3 kênh.

## 4. `metaData` là gì?

Folder:

```text
DataFinal/metaData/
```

có nhiều file CSV theo class, ví dụ:

```text
no_entry_meta.csv
stop_sign_meta.csv
priority_road_meta.csv
```

Một dòng có dạng:

```csv
image_name,folder,x1,x2,y1,y2
no_entry_0003.jpg,no_entry,226.9,448.41,24.08,273.72
```

Ý nghĩa:

| Cột | Ý nghĩa |
|---|---|
| `image_name` | Tên ảnh gốc |
| `folder` | Tên class/folder |
| `x1` | Tọa độ trái của bounding box |
| `x2` | Tọa độ phải của bounding box |
| `y1` | Tọa độ trên của bounding box |
| `y2` | Tọa độ dưới của bounding box |

Tọa độ này dùng để crop:

```python
cropped = image.crop((x1, y1, x2, y2))
```

Sau khi crop xong và có `SplitData`, notebook classification không cần dùng `metaData` nữa.

Nói chính xác:

```text
metaData thuộc pipeline chuẩn bị dữ liệu
SplitData thuộc pipeline huấn luyện model
```

## 5. `SplitData` là gì?

`SplitData` là dữ liệu cuối:

```text
SplitData/
  train/
  val/
  test/
```

Mỗi split lại có class folder:

```text
SplitData/train/no_entry/*.jpg
SplitData/train/stop_sign/*.jpg
SplitData/val/no_entry/*.jpg
SplitData/test/no_entry/*.jpg
```

Đây là cấu trúc chuẩn cho `ImageFolder`.

## 6. `ImageFolder` đọc label như thế nào?

Notebook dùng:

```python
from torchvision.datasets import ImageFolder
```

Ví dụ:

```python
train_full = ImageFolder(train_dir, transform=train_transform)
```

`ImageFolder` giả định folder có dạng:

```text
root/class_name/image.jpg
```

Nó tự:

1. Liệt kê folder class.
2. Sắp xếp tên class theo alphabet.
3. Gán index cho từng class.
4. Mỗi ảnh nhận label theo folder chứa nó.

Ví dụ nếu class list là:

```python
['no_entry', 'no_stopping', 'no_vehicles', 'parking']
```

thì mapping:

```text
0 -> no_entry
1 -> no_stopping
2 -> no_vehicles
3 -> parking
```

Ảnh:

```text
train/no_entry/no_entry_0003.jpg
```

sẽ có label:

```text
0
```

Ảnh:

```text
train/parking/parking_0123.jpg
```

sẽ có label:

```text
3
```

### 6.1. Vì sao class order quan trọng?

Model chỉ biết output index, không biết tên class.

Nếu output cao nhất là index `0`, ta cần `CLASS_NAMES[0]` để biết đó là class gì.

Notebook lưu:

```python
CLASS_NAMES = raw_train_for_classes.classes
```

Vì vậy khi predict:

```python
CLASS_NAMES[pred_label]
```

trả về tên class đúng theo mapping.

Nếu train và test có mapping khác nhau, kết quả sẽ sai. Ví dụ train index 0 là `no_entry`, nhưng test index 0 lại là `parking`, lúc đó accuracy và confusion matrix vô nghĩa.

### 6.2. Vì sao train/val/test phải có cùng class folder?

Nếu train có 12 class nhưng test thiếu một class, vẫn có thể evaluate được nếu class mapping khớp, nhưng per-class metric class thiếu sẽ không có mẫu.

Nếu val/test có class lạ không có trong train, model không có output tương ứng. Đây là lỗi thiết kế dataset.

Vì vậy cần kiểm tra:

```text
train classes == val classes == test classes
```

## 7. Vai trò của `train.csv`, `val.csv`, `test.csv`

Trong `SplitData` có:

```text
train/train.csv
val/val.csv
test/test.csv
```

Ví dụ:

```csv
image_name,folder
no_entry_5773.jpg,test/no_entry/
```

Các CSV này là danh sách ảnh thuộc split tương ứng. Nhưng notebook đang dùng `ImageFolder`, nên label không lấy từ CSV.

CSV vẫn hữu ích để:

```text
kiểm tra số lượng ảnh
đối chiếu file
ghi nhận split
debug dữ liệu
```

Nhưng nguồn label thực tế khi train là:

```text
tên folder class
```

## 8. Split train/val/test là gì?

Split nghĩa là chia dữ liệu thành ba phần:

```text
train: dùng để cập nhật trọng số model
val: dùng để chọn model/early stopping/tuning
test: dùng để đánh giá cuối
```

Với tỷ lệ 70/15/15:

```text
70% ảnh -> train
15% ảnh -> validation
15% ảnh -> test
```

Sau khi chạy lại split từ `RGBData`, dữ liệu có:

```text
train: 8578 ảnh
val:   1839 ảnh
test:  1835 ảnh
total: 12252 ảnh
```

Tỷ lệ:

```text
8578 / 12252 ≈ 70.0%
1839 / 12252 ≈ 15.0%
1835 / 12252 ≈ 15.0%
```

## 9. Vì sao không dùng toàn bộ data để train?

Nếu dùng toàn bộ ảnh để train, ta không còn dữ liệu độc lập để đánh giá.

Model có thể học thuộc train set. Nếu chỉ đo trên train set, kết quả có thể rất cao nhưng không biết model có generalize không.

Vì vậy cần:

```text
train: học
val: theo dõi khi train
test: kiểm tra cuối cùng
```

## 10. Train, validation, test khác nhau thế nào?

### 10.1. Train set

Train set được dùng để cập nhật weight:

```python
loss.backward()
optimizer.step()
```

Mỗi epoch model đi qua train set một lần.

Train có thể được augment vì augment giúp model học nhiều biến thể.

### 10.2. Validation set

Validation set dùng trong quá trình train để:

```text
chọn checkpoint tốt nhất
early stopping
theo dõi overfit
```

Validation không cập nhật weight.

Không nên augment validation vì metric sẽ không ổn định và không phản ánh dữ liệu thật.

### 10.3. Test set

Test set chỉ nên dùng sau khi đã train/chọn model xong.

Test dùng để báo cáo kết quả cuối.

Không nên nhìn test nhiều lần để chỉnh hyperparameter, vì như vậy test dần trở thành validation trá hình.

## 11. Data leakage là gì?

Data leakage là khi thông tin từ val/test lọt vào quá trình train.

Ví dụ dễ gặp trong bài ảnh:

```text
ảnh gốc A
augment ra A_0001, A_0002, A_0003
split sau augment
train chứa A_0001
test chứa A_0002
```

Model thấy ảnh gần như giống test trong train. Test accuracy sẽ cao giả tạo.

Đây là lỗi nghiêm trọng vì test không còn độc lập.

## 12. Vì sao phải split trước augment?

Quy trình đúng:

```text
ảnh gốc -> split train/val/test -> augment train only
```

Hoặc:

```text
ảnh gốc -> split train/val/test -> online augment trong train DataLoader
```

Quy trình sai:

```text
ảnh gốc -> augment toàn bộ -> split train/val/test
```

Lý do sai: biến thể của cùng một ảnh có thể rơi vào cả train và test.

## 13. Augment lưu file và augment online

### 13.1. Augment lưu file

Script `augment_images.py` tạo ảnh mới trong folder:

```text
no_entry_0003.jpg
no_entry_0003_0001.jpg
no_entry_0003_0002.jpg
```

Ưu điểm:

- Dễ nhìn thấy ảnh augment.
- Không phải augment lại mỗi epoch.

Nhược điểm:

- Tốn ổ cứng.
- Nếu làm trước split dễ leakage.
- Dataset phình to.

### 13.2. Augment online

Notebook v3 có thể augment trong transform:

```python
'augment_enabled': 1
```

Khi DataLoader đọc ảnh, nó biến đổi ngẫu nhiên trong RAM:

```text
ảnh gốc -> biến thể tạm thời -> model
```

Không lưu file mới.

Ưu điểm:

- Không tốn ổ cứng.
- Mỗi epoch có thể có biến thể khác.
- Không rò rỉ sang val/test nếu chỉ áp dụng train.

Nhược điểm:

- Tốn thời gian CPU/GPU khi train.
- Khó tái hiện đúng từng ảnh augment.

## 14. Kiểm tra train đã bị augment sẵn chưa

Một dấu hiệu là tên file có hậu tố:

```text
no_entry_0003_0001.jpg
no_entry_0003_0002.jpg
```

Trong đó:

```text
no_entry_0003.jpg = ảnh gốc
no_entry_0003_0001.jpg = ảnh augment
```

Có thể kiểm tra bằng logic:

```python
import re
from pathlib import Path

root = Path('DataFinal/SplitData/train')
for p in root.rglob('*.jpg'):
    m = re.search(r'_(\d{4})$', p.stem)
    if m:
        base = p.with_name(p.stem[:m.start()] + p.suffix)
        if base.exists():
            print('augment:', p)
```

Nếu có nhiều ảnh dạng này trong train, khả năng train đã augment lưu file.

Sau khi chạy lại split từ `RGBData`, train còn 8578 ảnh và không còn mẫu augment dạng này.

## 15. Kiểm tra số lượng ảnh

Có thể đếm ảnh trong từng split:

```python
from pathlib import Path

root = Path('DataFinal/SplitData')
exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

for split in ['train', 'val', 'test']:
    count = sum(
        1 for p in (root / split).rglob('*')
        if p.suffix.lower() in exts
    )
    print(split, count)
```

Kết quả mong đợi sau split lại:

```text
train 8578
val 1839
test 1835
```

## 16. Kiểm tra CSV có khớp số ảnh không

Nếu CSV ghi 8578 dòng nhưng folder chỉ có 8000 ảnh, dữ liệu có vấn đề.

Code kiểm tra:

```python
import csv
from pathlib import Path

root = Path('DataFinal/SplitData')
exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

for split in ['train', 'val', 'test']:
    split_dir = root / split
    img_count = sum(1 for p in split_dir.rglob('*') if p.suffix.lower() in exts)

    csv_path = split_dir / f'{split}.csv'
    with csv_path.open('r', encoding='utf-8', newline='') as f:
        csv_count = sum(1 for _ in csv.DictReader(f))

    print(split, img_count, csv_count)
```

Nếu hai số bằng nhau, CSV và folder khớp về số lượng.

## 17. Kiểm tra class folder có khớp không

Code:

```python
from pathlib import Path

root = Path('DataFinal/SplitData')

for split in ['train', 'val', 'test']:
    classes = sorted([p.name for p in (root / split).iterdir() if p.is_dir()])
    print(split, len(classes), classes)
```

Kỳ vọng:

```text
train 12 [...]
val 12 [...]
test 12 [...]
```

Nếu train có 12 class nhưng test có 11 class, cần xem class nào thiếu.

## 18. Kiểm tra ảnh có đúng 224x224 không

Code:

```python
from pathlib import Path
from PIL import Image

root = Path('DataFinal/SplitData')
sizes = set()

for p in root.rglob('*.jpg'):
    with Image.open(p) as im:
        sizes.add(im.size)
    if len(sizes) > 10:
        break

print(sizes)
```

Kỳ vọng:

```text
{(224, 224)}
```

Nếu ảnh nhiều size khác nhau và `resize_enabled = 0`, DataLoader có thể lỗi khi ghép batch. Khi đó cần bật:

```python
'resize_enabled': 1
```

## 19. Kiểm tra ảnh có đúng RGB không

Code:

```python
from pathlib import Path
from PIL import Image

root = Path('DataFinal/SplitData')
modes = set()

for p in root.rglob('*.jpg'):
    with Image.open(p) as im:
        modes.add(im.mode)
    if len(modes) > 10:
        break

print(modes)
```

Kỳ vọng:

```text
{'RGB'}
```

Nếu có `L`, ảnh grayscale. Nếu có `RGBA`, ảnh 4 kênh. Notebook dùng `ImageFolder` với transform `ToTensor`, nếu ảnh không RGB có thể dẫn đến shape không đúng. Pipeline `RGBData` giúp tránh vấn đề này.

## 20. Kiểm tra file lỗi

Một ảnh hỏng có thể làm DataLoader dừng giữa training.

Code kiểm tra:

```python
from pathlib import Path
from PIL import Image

root = Path('DataFinal/SplitData')
bad = []

for p in root.rglob('*'):
    if p.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
        continue
    try:
        with Image.open(p) as im:
            im.verify()
    except Exception as e:
        bad.append((p, e))

print('bad files:', len(bad))
for p, e in bad[:10]:
    print(p, e)
```

Nếu `bad files > 0`, nên sửa hoặc xóa ảnh hỏng.

## 21. Class imbalance là gì?

Class imbalance xảy ra khi số ảnh mỗi class lệch nhiều.

Ví dụ:

```text
no_entry: 1198 ảnh train
speed_limit_50: 516 ảnh train
```

Class nhiều ảnh có thể được model học tốt hơn. Class ít ảnh có thể accuracy thấp hơn.

Notebook có phần vẽ phân bố lớp:

```python
label_counts = Counter(all_train_labels)
```

Nếu imbalance mạnh, có thể cân nhắc:

- Thu thập thêm ảnh cho class ít.
- Augment nhiều hơn cho class ít.
- Dùng weighted loss.
- Dùng weighted sampler.

Hiện tại notebook chưa dùng weighted loss.

## 22. Vì sao validation/test ít hơn train?

Vì split 70/15/15:

```text
train chiếm 70%
val chiếm 15%
test chiếm 15%
```

Train cần nhiều nhất vì dùng để học weight.

Val/test ít hơn nhưng vẫn đủ để đánh giá. Nếu val/test quá ít, metric dễ dao động. Với khoảng 1800 ảnh val/test, kết quả tương đối ổn.

## 23. Vì sao test không được augment?

Test phải mô phỏng dữ liệu thật trong phạm vi bài toán. Nếu augment test ngẫu nhiên:

- Mỗi lần chạy metric có thể khác.
- Test không còn là tập cố định.
- Khó so sánh model.

Test nên giữ cố định.

Nếu muốn đánh giá robust, có thể tạo một benchmark riêng:

```text
test_clean
test_noisy
test_blur
test_shifted
```

Nhưng đó là đánh giá bổ sung, không thay thế test gốc.

## 24. Vì sao notebook tính mean/std từ train source

Notebook dùng:

```python
TRAIN_SOURCE_DIR = train_dir if train_dir is not None else DATA_DIR
stats_dataset = ImageFolder(TRAIN_SOURCE_DIR, transform=stats_transform)
```

Tức mean/std chỉ tính từ train.

Lý do:

```text
val/test phải đóng vai trò dữ liệu chưa biết
```

Nếu dùng cả val/test để tính mean/std, thông tin thống kê từ val/test đã lọt vào pipeline train. Rò rỉ này nhẹ hơn rò rỉ ảnh augment, nhưng về nguyên tắc vẫn không sạch.

## 25. Vì sao notebook sample tối đa 3000 ảnh để tính mean/std

Notebook có:

```python
MEAN_STD_MAX_IMAGES = 3000
```

Nếu train rất lớn, tính mean/std toàn bộ có thể lâu. Lấy mẫu 3000 ảnh giúp nhanh hơn.

Ưu điểm:

```text
nhanh
đủ gần đúng nếu sample đại diện tốt
```

Nhược điểm:

```text
mean/std chỉ là ước lượng
```

Với dataset vài nghìn ảnh, có thể tăng lên toàn bộ train nếu muốn chính xác hơn.

## 26. Khi nào nên chạy lại split?

Nên chạy lại split nếu:

```text
SplitData/train đã bị augment lưu file nhưng bạn muốn data gốc
nghi ngờ augment trước split gây leakage
muốn đổi random_seed
muốn đổi tỷ lệ split
thêm/xóa ảnh trong RGBData
```

Không cần chạy lại split nếu:

```text
chỉ đổi batch_size
chỉ đổi learning rate
chỉ đổi augment online trong notebook
chỉ train lại model
```

## 27. Chạy lại split từ `RGBData`

Script gốc:

```text
Code/split_dataset.py
```

Ý tưởng:

```text
source_folder_split = DataFinal/RGBData
output_folder_split = DataFinal/SplitData
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15
random_seed = 42
clear_output_split = True
```

Khi `clear_output_split=True`, script xóa `SplitData` cũ rồi tạo lại. Cần chắc chắn không còn file cần giữ trong `SplitData` trước khi chạy.

## 28. Random seed trong split

Script split dùng:

```python
rng = random.Random(config["random_seed"])
rng.shuffle(shuffled_images)
```

`random_seed = 42` giúp lần chạy sau chia giống lần trước nếu dữ liệu đầu vào không đổi.

Nếu đổi seed, train/val/test sẽ khác. Kết quả model có thể hơi khác.

## 29. Split theo từng class

Script duyệt từng class folder và chia 70/15/15 trong class đó:

```python
for class_dir in iter_class_dirs(source_dir):
    images = iter_image_files(class_dir)
    rng.shuffle(shuffled_images)
    splits = split_images(...)
```

Điều này tốt hơn chia toàn bộ ảnh một lần, vì mỗi class đều có tỷ lệ train/val/test gần 70/15/15.

Nếu chia toàn bộ ảnh không stratified, class ít ảnh có thể bị thiếu ở val/test.

## 30. Rounding trong split

Script dùng:

```python
train_count = round(total * train_ratio)
val_count = round(total * val_ratio)
test_count = total - train_count - val_count
```

Vì số ảnh mỗi class là số nguyên, không thể chia đúng tuyệt đối 70/15/15 cho từng class. `round` làm tròn, phần còn lại đưa vào test.

Ví dụ class có 1711 ảnh:

```text
train = round(1711 * 0.70) = 1198
val = round(1711 * 0.15) = 257
test = 1711 - 1198 - 257 = 256
```

Tổng vẫn đúng:

```text
1198 + 257 + 256 = 1711
```

## 31. Vì sao kiểm tra overlap quan trọng?

Script có:

```python
check_no_overlap(splits)
```

Nó đảm bảo cùng một file gốc không xuất hiện ở nhiều split.

Nếu một ảnh xuất hiện cả train và test:

```text
model có thể học thuộc ảnh đó
test accuracy bị ảo
```

Tuy nhiên, `check_no_overlap` chỉ kiểm tra cùng file path. Nếu có ảnh augment lưu file với tên khác nhưng nội dung gần giống, check này không phát hiện được leakage ngữ nghĩa. Vì vậy vẫn cần đảm bảo augment diễn ra sau split.

## 32. Data tốt cho bài này cần điều kiện gì?

Dữ liệu tốt nên có:

```text
label đúng
ảnh crop đủ biển báo
ảnh cùng format RGB
ảnh cùng hoặc được resize về cùng size
train/val/test không overlap
mỗi class có đủ ảnh
val/test đại diện cho tình huống cần đánh giá
```

Dữ liệu chưa tốt khi:

```text
label sai nhiều
ảnh hỏng
class thiếu ảnh
ảnh quá giống nhau giữa train và test
crop mất ký hiệu chính
test quá dễ so với thực tế
```

## 33. Cách trình bày trong báo cáo

Có thể viết:

```text
Dữ liệu sau tiền xử lý được tổ chức theo cấu trúc ImageFolder gồm ba tập train, validation và test. Mỗi tập chứa các thư mục con tương ứng với từng lớp biển báo. Nhãn của ảnh được xác định từ tên thư mục chứa ảnh. Dữ liệu được chia theo tỷ lệ 70/15/15 từ tập RGBData đã được crop, resize về 224x224 và chuyển sang RGB. Việc chia dữ liệu được thực hiện trước khi augment để tránh rò rỉ dữ liệu giữa train và validation/test.
```

Nếu nói về CSV:

```text
Các file train.csv, val.csv và test.csv được dùng để ghi nhận danh sách ảnh trong từng split, trong khi quá trình huấn luyện sử dụng torchvision ImageFolder nên nhãn được lấy trực tiếp từ cấu trúc thư mục.
```

Nếu nói về augment:

```text
Augmentation chỉ được áp dụng cho tập train. Validation và test được giữ nguyên để đảm bảo quá trình đánh giá ổn định và không bị rò rỉ thông tin.
```

## 34. Cách nói khi thuyết trình

Có thể nói:

```text
Sau khi crop và resize, em dùng RGBData để chia dữ liệu thành train, validation và test theo tỷ lệ 70/15/15. Mỗi class là một folder riêng, nên khi dùng ImageFolder, label được lấy từ tên folder. Train dùng để cập nhật trọng số, validation dùng để chọn checkpoint tốt nhất và early stopping, test dùng để đánh giá cuối cùng. Augmentation nếu dùng thì chỉ áp dụng cho train để tránh rò rỉ dữ liệu.
```

Nếu bị hỏi “CSV để làm gì?”:

```text
CSV ghi lại danh sách ảnh trong từng split để kiểm tra và quản lý dữ liệu. Còn khi train bằng ImageFolder, nhãn thực tế được lấy từ cấu trúc folder class.
```

Nếu bị hỏi “vì sao phải split trước augment?”:

```text
Nếu augment trước split, biến thể của cùng một ảnh có thể nằm ở cả train và test, làm test accuracy cao giả tạo. Vì vậy phải split ảnh gốc trước, rồi chỉ augment tập train.
```

## 35. Checklist kiểm tra dữ liệu

Trước khi train:

```text
[ ] DATA_DIR trỏ đúng SplitData
[ ] train/val/test đều tồn tại
[ ] train/val/test có cùng danh sách class
[ ] số ảnh trong CSV khớp số file ảnh
[ ] ảnh là RGB
[ ] ảnh là 224x224 hoặc resize_enabled=1
[ ] train không chứa augment lưu file nếu muốn dùng data gốc
[ ] augment online chỉ bật nếu cần
[ ] không augment val/test
[ ] mean/std tính từ train
```

## 36. Kết luận phần 2

Cấu trúc dữ liệu là nền móng của toàn bộ notebook.

Điểm quan trọng nhất:

```text
Notebook train bằng ImageFolder, label lấy từ folder class.
SplitData là dữ liệu cuối để train.
Train/val/test phải tách độc lập.
Augment phải chỉ áp dụng train.
Mean/std nên tính từ train.
```

Nếu các điểm này đúng, kết quả train/test mới đáng tin. Nếu một trong các điểm này sai, model có thể đạt accuracy cao nhưng kết luận khoa học không vững.
