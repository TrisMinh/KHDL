# Phần 3: Load dữ liệu từ Google Drive, giải nén và tìm đúng dataset

## 1. Mục tiêu của phần này

Phần này giải thích sâu Cell 2 trong notebook:

```text
LOAD DATASET TỪ GOOGLE DRIVE + GIẢI NÉN VỀ LOCAL COLAB
```

Cell này có nhiệm vụ:

```text
1. Mount Google Drive
2. Trỏ tới file nén dataset trên Drive
3. Copy file nén về local Colab
4. Giải nén vào /content
5. Xem nhanh cấu trúc sau giải nén
6. Tự tìm đúng thư mục SplitData để train
```

Đây là cell rất quan trọng. Nếu Cell 2 chọn sai folder, các cell sau vẫn có thể chạy nhưng train sai dữ liệu.

## 2. Vì sao không đọc trực tiếp từ Google Drive?

Trong Colab, khi mount Drive:

```python
drive.mount('/content/drive')
```

ta có thể truy cập file qua:

```text
/content/drive/MyDrive/...
```

Nhưng Google Drive mount trong Colab thường có I/O chậm hơn ổ local của runtime:

```text
Drive mount: tiện lưu trữ lâu dài nhưng đọc nhiều file nhỏ chậm
/content: ổ local tạm thời của runtime, đọc nhanh hơn
```

Dataset ảnh thường có rất nhiều file nhỏ. Khi training, DataLoader phải đọc ảnh liên tục. Nếu đọc trực tiếp từ Drive, có thể bị:

```text
train chậm
DataLoader chờ I/O
GPU rảnh nhưng CPU/Drive đang đọc ảnh
epoch kéo dài
```

Vì vậy notebook làm đúng hướng:

```text
file nén trên Drive -> copy về /content -> giải nén -> train từ /content
```

## 3. Google Drive và `/content` khác nhau thế nào?

### 3.1. Google Drive

Đường dẫn:

```text
/content/drive/MyDrive/
```

Đặc điểm:

```text
lưu lâu dài
không mất khi runtime tắt
đọc/ghi chậm hơn local
phù hợp để lưu file nén, checkpoint, logs
```

### 3.2. Local Colab `/content`

Đường dẫn:

```text
/content/
```

Đặc điểm:

```text
đọc/ghi nhanh hơn
phù hợp để giải nén dataset và train
mất khi runtime bị xóa/reset
```

Do đó:

```text
dataset nén: lưu trên Drive
dataset đã giải nén: để ở /content
checkpoint/log: lưu trên Drive
```

## 4. Runtime/session là gì?

Mỗi lần Colab cấp máy cho notebook, đó là một runtime/session.

Khi runtime còn sống:

```text
/content/data_bien_bao vẫn còn
model trong RAM/GPU vẫn còn
biến Python vẫn còn
```

Khi restart runtime:

```text
biến Python mất
model trong RAM mất
GPU memory được reset
các file trong /content thường vẫn có thể còn nếu chỉ restart mềm, nhưng không nên phụ thuộc hoàn toàn
```

Khi disconnect and delete runtime:

```text
/content mất
phải copy và giải nén lại dataset
Drive vẫn còn file nén/checkpoint/log
```

Vì vậy notebook lưu checkpoint ở Drive:

```python
'checkpoint_dir': '/content/drive/MyDrive/mobilenetv2_gtsrb/checkpoints/'
```

để không mất model khi runtime mất.

## 5. Các biến đường dẫn trong Cell 2

Cell 2 có các biến:

```python
DRIVE_ARCHIVE_PATH = '/content/drive/MyDrive/data_bien_bao.rar'
archive_path = Path(DRIVE_ARCHIVE_PATH)
LOCAL_ARCHIVE_PATH = f'/content/{archive_path.name}'
EXTRACT_ROOT = Path('/content/data_bien_bao')
```

### 5.1. `DRIVE_ARCHIVE_PATH`

Đây là đường dẫn file nén trên Google Drive.

Ví dụ:

```python
DRIVE_ARCHIVE_PATH = '/content/drive/MyDrive/data_bien_bao.rar'
```

Nếu bạn để file trong folder:

```text
MyDrive/Datasets/data_bien_bao.rar
```

thì phải sửa:

```python
DRIVE_ARCHIVE_PATH = '/content/drive/MyDrive/Datasets/data_bien_bao.rar'
```

Nếu đường dẫn sai, notebook báo:

```python
FileNotFoundError
```

### 5.2. `archive_path`

```python
archive_path = Path(DRIVE_ARCHIVE_PATH)
```

Biến này biến string đường dẫn thành object `Path`, giúp xử lý tên file, suffix, kiểm tra tồn tại.

Ví dụ:

```python
archive_path.name
```

trả về:

```text
data_bien_bao.rar
```

### 5.3. `LOCAL_ARCHIVE_PATH`

```python
LOCAL_ARCHIVE_PATH = f'/content/{archive_path.name}'
```

Nếu file Drive là:

```text
data_bien_bao.rar
```

thì file local sẽ là:

```text
/content/data_bien_bao.rar
```

Lý do dùng `archive_path.name`: giữ nguyên tên và đuôi file. Nếu file là `.zip`, local vẫn là `.zip`; nếu `.rar`, local vẫn là `.rar`.

### 5.4. `EXTRACT_ROOT`

```python
EXTRACT_ROOT = Path('/content/data_bien_bao')
```

Đây là thư mục giải nén dataset.

Sau giải nén có thể có cấu trúc:

```text
/content/data_bien_bao/DataFinal/SplitData/...
```

hoặc:

```text
/content/data_bien_bao/SplitData/...
```

Notebook sẽ tự tìm `SplitData`.

## 6. Mount Drive

Code:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Nếu Drive đã mount, Colab có thể in:

```text
Drive already mounted at /content/drive
```

Đây không phải lỗi. Nó chỉ nói Drive đã được mount rồi.

Nếu muốn mount lại cưỡng bức:

```python
drive.mount('/content/drive', force_remount=True)
```

Nhưng bình thường không cần.

## 7. Kiểm tra file có tồn tại không

Code:

```python
if not archive_path.exists():
    raise FileNotFoundError(...)
```

Mục đích là dừng sớm nếu đường dẫn sai.

Nếu không kiểm tra, code có thể lỗi ở `shutil.copy2` với thông báo khó hiểu hơn.

Lỗi thường gặp:

```text
FileNotFoundError: Không tìm thấy file nén trên Drive
```

Cách xử lý:

```text
1. Kiểm tra tên file có đúng không
2. Kiểm tra file nằm trong MyDrive hay folder con
3. Kiểm tra đuôi .rar/.zip có đúng không
4. Copy path trong file browser của Colab nếu cần
```

## 8. Copy file nén về local

Code:

```python
shutil.copy2(archive_path, LOCAL_ARCHIVE_PATH)
```

`copy2` copy cả nội dung và metadata cơ bản.

Về mặt train, dùng `copy` hay `copy2` đều được. `copy2` chỉ đầy đủ hơn.

Sau copy:

```python
print(f'Dung lượng: {Path(LOCAL_ARCHIVE_PATH).stat().st_size / (1024**3):.2f} GB')
```

Dung lượng giúp biết file đã copy thật chưa, tránh trường hợp file 0 byte hoặc copy nhầm file.

## 9. Vì sao file nén 2GB vẫn nên copy?

Copy file 2GB từ Drive về local có thể mất vài phút, nhưng chỉ làm một lần mỗi runtime.

Nếu không copy mà đọc trực tiếp nhiều ảnh từ Drive trong training, mỗi epoch đều bị chậm.

So sánh:

```text
copy + unzip: tốn thời gian ban đầu
train từ /content: nhanh trong toàn bộ quá trình train
```

Với dataset ảnh, cách này thường lợi hơn nhiều.

## 10. Giải nén file

Cell 2 xác định định dạng bằng:

```python
suffixes = ''.join(Path(LOCAL_ARCHIVE_PATH).suffixes).lower()
```

Ví dụ:

```text
data.zip -> .zip
data.tar.gz -> .tar.gz
data.rar -> .rar
```

Dùng `suffixes` thay vì `suffix` để nhận đúng `.tar.gz`, vì `Path(...).suffix` của `data.tar.gz` chỉ trả `.gz`.

## 11. Giải nén `.zip`

Code:

```python
with zipfile.ZipFile(LOCAL_ARCHIVE_PATH, 'r') as zf:
    zf.extractall(EXTRACT_ROOT)
```

Python có sẵn `zipfile`, không cần cài thêm.

Ưu điểm `.zip`:

```text
hỗ trợ sẵn
ít lỗi trên Colab
dễ dùng
```

## 12. Giải nén `.tar`, `.tar.gz`, `.tgz`

Code:

```python
with tarfile.open(LOCAL_ARCHIVE_PATH, 'r:*') as tf:
    tf.extractall(EXTRACT_ROOT)
```

`tarfile` cũng có sẵn trong Python.

`r:*` cho phép tự nhận dạng kiểu nén tar.

## 13. Giải nén `.rar`

Python chuẩn không có module built-in để giải nén RAR. Vì vậy notebook cài `unrar` trên Colab:

```python
subprocess.run(['apt-get', '-qq', 'update'], check=True)
subprocess.run(['apt-get', '-qq', 'install', '-y', 'unrar'], check=True)
subprocess.run(['unrar', 'x', '-o+', LOCAL_ARCHIVE_PATH, str(EXTRACT_ROOT) + '/'], check=True)
```

Ý nghĩa:

```text
apt-get update: cập nhật danh sách package
apt-get install -y unrar: cài công cụ unrar
unrar x: giải nén giữ cấu trúc thư mục
-o+: ghi đè nếu file đã tồn tại
```

Nếu Colab không cài được `unrar`, có thể đổi file sang `.zip` hoặc `.tar.gz`.

## 14. `subprocess.run(..., check=True)` là gì?

`subprocess.run` chạy lệnh hệ thống từ Python.

`check=True` nghĩa là nếu lệnh thất bại, Python sẽ raise lỗi ngay.

Điều này tốt hơn im lặng chạy tiếp, vì nếu giải nén lỗi mà notebook vẫn đi tiếp, các cell sau có thể train trên folder rỗng hoặc folder sai.

## 15. Preview cây thư mục sau giải nén

Notebook có:

```python
preview_tree(EXTRACT_ROOT)
```

Hàm này in cây thư mục tối đa vài cấp:

```text
[D] DataFinal
  [D] SplitData
    [D] train
    [D] val
    [D] test
```

Mục đích:

```text
nhìn nhanh file nén giải ra có đúng cấu trúc không
phát hiện nén lồng folder quá sâu
phát hiện thiếu SplitData
phát hiện giải nén nhầm file
```

## 16. Tự tìm `DATA_DIR`

Notebook không giả định `SplitData` nằm đúng một vị trí cố định. Nó tìm trong `EXTRACT_ROOT`.

Logic ưu tiên:

```text
1. Folder tên SplitData có train/<class>/*.jpg
2. Bất kỳ folder nào có train/<class>/*.jpg
3. Folder trực tiếp có <class>/*.jpg
```

Lý do cần tự tìm:

File nén có thể giải ra nhiều kiểu:

```text
/content/data_bien_bao/SplitData/train/...
```

hoặc:

```text
/content/data_bien_bao/DataFinal/SplitData/train/...
```

hoặc:

```text
/content/data_bien_bao/data_bien_bao/DataFinal/SplitData/train/...
```

Nếu hard-code một đường dẫn, rất dễ sai.

## 17. Hàm `count_images`

Code:

```python
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def count_images(folder):
    return sum(1 for p in Path(folder).rglob('*') if p.suffix.lower() in IMAGE_EXTS)
```

Hàm này đếm ảnh trong folder và mọi folder con.

Vai trò:

```text
kiểm tra folder có ảnh thật không
in tổng số ảnh tìm thấy
giúp xác định class folder hợp lệ
```

## 18. Hàm `has_class_subfolders`

Code:

```python
def has_class_subfolders(folder):
    folder = Path(folder)
    return any(p.is_dir() and count_images(p) > 0 for p in folder.iterdir())
```

Nó kiểm tra folder có thư mục con chứa ảnh không.

Ví dụ:

```text
train/no_entry/*.jpg
train/stop_sign/*.jpg
```

thì `train` có class subfolders.

## 19. Hàm `has_train_split`

Code:

```python
def has_train_split(folder):
    folder = Path(folder)
    lower_children = {p.name.lower(): p for p in folder.iterdir() if p.is_dir()}
    return 'train' in lower_children and has_class_subfolders(lower_children['train'])
```

Nó kiểm tra folder có:

```text
train/<class>/*.jpg
```

Nếu có, folder đó có thể là root dataset.

Ví dụ:

```text
SplitData/train/no_entry/*.jpg
```

thì `SplitData` là root dataset.

## 20. Vì sao ưu tiên `SplitData`

Trong `DataFinal`, có nhiều folder chứa ảnh:

```text
CropData
ResizeData
RGBData
SplitData
```

Nếu chỉ tìm folder nào có class subfolders, notebook có thể chọn nhầm `CropData` hoặc `RGBData`.

`SplitData` là dữ liệu cuối đã chia train/val/test, nên notebook ưu tiên:

```python
if cand.name.lower() in {'splitdata', 'split_data', 'split-data'} and has_train_split(cand):
    DATA_DIR = cand
```

Đây là thiết kế đúng vì train cần split rõ ràng.

## 21. Vì sao bỏ qua một số folder stage

Notebook có danh sách:

```python
ignored_stage_names = {'datafinal', 'code', 'metadata', 'filterdata', 'cropdata', 'resizedata', 'rgbdata'}
```

Khi fallback tìm folder class trực tiếp, nó tránh chọn các stage trung gian.

Ví dụ nếu không có `SplitData`, nhưng `RGBData` có:

```text
RGBData/no_entry/*.jpg
```

Notebook có thể chọn `RGBData` nếu không bỏ qua. Nhưng nếu đã có `SplitData`, phải chọn `SplitData`.

## 22. Khi nào notebook dùng fallback direct class folder?

Nếu dataset nén chỉ có:

```text
data/no_entry/*.jpg
data/stop_sign/*.jpg
```

không có train/val/test, notebook có thể chọn folder đó làm `DATA_DIR`.

Khi đó Cell 4 sẽ tự chia:

```text
70% train
15% val
15% test
```

Nhưng với data hiện tại, nên dùng `SplitData` đã chia sẵn.

## 23. Các lỗi thường gặp ở Cell 2

### 23.1. `Drive already mounted`

Thông báo:

```text
Drive already mounted at /content/drive
```

Không phải lỗi. Có thể bỏ qua.

### 23.2. `NameError: archive_path is not defined`

Lỗi này xảy ra nếu dùng:

```python
LOCAL_ARCHIVE_PATH = f'/content/{archive_path.name}'
```

trước khi khai báo:

```python
archive_path = Path(DRIVE_ARCHIVE_PATH)
```

Thứ tự đúng:

```python
archive_path = Path(DRIVE_ARCHIVE_PATH)
LOCAL_ARCHIVE_PATH = f'/content/{archive_path.name}'
```

### 23.3. `FileNotFoundError`

Nguyên nhân:

```text
đường dẫn Drive sai
tên file sai
file nằm trong folder khác
đuôi .rar/.zip sai
Drive chưa mount đúng tài khoản
```

Cách xử lý:

```text
kiểm tra file browser bên trái Colab
copy path đúng
sửa DRIVE_ARCHIVE_PATH
```

### 23.4. `unrar: command not found`

Nếu cài `unrar` thất bại, lệnh giải nén `.rar` sẽ lỗi.

Cách xử lý:

```text
chạy lại cell
đổi file sang .zip
kiểm tra Colab có cho apt-get không
```

### 23.5. Không tìm thấy `SplitData`

Nếu giải nén xong nhưng không thấy cấu trúc:

```text
SplitData/train/<class>/*.jpg
```

notebook có thể báo:

```text
Không tìm thấy cấu trúc dataset dạng folder class
```

Cách xử lý:

```text
xem output preview_tree
kiểm tra file nén có đúng dataset không
kiểm tra có nén lồng folder quá sâu không
kiểm tra tên folder train có viết đúng không
```

## 24. Vì sao không dùng CSV để tìm data trong Cell 2?

Cell 2 tìm folder ảnh, không tìm CSV, vì train bằng `ImageFolder`.

Điều kiện quan trọng là:

```text
folder class chứa ảnh
```

CSV chỉ là danh sách phụ trợ.

Nếu CSV tồn tại nhưng ảnh thiếu, model vẫn lỗi khi đọc ảnh. Nếu ảnh tồn tại nhưng CSV thiếu, `ImageFolder` vẫn train được.

## 25. Cần xóa folder giải nén cũ không?

Cell 2 hiện tạo:

```python
EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
```

và giải nén vào đó. Nếu chạy lại cell nhiều lần, file cũ có thể vẫn còn.

Với `.rar`, lệnh:

```text
unrar x -o+
```

ghi đè file trùng tên.

Tuy nhiên, nếu lần trước giải nén dataset A, lần sau giải nén dataset B vào cùng folder, có thể còn file thừa từ A nếu B không ghi đè hết.

Cách sạch hơn nếu cần:

```python
if EXTRACT_ROOT.exists():
    shutil.rmtree(EXTRACT_ROOT)
EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
```

Nhưng xóa folder lớn tốn thời gian. Chỉ cần làm khi nghi ngờ dữ liệu cũ lẫn dữ liệu mới.

## 26. Kiểm tra sau khi Cell 2 chạy xong

Sau Cell 2, cần nhìn các dòng:

```text
DATA_DIR được dùng cho training: ...
Tổng số ảnh tìm thấy: ...
```

Kỳ vọng:

```text
DATA_DIR ... SplitData
Tổng số ảnh khoảng 12252 nếu data gốc chưa augment
```

Nếu tổng số ảnh là 46540, có thể train đang chứa augment lưu file.

Nếu `DATA_DIR` trỏ vào:

```text
CropData
RGBData
Data
```

thì cần kiểm tra lại, vì notebook có thể đang không dùng split cuối.

## 27. Liên hệ với Cell 4

Cell 2 tạo:

```python
DATA_DIR
```

Cell 4 dùng:

```python
DATA_DIR = Path(DATA_DIR)
train_dir = find_split_dir(DATA_DIR, 'train')
val_dir = ...
test_dir = ...
```

Nếu Cell 2 chưa chạy, Cell 4 sẽ lỗi:

```text
NameError: DATA_DIR is not defined
```

Nếu Cell 2 chọn sai `DATA_DIR`, Cell 4 vẫn có thể chạy nhưng dataset sai.

## 28. Tại sao checkpoint không lưu ở `/content`

Dataset giải nén ở `/content` vì cần đọc nhanh.

Checkpoint lưu ở Drive vì cần bền:

```python
'checkpoint_dir': '/content/drive/MyDrive/mobilenetv2_gtsrb/checkpoints/'
```

Nếu checkpoint lưu ở `/content`, khi runtime bị xóa sẽ mất model đã train.

Thiết kế hợp lý:

```text
/content: data tạm để train nhanh
Drive: file nén gốc + checkpoint + logs
```

## 29. Có nên giải nén lại sau restart runtime không?

Nếu chỉ restart runtime, file trong `/content` đôi khi vẫn còn, nhưng biến Python mất. Tuy nhiên để chắc chắn, nên chạy lại Cell 2 nếu không chắc dataset còn tồn tại.

Nếu disconnect and delete runtime, chắc chắn phải chạy lại Cell 2.

Một cách kiểm tra nhanh:

```python
from pathlib import Path
Path('/content/data_bien_bao').exists()
```

Nếu `False`, cần giải nén lại.

## 30. Tối ưu thời gian khi chạy lại nhiều lần

Nếu dataset đã giải nén và không đổi, có thể thêm logic skip giải nén nếu `SplitData` đã tồn tại. Ví dụ:

```python
if (EXTRACT_ROOT / 'DataFinal' / 'SplitData').exists():
    print('Dataset already extracted, skip extraction')
else:
    # extract
```

Nhưng notebook hiện tại ưu tiên đơn giản và chắc chắn: chạy Cell 2 sẽ copy và giải nén lại.

## 31. Cách viết trong báo cáo

Có thể viết:

```text
Do Google Drive có tốc độ đọc nhiều file nhỏ tương đối chậm trong môi trường Colab, file dataset nén được lưu trên Drive nhưng sẽ được copy về ổ local `/content` của runtime trước khi giải nén. Sau khi giải nén, chương trình tự động tìm thư mục `SplitData` chứa ba tập train/validation/test và sử dụng thư mục này làm nguồn dữ liệu cho quá trình huấn luyện. Cách làm này giúp giảm nghẽn I/O trong quá trình train và đảm bảo checkpoint/log vẫn được lưu lâu dài trên Google Drive.
```

Nếu nói về `.rar`:

```text
Notebook hỗ trợ giải nén các định dạng `.zip`, `.tar`, `.tar.gz`, `.tgz` và `.rar`. Với file `.rar`, môi trường Colab cần cài thêm công cụ `unrar` trước khi giải nén.
```

## 32. Cách nói khi thuyết trình

Có thể nói:

```text
Em không train trực tiếp từ Google Drive vì đọc ảnh từ Drive chậm. Em lưu file nén trên Drive để không mất dữ liệu, sau đó copy file nén về `/content`, giải nén ở local Colab rồi train từ đó. Sau khi giải nén, notebook tự tìm thư mục `SplitData` để tránh chọn nhầm các folder trung gian như CropData hoặc RGBData.
```

Nếu thầy hỏi “vì sao checkpoint lại lưu Drive?”:

```text
Vì `/content` là bộ nhớ tạm của runtime, có thể mất khi Colab reset. Checkpoint cần lưu ở Drive để có thể resume hoặc đánh giá lại sau khi runtime bị ngắt.
```

## 33. Checklist Cell 2

Trước khi chạy train, kiểm tra:

```text
[ ] Drive đã mount
[ ] DRIVE_ARCHIVE_PATH đúng
[ ] File nén copy về /content thành công
[ ] Dung lượng file local hợp lý
[ ] Giải nén không lỗi
[ ] preview_tree thấy DataFinal hoặc SplitData
[ ] DATA_DIR trỏ đúng SplitData
[ ] Tổng số ảnh đúng kỳ vọng
[ ] Không chọn nhầm CropData/RGBData
```

## 34. Kết luận phần 3

Cell 2 không chỉ là “load data”. Nó quyết định dữ liệu thực sự đi vào toàn bộ pipeline.

Điểm quan trọng:

```text
Drive dùng để lưu lâu dài.
/content dùng để train nhanh.
File nén cần copy về local rồi giải nén.
Notebook phải tìm đúng SplitData.
DATA_DIR là biến nối Cell 2 với Cell 4.
```

Nếu Cell 2 đúng, các bước sau mới có nền dữ liệu đúng. Nếu Cell 2 sai, model có thể train nhầm folder mà người dùng không nhận ra ngay.
