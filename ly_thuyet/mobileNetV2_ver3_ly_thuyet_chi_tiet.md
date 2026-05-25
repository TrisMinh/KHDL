# Lý thuyết chi tiết cho notebook MobileNetV2 ver3

Tài liệu này giải thích từ đầu đến cuối notebook `version/mobilenetv2_gtsrb_ver3_datatulam.ipynb`: dữ liệu đi vào như thế nào, từng tham số trong `CONFIG` nghĩa là gì, vì sao cần crop, normalize, augment, MobileNetV2 hoạt động ra sao, quá trình train cập nhật trọng số thế nào, checkpoint dùng để làm gì, đọc log ra sao, và khi nào nghi ngờ overfit.

Notebook này đang giải bài toán **phân loại biển báo giao thông từ ảnh đã crop vùng biển báo**, không phải bài toán object detection trên ảnh đường phố nguyên cảnh.

## 1. Phạm vi bài toán

### 1.1. Classification khác detection thế nào

Notebook này làm **image classification**:

```text
ảnh đã chứa biển báo chính -> model -> tên lớp biển báo
```

Ví dụ input hợp lý:

```text
ảnh crop sát biển báo cấm vào -> no_entry
ảnh crop sát biển stop -> stop_sign
```

Notebook này không làm **object detection**:

```text
ảnh đường phố nguyên cảnh -> model tìm biển báo nằm ở đâu + phân loại
```

Vì vậy, nếu test ảnh ngoài là ảnh nguyên tấm có nhiều nền như đường, cây, xe, trời, nhà, thì cần crop vùng biển báo trước rồi mới đưa vào model. Cell dự đoán ảnh ngoài trong notebook cho phép kéo chuột crop chính là để phục vụ bước này.

### 1.2. Vì sao phải crop

MobileNetV2 trong notebook được train trên ảnh `SplitData` đã qua pipeline:

```text
ảnh gốc -> crop biển báo -> resize 224x224 -> RGB -> split train/val/test -> train classifier
```

Nếu lúc train model chỉ thấy ảnh crop sát biển báo, nhưng lúc test lại đưa ảnh nguyên cảnh, phân phối dữ liệu đã thay đổi. Model có thể nhìn quá nhiều background và dự đoán sai.

Nói chính xác:

```text
train distribution: ảnh crop biển báo
test distribution nếu không crop: ảnh nguyên cảnh
```

Hai phân phối khác nhau làm độ chính xác giảm. Đây không phải lỗi riêng của MobileNetV2 mà là sai phạm vi bài toán.

## 2. Cấu trúc dữ liệu

Theo tài liệu Word và folder hiện tại, dữ liệu có các bước:

```text
DataFinal/
  Data/        ảnh ban đầu
  FilterData/  ảnh bị loại
  CropData/    ảnh crop sát biển báo
  ResizeData/  ảnh resize
  RGBData/     ảnh RGB 3 kênh
  metaData/    CSV tọa độ crop của ảnh gốc
  SplitData/   dữ liệu cuối để train
```

`metaData/*.csv` chứa thông tin như:

```csv
image_name,folder,x1,x2,y1,y2
no_entry_0003.jpg,no_entry,226.9,448.41,24.08,273.72
```

Ý nghĩa:

- `image_name`: tên ảnh gốc.
- `folder`: lớp hoặc thư mục chứa ảnh.
- `x1, x2, y1, y2`: tọa độ bounding box để crop biển báo từ ảnh gốc.

Sau khi ảnh đã crop, resize, RGB và split xong, notebook dùng `SplitData`, không cần dùng tọa độ crop nữa.

### 2.1. Cấu trúc `SplitData`

Notebook ưu tiên tìm thư mục:

```text
SplitData/
  train/
    no_entry/
    no_stopping/
    ...
  val/
    no_entry/
    no_stopping/
    ...
  test/
    no_entry/
    no_stopping/
    ...
```

Đây là cấu trúc chuẩn cho `torchvision.datasets.ImageFolder`.

Với `ImageFolder`, label không lấy từ CSV mà lấy từ tên folder. Ví dụ:

```text
train/no_entry/abc.jpg -> class no_entry
train/stop_sign/xyz.jpg -> class stop_sign
```

Notebook vẫn có thể xem `train.csv`, `val.csv`, `test.csv`, nhưng khi train bằng `ImageFolder`, CSV không phải nguồn label chính.

### 2.2. Số lớp

Notebook không còn cố định 43 lớp như GTSRB. Cell 4 tự lấy:

```python
CLASS_NAMES = raw_train_for_classes.classes
CONFIG['num_classes'] = len(CLASS_NAMES)
NUM_CLASSES = CONFIG['num_classes']
```

Nếu dữ liệu có 12 folder class, model sẽ có 12 output. Nếu dữ liệu có 20 folder class, model sẽ có 20 output.

Điều này rất quan trọng vì tầng classifier cuối cùng phải có số neuron bằng số lớp:

```python
nn.Linear(last_channels, num_classes)
```

Nếu `num_classes` sai, model vẫn có thể chạy nhưng label/output sẽ lệch hoặc lỗi khi tính loss.

## 3. Load data từ Google Drive

Cell 2 làm các việc:

1. Mount Google Drive:

```python
drive.mount('/content/drive')
```

2. Lấy file nén từ Drive:

```python
DRIVE_ARCHIVE_PATH = '/content/drive/MyDrive/data_bien_bao.rar'
```

3. Copy file nén từ Drive về local Colab:

```python
LOCAL_ARCHIVE_PATH = f'/content/{archive_path.name}'
shutil.copy2(archive_path, LOCAL_ARCHIVE_PATH)
```

Lý do copy về `/content`: đọc file trực tiếp từ Drive chậm hơn nhiều so với local disk của runtime Colab. Với dataset lớn, train trực tiếp trên Drive dễ bị chậm và nghẽn I/O.

4. Giải nén:

Notebook hỗ trợ:

```text
.zip
.rar
.tar
.tar.gz
.tgz
```

Với `.rar`, Colab cần `unrar`:

```python
subprocess.run(['apt-get', '-qq', 'install', '-y', 'unrar'], check=True)
subprocess.run(['unrar', 'x', '-o+', LOCAL_ARCHIVE_PATH, str(EXTRACT_ROOT) + '/'], check=True)
```

5. Preview folder:

```python
preview_tree(EXTRACT_ROOT)
```

Mục đích là để nhìn nhanh sau giải nén có những thư mục gì, tránh train nhầm `Data`, `CropData`, `RGBData` thay vì `SplitData`.

6. Tự tìm `DATA_DIR`:

Notebook ưu tiên:

```text
SplitData/train/<class>/*.jpg
```

Nếu không có `SplitData`, nó tìm cấu trúc `train/val/test`, rồi mới đến cấu trúc folder class trực tiếp.

## 4. Cấu hình chính trong `CONFIG`

Cell 3 chứa cấu hình:

```python
CONFIG = {
    'img_size': 224,
    'resize_enabled': 0,
    'resize_size': 224,
    'augment_enabled': 0,
    'batch_size': 32,
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
    'checkpoint_dir': ...,
    'log_dir': ...,
}
```

### 4.1. `img_size`

Kích thước ảnh đầu vào mong muốn của model.

Hiện tại:

```python
'img_size': 224
```

Ảnh trong `SplitData` đã là `224x224`, nên cấu hình này hợp lý.

Ảnh lớn hơn làm model thấy nhiều chi tiết hơn nhưng tốn VRAM hơn. Ảnh nhỏ hơn train nhanh hơn nhưng có thể mất thông tin như chữ số trên biển tốc độ.

### 4.2. `resize_enabled`

```python
'resize_enabled': 0
```

Ý nghĩa:

```text
0: không resize trong notebook, giữ size ảnh đã chuẩn bị sẵn
1: resize trong notebook về resize_size
```

Vì dữ liệu đã resize trước về `224x224`, để `0` là hợp lý. Nếu dùng dataset mới chưa resize đều, bật `1` để tránh lỗi batch do ảnh khác size.

### 4.3. `resize_size`

```python
'resize_size': 224
```

Kích thước resize khi `resize_enabled = 1`, đồng thời được dùng làm `IMG_SIZE` trong transform và dummy input.

Nếu `resize_enabled = 0`, vẫn nên đặt `resize_size` bằng size ảnh thật để các bước như crop, model input, predict ảnh ngoài nhất quán.

### 4.4. `augment_enabled`

```python
'augment_enabled': 0
```

Ý nghĩa:

```text
0: không augment online
1: augment online khi train
```

Augment online nghĩa là ảnh được biến đổi ngẫu nhiên trong RAM khi DataLoader đọc ảnh, không lưu thêm file.

Nếu dữ liệu `SplitData/train` đã augment sẵn thành nhiều file, nên để `0` để tránh augment chồng augment. Nếu đã chạy lại split từ `RGBData` và train không còn ảnh augment sẵn, có thể bật `1`.

### 4.5. `batch_size`

```python
'batch_size': 32
```

Batch size là số ảnh đưa vào model trong một lần cập nhật.

Với `train = 8578` ảnh và `batch_size = 32`:

```text
số batch mỗi epoch ≈ 8578 / 32 ≈ 269 batch
```

Ảnh hưởng:

- Batch lớn: train nhanh hơn theo epoch, gradient ổn định hơn, nhưng tốn VRAM.
- Batch nhỏ: đỡ OOM, gradient nhiễu hơn, có thể generalize tốt hơn, nhưng train lâu hơn.

Nếu Colab báo CUDA out of memory:

```python
'batch_size': 16
```

hoặc:

```python
'batch_size': 8
```

### 4.6. `epochs`

```python
'epochs': 50
```

Một epoch nghĩa là model đi qua toàn bộ train set một lần. Nếu train có 8578 ảnh, batch 32, một epoch có khoảng 269 lần cập nhật.

Không phải epoch 1 là model mới xem vài ảnh. Kết thúc epoch 1 nghĩa là model đã nhìn toàn bộ train set một vòng.

Nếu val accuracy đã rất cao sau vài epoch, không nhất thiết train đủ 50 epoch. Có thể chỉnh:

```python
'epochs': 15
'patience': 5
```

### 4.7. `lr`

```python
'lr': 0.01
```

Learning rate là độ lớn bước cập nhật trọng số.

Nếu learning rate quá cao:

```text
loss dao động mạnh, có thể không hội tụ
```

Nếu learning rate quá thấp:

```text
train rất chậm, dễ kẹt ở kết quả chưa tốt
```

Notebook dùng warmup + cosine scheduler nên learning rate không cố định 0.01 suốt training.

### 4.8. `momentum`

```python
'momentum': 0.9
```

Momentum giúp SGD có quán tính. Thay vì cập nhật chỉ dựa vào gradient hiện tại, optimizer còn nhớ hướng cập nhật trước đó.

Ý tưởng:

```text
nếu nhiều batch liên tiếp đều chỉ cùng một hướng giảm loss, momentum giúp đi nhanh hơn
nếu gradient nhiễu, momentum giúp đường đi mượt hơn
```

### 4.9. `weight_decay`

```python
'weight_decay': 1e-4
```

Weight decay là L2 regularization. Nó phạt trọng số quá lớn.

Loss hiệu dụng có dạng:

```text
loss_total = loss_classification + λ * ||W||²
```

Trong đó `λ = weight_decay`.

Tác dụng:

- Giảm overfit.
- Làm model ít phụ thuộc vào vài trọng số cực lớn.
- Giúp nghiệm mượt hơn.

Nếu quá lớn, model có thể underfit vì trọng số bị ép quá mạnh.

### 4.10. `width_mult`

```python
'width_mult': 1.0
```

`width_mult` điều chỉnh số channel trong MobileNetV2.

Ví dụ:

```text
width_mult = 1.0: số channel chuẩn
width_mult = 0.75: model nhỏ hơn, ít tham số hơn
width_mult = 1.4: model lớn hơn, mạnh hơn nhưng tốn VRAM hơn
```

Trong code, số channel được nhân với `width_mult` rồi đưa qua `_make_divisible` để chia hết cho 8.

### 4.11. `warmup_epochs`

```python
'warmup_epochs': 5
```

Warmup là giai đoạn tăng learning rate từ thấp lên `lr` chính trong vài epoch đầu.

Với `lr = 0.01`, `warmup_epochs = 5`, learning rate gần như:

```text
epoch 1: 0.002
epoch 2: 0.004
epoch 3: 0.006
epoch 4: 0.008
epoch 5: 0.010
```

Tác dụng:

- Tránh cập nhật quá mạnh khi model mới khởi tạo ngẫu nhiên.
- Giúp loss ổn định hơn.
- Hữu ích khi train từ đầu.

Nếu model học quá nhanh, có thể giảm:

```python
'warmup_epochs': 2
```

### 4.12. `label_smoothing`

```python
'label_smoothing': 0.1
```

Cross entropy thường dùng nhãn one-hot:

```text
class đúng: 1.0
class sai: 0.0
```

Label smoothing làm nhãn bớt tuyệt đối:

```text
class đúng: khoảng 0.9
class sai: chia một phần nhỏ cho các class còn lại
```

Tác dụng:

- Giảm overconfidence.
- Giúp model không quá chắc chắn 100%.
- Có thể cải thiện generalization.

Nếu label smoothing quá lớn, model có thể khó đạt confidence cao và underfit.

### 4.13. `dropout`

```python
'dropout': 0.2
```

Dropout tắt ngẫu nhiên một phần neuron trong classifier khi train.

Trong model:

```python
self.classifier = nn.Sequential(
    nn.Dropout(p=dropout),
    nn.Linear(last_channels, num_classes),
)
```

Tác dụng:

- Giảm phụ thuộc vào một vài feature.
- Giảm overfit.

Khi evaluate hoặc predict, dropout tự tắt vì model gọi `model.eval()`.

### 4.14. `grad_clip`

```python
'grad_clip': 5.0
```

Gradient clipping giới hạn độ lớn gradient:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
```

Nếu gradient quá lớn, cập nhật weight có thể nhảy quá mạnh làm loss bất ổn. Clipping giúp training an toàn hơn.

### 4.15. `patience`

```python
'patience': 10
```

Early stopping: nếu `val_acc` không cải thiện trong `patience` epoch liên tiếp, dừng train.

Tác dụng:

- Tránh train quá lâu.
- Giảm nguy cơ overfit.
- Tiết kiệm GPU.

### 4.16. `save_every`

```python
'save_every': 10
```

Cứ mỗi 10 epoch lưu checkpoint định kỳ:

```text
checkpoint_epoch_10.pth
checkpoint_epoch_20.pth
...
```

Ngoài ra notebook còn lưu:

```text
best_model.pth
checkpoint_latest.pth
```

### 4.17. `seed`

```python
'seed': 42
```

Seed giúp kết quả tái lập hơn:

```python
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
random.seed(CONFIG['seed'])
```

Tuy vậy, trên GPU vẫn có một số phép toán có thể không hoàn toàn deterministic tùy môi trường.

### 4.18. `resume`

```python
'resume': False
```

Nếu `False`, notebook xóa checkpoint/log cũ và train mới.

Nếu `True`, notebook cố load:

```text
checkpoint_latest.pth
```

để train tiếp.

Cẩn thận: nếu đang muốn test model đã train, không cần train lại, chỉ cần chạy cell tạo model rồi cell evaluation load `best_model.pth`.

## 5. Transform, ToTensor, Normalize

### 5.1. `ToTensor`

Ảnh gốc thường có pixel:

```text
0 -> 255
```

`transforms.ToTensor()` đổi về tensor:

```text
0.0 -> 1.0
```

Đồng thời đổi shape:

```text
PIL image: H x W x C
PyTorch tensor: C x H x W
```

Với ảnh RGB:

```text
3 x 224 x 224
```

### 5.2. Normalize là loại gì

Notebook dùng:

```python
transforms.Normalize(MEAN, STD)
```

Đây là **Z-score normalization theo từng kênh RGB**, không phải min-max.

Công thức:

```text
x_norm = (x - mean) / std
```

Trong đó `x` đã nằm trong khoảng `0 -> 1` sau `ToTensor`.

Notebook tính `MEAN` và `STD` từ tập train:

```python
MEAN = (channel_sum / pixel_count).tolist()
STD = torch.sqrt(channel_sq_sum / pixel_count - torch.tensor(MEAN) ** 2).tolist()
```

Không dùng val/test để tính mean/std vì val/test phải đóng vai trò dữ liệu chưa biết. Dùng val/test để tính thống kê có thể làm rò rỉ thông tin nhẹ.

### 5.3. Vì sao normalize giúp train

Nếu input có phân phối lệch, các layer đầu có thể nhận giá trị quá lớn/quá nhỏ. Normalize đưa input về phân phối ổn định hơn, giúp:

- Gradient ổn định.
- Loss hội tụ nhanh hơn.
- Model ít nhạy với độ sáng/màu tổng thể.

### 5.4. Inverse normalize để hiển thị

Ảnh sau Normalize không còn nằm trong khoảng màu tự nhiên. Muốn hiển thị bằng Matplotlib, notebook dùng inverse normalize:

```python
transforms.Normalize(mean=[-m/s for m, s in zip(MEAN, STD)],
                     std=[1/s for s in STD])
```

Đây là phép ngược của:

```text
x_norm = (x - mean) / std
```

## 6. Augmentation

### 6.1. Augment là gì

Augmentation là tạo biến thể của ảnh train để model học robust hơn:

- Xoay nhẹ.
- Dịch chuyển.
- Thay đổi sáng/tương phản.
- Làm mờ nhẹ.
- Che một vùng nhỏ.
- Biến dạng phối cảnh.

Mục tiêu là giúp model không chỉ nhớ ảnh train y nguyên mà học đặc trưng bền hơn.

### 6.2. Augment online trong notebook

Nếu:

```python
'augment_enabled': 1
```

Cell 4 dùng:

```python
RandomCrop
RandomRotation
RandomAffine
RandomPerspective
ColorJitter
RandomGrayscale
GaussianBlur
RandomErasing
```

Các biến đổi này chỉ áp dụng cho train:

```text
train: augment + ToTensor + Normalize
val/test: ToTensor + Normalize
```

Không augment val/test vì val/test phải đại diện cho dữ liệu đánh giá thật, không bị biến đổi ngẫu nhiên mỗi lần.

### 6.3. Online augment khác augment lưu file

Script `augment_images.py` tạo file mới:

```text
ảnh gốc -> ảnh_0001.jpg, ảnh_0002.jpg, ...
```

Notebook v3 dùng online augment:

```text
mỗi epoch đọc ảnh -> biến đổi ngẫu nhiên trong RAM -> đưa vào model
```

Không lưu file mới. Nếu train 20 epoch, một ảnh có thể được nhìn 20 lần với 20 biến thể ngẫu nhiên khác nhau.

### 6.4. Khi nào bật augment

Nếu `SplitData/train` đã augment sẵn thành nhiều file, nên:

```python
'augment_enabled': 0
```

Nếu vừa split lại từ `RGBData` và train chỉ còn ảnh gốc, có thể:

```python
'augment_enabled': 1
```

### 6.5. Augment quá mạnh có hại không

Có. Nếu ảnh bị xoay/crop/che quá mạnh làm mất ký hiệu chính, label không còn rõ. Ví dụ biển tốc độ mà mất số ở giữa thì người cũng khó phân loại.

Nguyên tắc:

```text
biến đổi phải giống tình huống thực tế nhưng vẫn giữ biển báo nhận ra được
```

## 7. DataLoader và batch

`DataLoader` chịu trách nhiệm lấy ảnh từ dataset thành từng batch:

```python
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                          shuffle=True, num_workers=2, pin_memory=True)
```

### 7.1. `shuffle=True`

Train cần shuffle để mỗi epoch ảnh đi vào model theo thứ tự khác nhau. Nếu không shuffle, model có thể học theo thứ tự class hoặc thứ tự file.

Val/test không shuffle vì không cần cập nhật weight, chỉ đánh giá.

### 7.2. `num_workers=2`

Số process phụ để đọc ảnh và transform. Tăng `num_workers` có thể tăng tốc data loading, nhưng quá cao có thể tốn RAM hoặc lỗi trong Colab.

### 7.3. `pin_memory=True`

Giúp copy tensor từ CPU sang GPU nhanh hơn khi dùng CUDA.

## 8. MobileNetV2 từ đầu

### 8.1. CNN là gì

MobileNetV2 là một CNN. CNN học đặc trưng ảnh bằng convolution:

```text
cạnh/đường nét -> hình tròn/tam giác -> ký hiệu -> lớp biển báo
```

Convolution dùng kernel nhỏ trượt trên ảnh để học pattern cục bộ.

### 8.2. Vì sao MobileNetV2 nhẹ

MobileNetV2 dùng **depthwise separable convolution** thay vì convolution thường.

Convolution thường với input `H x W x Cin`, output `Cout`, kernel `K x K` có chi phí xấp xỉ:

```text
H * W * Cin * Cout * K * K
```

Depthwise separable convolution tách thành:

1. Depthwise convolution:

```text
H * W * Cin * K * K
```

2. Pointwise convolution `1x1`:

```text
H * W * Cin * Cout
```

Tổng:

```text
H * W * Cin * K * K + H * W * Cin * Cout
```

So với convolution thường, chi phí giảm rất nhiều khi `Cout` lớn.

### 8.3. `ConvBNReLU6`

Block cơ bản:

```python
nn.Conv2d(...)
nn.BatchNorm2d(...)
nn.ReLU6(...)
```

Ý nghĩa:

- `Conv2d`: học filter trích đặc trưng.
- `BatchNorm2d`: chuẩn hóa activation theo batch, giúp train ổn định.
- `ReLU6`: activation giới hạn trong khoảng 0 đến 6, thường dùng trong MobileNet.

### 8.4. ReLU6 là gì

ReLU thường:

```text
ReLU(x) = max(0, x)
```

ReLU6:

```text
ReLU6(x) = min(max(0, x), 6)
```

Giới hạn trên 6 giúp activation không quá lớn, hữu ích cho model nhẹ và triển khai trên thiết bị hạn chế.

### 8.5. Inverted residual block

MobileNetV2 dùng block:

```text
input hẹp -> expand rộng -> depthwise conv -> project hẹp
```

Trong code:

1. Expansion:

```python
ConvBNReLU6(in_channels, hidden_dim, kernel_size=1)
```

2. Depthwise:

```python
ConvBNReLU6(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim)
```

3. Projection:

```python
nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False)
nn.BatchNorm2d(out_channels)
```

Projection không dùng ReLU. Đây gọi là **linear bottleneck**.

### 8.6. Vì sao projection không ReLU

Ở bottleneck ít channel, nếu dùng ReLU có thể làm mất thông tin vì ReLU cắt toàn bộ giá trị âm về 0. MobileNetV2 giữ projection tuyến tính để bảo toàn thông tin tốt hơn.

### 8.7. Skip connection

Code:

```python
self.use_skip = (stride == 1 and in_channels == out_channels)
```

Nếu input và output cùng shape, block trả:

```python
x + self.conv(x)
```

Skip connection giúp gradient đi qua mạng sâu dễ hơn. Nó cũng cho block học phần chênh lệch so với input thay vì học toàn bộ mapping từ đầu.

### 8.8. `stride`

`stride=1` giữ kích thước feature map.

`stride=2` giảm kích thước feature map khoảng một nửa.

Với input 224x224, first conv dùng:

```python
first_stride = 2 if IMG_SIZE >= 160 else 1
```

Lý do: nếu giữ stride 1 ở ảnh 224x224, feature map lớn, tốn VRAM, dễ CUDA OOM. Dùng stride 2 giống thiết kế MobileNetV2 gốc hơn và tiết kiệm memory.

### 8.9. `_make_divisible`

MobileNet thường yêu cầu số channel chia hết cho 8 để tối ưu phần cứng:

```python
_make_divisible(v, divisor=8)
```

Nếu `width_mult` làm channel thành số lẻ, hàm này làm tròn về bội số của 8.

### 8.10. Classifier

Sau feature extractor:

```python
x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
x = torch.flatten(x, 1)
x = self.classifier(x)
```

Adaptive average pooling biến feature map thành vector mỗi channel một giá trị trung bình. Sau đó `Linear` đưa về số lớp.

## 9. Khởi tạo trọng số

Notebook tự train từ đầu, không dùng pretrained. Trọng số được khởi tạo:

```python
nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
```

cho convolution.

BatchNorm:

```python
weight = 1
bias = 0
```

Linear:

```python
normal mean=0, std=0.01
```

Không có dòng:

```python
pretrained=True
weights=...
load_state_dict(...)
```

trước train. Vì vậy model ban đầu là random initialization.

## 10. Loss function

Notebook dùng:

```python
criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
```

### 10.1. Cross entropy

Model output logits:

```text
z = [z1, z2, ..., zC]
```

Softmax đổi logits thành xác suất:

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

Cross entropy cho class đúng `y`:

```text
loss = -log(p_y)
```

Nếu model tự tin đúng, `p_y` gần 1:

```text
loss gần 0
```

Nếu model tự tin sai, `p_y` rất nhỏ:

```text
loss lớn
```

### 10.2. Vì sao accuracy cao nhưng loss vẫn không bằng 0

Accuracy chỉ xét đúng/sai:

```text
predicted_class == true_class
```

Loss xét cả độ tự tin. Nếu dự đoán đúng nhưng confidence chưa cao, loss vẫn còn.

Label smoothing cũng làm loss khó về 0 tuyệt đối vì target không còn là one-hot 100%.

## 11. Optimizer SGD Nesterov

Notebook dùng:

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=CONFIG['lr'],
    momentum=CONFIG['momentum'],
    weight_decay=CONFIG['weight_decay'],
    nesterov=True
)
```

SGD cập nhật weight theo hướng ngược gradient:

```text
W_new = W_old - lr * gradient
```

Momentum thêm vận tốc:

```text
v = momentum * v + gradient
W = W - lr * v
```

Nesterov momentum nhìn trước một bước, thường giúp cập nhật chính xác hơn so với momentum thường.

## 12. Learning rate scheduler

Notebook dùng `WarmupCosineScheduler`.

### 12.1. Warmup

Trong `warmup_epochs`, learning rate tăng tuyến tính:

```python
lr = base_lr * (current_epoch / warmup_epochs)
```

Nếu `base_lr = 0.01`, `warmup_epochs = 5`:

```text
epoch 1: 0.002
epoch 2: 0.004
epoch 3: 0.006
epoch 4: 0.008
epoch 5: 0.010
```

### 12.2. Cosine annealing

Sau warmup:

```python
lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
```

Learning rate giảm mượt từ `base_lr` về `min_lr`.

Tác dụng:

- Đầu training học nhanh.
- Cuối training bước nhỏ hơn để tinh chỉnh.

## 13. Mixed precision training

Notebook dùng:

```python
from torch.cuda.amp import GradScaler, autocast
```

Trong train:

```python
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
```

Mixed precision dùng FP16 cho một số phép toán để:

- Giảm VRAM.
- Tăng tốc trên GPU hỗ trợ Tensor Cores.

`GradScaler` giúp tránh underflow gradient khi dùng FP16.

## 14. Training loop

Một batch train:

```python
optimizer.zero_grad()
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(...)
scaler.step(optimizer)
scaler.update()
```

Giải thích:

1. `zero_grad`: xóa gradient cũ.
2. Forward: model dự đoán.
3. Loss: đo sai số.
4. Backward: tính gradient.
5. Clip gradient: giới hạn gradient.
6. Optimizer step: cập nhật weight.
7. Scaler update: cập nhật scale cho mixed precision.

### 14.1. Một epoch là gì

Một epoch là model đi qua toàn bộ train set một lần.

Nếu:

```text
train = 8578 ảnh
batch_size = 32
```

thì:

```text
8578 / 32 ≈ 269 batch
```

Kết thúc epoch 1 nghĩa là model đã cập nhật khoảng 269 lần.

Vì vậy epoch 1 accuracy cao không nhất thiết bất thường, nhất là với ảnh crop sạch và 12 lớp.

## 15. Validation

Validation không cập nhật weight:

```python
@torch.no_grad()
def validate(...)
```

Model được chuyển sang:

```python
model.eval()
```

Điều này tắt dropout và dùng BatchNorm ở chế độ đánh giá.

Validation giúp kiểm tra model có học tổng quát không, thay vì chỉ nhớ train.

## 16. Checkpoint

Notebook lưu:

```text
best_model.pth
checkpoint_latest.pth
checkpoint_epoch_N.pth
```

### 16.1. `best_model.pth`

Lưu khi `val_acc` tốt nhất từ trước đến nay.

Đây thường là file nên dùng để test và export.

### 16.2. `checkpoint_latest.pth`

Lưu sau mỗi epoch để resume nếu bị ngắt.

### 16.3. `checkpoint_epoch_N.pth`

Lưu định kỳ theo `save_every`.

### 16.4. Resume

Nếu:

```python
'resume': True
```

notebook load `checkpoint_latest.pth` và train tiếp.

Nếu:

```python
'resume': False
```

notebook xóa checkpoint/log cũ và train mới.

## 17. Đọc log training

Log có dạng:

```text
Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR | Time | Status
```

Ví dụ:

```text
1 | 1.0613 | 78.44% | 0.6155 | 98.97% | 0.002000 | 273.5s | BEST
2 | 0.6061 | 98.56% | 0.5616 | 99.73% | 0.004000 | 248.6s | BEST
```

### 17.1. Train loss

Loss trung bình trên train set. Xu hướng mong muốn là giảm.

Nhưng loss không bắt buộc giảm đều từng epoch. Nó có thể dao động do:

- Shuffle batch.
- Augment ngẫu nhiên.
- Learning rate.
- Batch size.

### 17.2. Train accuracy

Tỷ lệ ảnh train dự đoán đúng.

Nếu train accuracy tăng, model đang học train set tốt hơn.

### 17.3. Val loss

Loss trên validation set. Đây là chỉ số quan trọng để phát hiện overfit.

Nếu train loss giảm nhưng val loss tăng liên tục, cần nghi ngờ overfit.

### 17.4. Val accuracy

Tỷ lệ val dự đoán đúng.

Val accuracy cao không tự động là overfit. Overfit là khi train tốt nhưng val/test hoặc ảnh ngoài kém.

### 17.5. LR

Learning rate hiện tại. Với warmup, LR tăng vài epoch đầu, sau đó giảm theo cosine.

### 17.6. Time

Thời gian một epoch. Ảnh lớn, batch lớn, model lớn, augment nhiều đều làm epoch lâu hơn.

## 18. Vì sao epoch đầu có thể cao

Trong bài này, epoch 1 có thể đạt cao vì:

- Chỉ có 12 lớp.
- Ảnh đã crop sát biển báo.
- Ảnh đã resize/RGB chuẩn.
- Biển báo có màu/hình rất đặc trưng.
- Một epoch đã đi qua toàn bộ train set.
- Nếu train có augment hoặc data sạch, model học pattern rất nhanh.

Điều này không chứng minh có pretrained.

Notebook custom MobileNetV2 không dùng pretrained nếu không có dòng load weight pretrained.

## 19. Overfit là gì

Overfit là khi model học quá kỹ train set nhưng không tổng quát tốt.

Dấu hiệu:

```text
train acc tăng cao
train loss giảm
val acc giảm hoặc đứng yên
val loss tăng nhiều epoch liên tục
test/ảnh ngoài kém
```

Không nên kết luận overfit chỉ vì:

```text
train acc cao
val acc cao
test acc cao
```

Nếu test nội bộ cao nhưng ảnh ngoài kém, vấn đề có thể là:

```text
dataset quá sạch hoặc khác ảnh thực tế
```

Đây là distribution shift, liên quan overfitting theo style dữ liệu.

## 20. Evaluation metrics

### 20.1. Accuracy

```text
accuracy = số dự đoán đúng / tổng số mẫu
```

Dễ hiểu nhưng có thể đánh lừa nếu class imbalance mạnh.

### 20.2. Precision

Với một class:

```text
precision = TP / (TP + FP)
```

Trong số mẫu model dự đoán là class đó, bao nhiêu là đúng.

### 20.3. Recall

```text
recall = TP / (TP + FN)
```

Trong số mẫu thật thuộc class đó, model tìm đúng bao nhiêu.

### 20.4. F1-score

```text
F1 = 2 * precision * recall / (precision + recall)
```

Cân bằng giữa precision và recall.

### 20.5. Weighted average

Notebook dùng:

```python
average='weighted'
```

Nghĩa là metric trung bình theo số mẫu mỗi class. Class nhiều ảnh ảnh hưởng nhiều hơn class ít ảnh.

## 21. Confusion matrix

Confusion matrix cho biết class nào bị nhầm với class nào.

Trục y:

```text
label thật
```

Trục x:

```text
label dự đoán
```

Nếu tất cả nằm trên đường chéo, model dự đoán đúng.

Nếu class `speed_limit_50` hay bị nhầm sang `speed_limit_60`, ô tương ứng sẽ sáng ngoài đường chéo.

## 22. Per-class accuracy

Notebook tính:

```python
mask = all_labels == i
acc = (all_preds[mask] == i).sum() / mask.sum() * 100
```

Chỉ số này trả lời:

```text
Trong class i, model đúng bao nhiêu phần trăm?
```

Quan trọng vì overall accuracy cao có thể che giấu class yếu.

## 23. Predict ảnh ngoài

Cell predict ảnh ngoài:

1. Upload ảnh.
2. Kéo chuột crop vùng biển báo.
3. Resize về `IMG_SIZE`.
4. ToTensor.
5. Normalize bằng `MEAN`, `STD` train.
6. Model dự đoán top-5.

Nếu ảnh ngoài đã crop sát biển báo, có thể bấm OK dùng toàn bộ ảnh. Nếu ảnh là ảnh nguyên cảnh, nên kéo crop.

## 24. CUDA OOM và cách xử lý

OOM nghĩa là GPU hết VRAM.

Nguyên nhân thường gặp:

- Batch size quá lớn.
- Ảnh quá lớn.
- Feature map quá lớn do stride nhỏ.
- Runtime vừa bị OOM và GPU memory chưa được giải phóng.

Notebook đã chỉnh:

```python
'batch_size': 32
first_stride = 2 if IMG_SIZE >= 160 else 1
```

Nếu vẫn OOM:

```python
'batch_size': 16
```

Nếu OOM sau lần chạy lỗi, nên:

```text
Runtime > Restart runtime
```

Vì có khi batch 1 cũng OOM do GPU memory bị giữ lại từ lần lỗi trước.

## 25. Các tình huống thường gặp

### 25.1. Val accuracy cao hơn train accuracy

Có thể bình thường nếu train khó hơn val:

- Train có augment.
- Train có dropout.
- Val không augment.
- Val sạch hơn.

### 25.2. Accuracy lên gần 100%

Không tự động là overfit.

Với bài classification ảnh crop 12 lớp, accuracy rất cao có thể hợp lý. Cần xem:

- Test set.
- Confusion matrix.
- Ảnh ngoài tự crop.
- Val loss có tăng không.

### 25.3. Loss không giảm đều

Bình thường nếu chỉ dao động nhẹ. Đáng lo khi val loss tăng liên tục trong khi train loss giảm.

### 25.4. Augment preview giống ảnh gốc

Nếu:

```python
'augment_enabled': 0
```

thì `augment_preview_transform` chỉ gần như `ToTensor`, nên ảnh Augment 1..7 giống ảnh gốc là đúng.

## 26. Có cần attention hoặc transformer không

Notebook hiện tại là MobileNetV2 CNN, chưa dùng attention hay transformer.

Phân biệt:

```text
Attention = cơ chế/kỹ thuật giúp model chú ý phần quan trọng
Transformer = kiến trúc model dùng attention làm lõi
```

Nếu yêu cầu là “áp dụng attention để cải thiện MobileNetV2”, có thể thêm SE block hoặc CBAM vào CNN.

Nếu yêu cầu là “dùng Transformer”, cần dùng Vision Transformer như ViT/Swin/DeiT, đó là model khác với MobileNetV2.

## 27. Cách trình bày trong báo cáo

Có thể viết:

```text
Đề tài tập trung vào bài toán phân loại biển báo giao thông từ ảnh đã crop vùng biển báo. Dữ liệu được tiền xử lý qua các bước crop, resize về 224x224, chuyển RGB, sau đó chia thành train/validation/test theo tỷ lệ 70/15/15. Mô hình sử dụng MobileNetV2 được xây dựng từ đầu bằng PyTorch, không dùng pretrained weights. Input được chuẩn hóa bằng Z-score normalization theo mean/std tính từ tập train. Quá trình huấn luyện sử dụng CrossEntropyLoss với label smoothing, SGD Nesterov, warmup + cosine learning rate schedule, mixed precision, gradient clipping, checkpoint và early stopping.
```

Nếu cần nói về ảnh ngoài:

```text
Vì mô hình là classifier, ảnh ngoài cần được crop vùng biển báo trước khi đưa vào dự đoán. Việc phát hiện vị trí biển báo trên ảnh nguyên cảnh thuộc bài toán object detection và nằm ngoài phạm vi mô hình hiện tại.
```

## 28. Tự phản biện và kiểm tra chất lượng

### 28.1. Có rò rỉ dữ liệu không?

Cần đảm bảo split được thực hiện trước khi augment file cố định. Nếu augment trước rồi mới split, ảnh gốc và biến thể của nó có thể rơi vào cả train và val/test, làm điểm ảo.

Hiện tại bạn đã chạy lại split từ `RGBData`, nên `SplitData` trở về ảnh gốc 70/15/15. Nếu bật augment online trong notebook, augment chỉ áp dụng train trong RAM nên không rò rỉ sang val/test.

### 28.2. Có dùng pretrained không?

Không, nếu notebook không có dòng:

```python
weights=...
pretrained=True
load_state_dict(pretrained...)
```

Model được khởi tạo từ đầu bằng class `MobileNetV2`.

### 28.3. Val/test quá cao có đáng nghi không?

Cao không tự động sai. Nhưng cần kiểm tra:

- Test set có cùng nguồn với train không?
- Có ảnh trùng không?
- Ảnh ngoài tự crop có đúng không?
- Confusion matrix có class nào yếu không?
- Val loss có tăng không?

Nếu test nội bộ rất cao nhưng ảnh ngoài sai, phải viết rõ model tốt trên dữ liệu crop sạch cùng pipeline, chưa chứng minh tốt trên ảnh thực tế chưa crop.

### 28.4. Normalize có đúng không?

Đúng nếu:

- Tính mean/std từ train.
- Dùng cùng mean/std cho train, val, test và ảnh ngoài.
- Không tính từ val/test.

Notebook đang làm đúng hướng này.

### 28.5. Augment có đúng không?

Đúng nếu chỉ áp dụng train. Val/test không augment.

Nếu dữ liệu train đã augment sẵn, nên tắt online augment. Nếu split lại từ ảnh gốc, có thể bật online augment.

### 28.6. Crop có làm bài toán dễ quá không?

Crop là đúng với bài classification. Nhưng phải nói rõ phạm vi:

```text
Mô hình phân loại ảnh biển báo đã crop, không phát hiện biển báo trong ảnh nguyên cảnh.
```

Nếu muốn xử lý ảnh nguyên cảnh tự động, cần thêm detector trước classifier.

## 29. Checklist trước khi train

1. File nén đã upload lên Drive.
2. `DRIVE_ARCHIVE_PATH` đúng.
3. Cell 2 preview thấy `SplitData/train`, `SplitData/val`, `SplitData/test`.
4. `img_size = 224`, `resize_enabled = 0` nếu ảnh đã 224x224.
5. `augment_enabled = 1` nếu train chưa augment sẵn; `0` nếu train đã augment sẵn.
6. `batch_size = 32`; nếu OOM giảm 16.
7. Chạy Cell 4 thấy `Classes` đúng.
8. Chạy Cell 5 xem ảnh sample đúng class.
9. Chạy train và theo dõi `val_loss`, `val_acc`.
10. Sau train chạy test, confusion matrix, ảnh ngoài tự crop.

## 30. Checklist sau khi train

1. Xem `best_model.pth` có lưu.
2. Xem train/val curves.
3. Test accuracy và F1.
4. Confusion matrix.
5. Per-class accuracy.
6. Sample predictions đúng/sai.
7. Test ảnh ngoài.
8. Nếu ảnh ngoài kém, bổ sung dữ liệu/crop thực tế hoặc thêm detector.

## 31. Kết luận ngắn

Notebook ver3 là pipeline classification tương đối đầy đủ:

```text
Drive archive -> local extract -> SplitData -> ImageFolder -> train mean/std -> transform -> MobileNetV2 custom -> SGD training -> checkpoint -> test metrics -> visualization -> external crop prediction
```

Để hiểu hết notebook cần nắm:

- Phân biệt classification và detection.
- Cấu trúc folder `ImageFolder`.
- ToTensor và Z-score Normalize.
- Augmentation train-only.
- MobileNetV2: depthwise separable conv, inverted residual, linear bottleneck, skip connection.
- Cross entropy, label smoothing.
- SGD, momentum, weight decay.
- Warmup cosine scheduler.
- Mixed precision, gradient clipping.
- Checkpoint, resume, early stopping.
- Accuracy, precision, recall, F1, confusion matrix.
- Overfit và distribution shift.

Nếu các điểm trên được trình bày rõ trong báo cáo, người đọc sẽ hiểu vì sao model có thể đạt kết quả cao trên ảnh crop sạch, và cũng hiểu giới hạn của mô hình khi gặp ảnh nguyên cảnh chưa crop.

## 32. Bản đồ từng cell trong notebook

Phần này tóm tắt notebook theo thứ tự chạy để biết mỗi cell phụ trách việc gì và cell sau phụ thuộc biến nào từ cell trước.

### Cell 0: tiêu đề notebook

Cell markdown giới thiệu:

```text
MobileNetV2 - phân loại biển báo giao thông từ data tự làm
input 224x224
custom MobileNetV2
checkpoint, resume, visualization
```

Cell này không tạo biến.

### Cell 1: setup

Tạo các import nền:

```python
os, sys, time, json, csv, random, math, warnings
Path
numpy
PIL.Image
drive.mount
```

Biến quan trọng:

```text
drive đã mount
numpy alias np
Image từ PIL
```

Nếu Cell 1 chưa chạy, các cell sau có thể lỗi vì thiếu import.

### Cell 2: load data từ Drive và giải nén

Tạo các biến:

```python
DRIVE_ARCHIVE_PATH
archive_path
LOCAL_ARCHIVE_PATH
EXTRACT_ROOT
DATA_DIR
IMAGE_EXTS
```

Vai trò quan trọng nhất:

```text
DATA_DIR = thư mục sẽ đưa vào Cell 4 để tạo dataset
```

Cell này cũng tự tìm `SplitData`. Nếu `DATA_DIR` chọn sai, toàn bộ train sẽ sai.

### Cell 3: config và device

Tạo:

```python
CONFIG
device
```

Cell này quyết định hầu hết hành vi train:

- Kích thước ảnh.
- Batch size.
- Learning rate.
- Số epoch.
- Có augment online hay không.
- Checkpoint/log lưu ở đâu.
- Có resume không.

### Cell 4: dataset, normalize, augment, dataloader

Tạo các biến rất quan trọng:

```python
IMG_SIZE
RESIZE_ENABLED
train_dir, val_dir, test_dir
TRAIN_SOURCE_DIR
CLASS_NAMES
NUM_CLASSES
MEAN, STD
train_transform
val_transform
train_dataset, val_dataset, test_dataset
train_loader, val_loader, test_loader
```

Cell này là cầu nối giữa dữ liệu và model. Nếu Cell 4 sai, model có thể train nhầm class, nhầm normalize, hoặc train/test không cùng preprocessing.

### Cell 5: visualize data

Tạo:

```python
inv_normalize
label_counts
```

`inv_normalize` được các cell visualization sau dùng để hiển thị ảnh tensor trở lại màu tự nhiên.

Cell này cũng giúp kiểm tra:

- Ảnh có đúng class không.
- Augment có bật thật không.
- Phân bố lớp có lệch nhiều không.

### Cell 6 markdown: mô tả kiến trúc

Không tạo biến. Chỉ giải thích MobileNetV2.

### Cell 7: định nghĩa và tạo model

Tạo:

```python
ConvBNReLU6
InvertedResidual
MobileNetV2
model
total_params
trainable_params
```

Cell này phụ thuộc:

```text
CONFIG['num_classes']
CONFIG['width_mult']
CONFIG['dropout']
IMG_SIZE
device
```

Nếu chưa chạy Cell 4, `IMG_SIZE` và `num_classes` có thể sai.

### Cell 8: loss, optimizer, scheduler

Tạo:

```python
criterion
optimizer
WarmupCosineScheduler
scheduler
```

Cell này phụ thuộc:

```text
model
CONFIG
```

Nếu đổi model hoặc đổi config train, nên chạy lại cell này.

### Cell 9: training loop

Tạo:

```python
train_one_epoch
validate
save_checkpoint
load_checkpoint
training_log
best_val_acc
```

Cell này thực sự train model. Nó phụ thuộc gần như toàn bộ các biến trước:

```text
model
train_loader
val_loader
criterion
optimizer
scheduler
device
CONFIG
```

Nếu dừng giữa epoch, epoch đó có thể chưa lưu log. Nếu dừng sau một dòng epoch hoàn chỉnh, checkpoint thường đã lưu.

### Cell 10: biểu đồ training history

Phụ thuộc:

```text
training_log
best_val_acc
CONFIG['log_dir']
```

Nếu chưa train xong ít nhất một epoch, cell này có thể lỗi hoặc biểu đồ rỗng.

### Cell 11: đánh giá test set

Tạo:

```python
all_preds
all_labels
all_probs
test_acc
test_precision
test_recall
test_f1
```

Nó load `best_model.pth` nếu có. Đây là cell quan trọng để lấy kết quả cuối, không nên chỉ dựa vào validation.

### Cell 12: confusion matrix và per-class accuracy

Tạo:

```python
cm
per_class_acc
sorted_acc
```

Phụ thuộc:

```text
all_preds
all_labels
CLASS_NAMES
NUM_CLASSES
```

Nếu chưa chạy Cell 11, cell này thiếu biến.

### Cell 13: sample predictions

Hiển thị ảnh đúng/sai. Phụ thuộc:

```text
all_preds
all_labels
test_dataset
inv_normalize
```

### Cell 14: overall metrics chart

Phụ thuộc:

```text
test_acc
test_precision
test_recall
test_f1
```

### Cell 15: model analysis và export

Lưu model cuối:

```text
mobilenetv2_gtsrb_final.pth
```

File này chứa:

```python
model_state_dict
config
test_accuracy
test_f1
class_names
num_classes
```

### Cell 16 và Cell 18: test 10 ảnh ngẫu nhiên

Hai cell này có chức năng gần giống nhau: lấy 10 ảnh ngẫu nhiên từ test set và dự đoán. Nếu muốn dọn notebook, có thể giữ một trong hai cell để tránh trùng lặp.

### Cell 17: dự đoán ảnh ngoài

Cho upload ảnh, kéo chuột crop vùng biển báo, rồi dự đoán top-5.

Phụ thuộc:

```text
model
MEAN, STD
IMG_SIZE
CLASS_NAMES
device
```

Nếu restart runtime, cần chạy lại các cell tạo model và load checkpoint trước khi dùng cell này.

## 33. Chứng minh và diễn giải công thức

### 33.1. Vì sao `ToTensor` có thể xem như min-max đơn giản

Ảnh 8-bit có pixel:

```text
x_raw ∈ {0, 1, 2, ..., 255}
```

`ToTensor` đổi:

```text
x = x_raw / 255
```

Vì min của ảnh 8-bit là 0 và max là 255, công thức min-max toàn cục là:

```text
x_minmax = (x_raw - 0) / (255 - 0) = x_raw / 255
```

Vậy `ToTensor` tương đương min-max cố định từ `[0,255]` về `[0,1]`.

Sau đó notebook làm thêm Z-score:

```text
z = (x - μ) / σ
```

Vì vậy pipeline đầy đủ là:

```text
0..255 -> 0..1 -> Z-score theo RGB train set
```

### 33.2. Vì sao Z-score làm dữ liệu ổn định hơn

Giả sử một kênh màu có trung bình `μ` và độ lệch chuẩn `σ`.

Sau biến đổi:

```text
z = (x - μ) / σ
```

Kỳ vọng:

```text
E[z] = E[(x - μ) / σ]
     = (E[x] - μ) / σ
     = (μ - μ) / σ
     = 0
```

Phương sai:

```text
Var(z) = Var((x - μ) / σ)
       = Var(x) / σ²
       = σ² / σ²
       = 1
```

Nên sau normalize, mỗi kênh xấp xỉ có mean 0 và std 1. Điều này giúp các layer đầu nhận input ở thang giá trị ổn định.

### 33.3. Công thức tính mean/std trong notebook

Notebook cộng tổng pixel theo từng kênh:

```python
channel_sum += images.sum(dim=[0, 2, 3])
channel_sq_sum += (images ** 2).sum(dim=[0, 2, 3])
pixel_count += images.size(0) * images.size(2) * images.size(3)
```

Mean:

```text
μ = sum(x) / N
```

Std:

```text
σ = sqrt(E[x²] - E[x]²)
```

Vì:

```text
Var(x) = E[(x - μ)²]
       = E[x² - 2μx + μ²]
       = E[x²] - 2μE[x] + μ²
       = E[x²] - 2μ² + μ²
       = E[x²] - μ²
```

Do đó:

```text
std = sqrt(E[x²] - mean²)
```

### 33.4. Cross entropy và gradient trực giác

Softmax:

```text
p_i = exp(z_i) / Σ_j exp(z_j)
```

Cross entropy với one-hot target `y`:

```text
L = -Σ_i y_i log(p_i)
```

Nếu class đúng là `k`, `y_k = 1`, các class khác bằng 0:

```text
L = -log(p_k)
```

Gradient theo logit có dạng:

```text
∂L/∂z_i = p_i - y_i
```

Ý nghĩa:

- Với class đúng: nếu `p_k < 1`, gradient âm, optimizer tăng logit class đúng.
- Với class sai: `p_i > 0`, gradient dương, optimizer giảm logit class sai.

Đây là lý do cross entropy trực tiếp đẩy xác suất về class đúng.

### 33.5. Label smoothing thay đổi target thế nào

Với `C` class và smoothing `ε`, target không còn:

```text
class đúng = 1
class sai = 0
```

mà gần như:

```text
class đúng ≈ 1 - ε
class sai ≈ ε / (C - 1)
```

Tác dụng là giảm việc model quá tự tin tuyệt đối. Khi model quá tự tin, softmax có thể cho xác suất gần 1 cho một class, dễ overfit và calibration kém.

### 33.6. Chứng minh depthwise separable conv tiết kiệm tham số

Convolution thường:

```text
Params_regular = K * K * Cin * Cout
```

Depthwise separable:

```text
Params_depthwise = K * K * Cin
Params_pointwise = Cin * Cout
Params_total = K*K*Cin + Cin*Cout
```

Tỷ lệ:

```text
Params_separable / Params_regular
= (K*K*Cin + Cin*Cout) / (K*K*Cin*Cout)
= 1/Cout + 1/(K*K)
```

Với `K=3`, `Cout=64`:

```text
1/64 + 1/9 ≈ 0.0156 + 0.1111 = 0.1267
```

Tức chỉ khoảng 12.7% số tham số của convolution thường, giảm rất nhiều.

### 33.7. Vì sao skip connection giúp gradient

Block có skip:

```text
y = x + F(x)
```

Gradient:

```text
∂L/∂x = ∂L/∂y * ∂y/∂x
      = ∂L/∂y * (1 + ∂F/∂x)
```

Thành phần `1` cho phép gradient đi trực tiếp qua block. Vì vậy mạng sâu dễ train hơn so với chỉ:

```text
y = F(x)
```

### 33.8. Vì sao augment trước split có thể gây điểm ảo

Giả sử ảnh gốc `A` tạo augment:

```text
A_1, A_2, A_3
```

Nếu augment trước rồi mới split, có thể xảy ra:

```text
train: A_1, A_2
val: A_3
```

Val không còn thật sự độc lập vì `A_3` rất giống ảnh train. Model có thể đạt val accuracy cao không phải vì generalize tốt, mà vì đã thấy biến thể gần như tương tự.

Đúng quy trình:

```text
split ảnh gốc -> augment train only
```

Hoặc:

```text
split ảnh gốc -> online augment train trong DataLoader
```

Val/test phải giữ nguyên.

### 33.9. Overfit dưới dạng quan hệ train/val

Gọi:

```text
E_train = lỗi trên train
E_val = lỗi trên validation
```

Nếu model học tốt thật:

```text
E_train giảm
E_val giảm hoặc ổn định
```

Nếu overfit:

```text
E_train tiếp tục giảm
E_val tăng
```

Khoảng cách:

```text
generalization gap = E_val - E_train
```

Gap lớn và tăng theo epoch là dấu hiệu overfit.

Trong thực tế, dùng loss dễ thấy overfit hơn accuracy vì accuracy có thể bão hòa ở 99-100%.

### 33.10. Vì sao validation accuracy có thể cao hơn train accuracy

Không mâu thuẫn nếu train khó hơn val:

- Train có augment.
- Train có dropout.
- Train dùng nhiều ảnh khó hơn.
- Val không augment.
- Val là ảnh crop sạch.

Trong train, `model.train()` bật dropout và BatchNorm train mode. Trong validation, `model.eval()` tắt dropout. Vì vậy val đôi khi cao hơn train.

## 34. Những điểm có thể cải thiện tiếp

### 34.1. Dọn duplicate random test cell

Notebook hiện có hai cell test 10 ảnh ngẫu nhiên tương tự nhau. Để báo cáo/sử dụng gọn hơn, có thể giữ một cell.

### 34.2. Đổi tên checkpoint final

File export đang tên:

```text
mobilenetv2_gtsrb_final.pth
```

Vì dataset hiện là custom data, có thể đổi thành:

```text
mobilenetv2_custom_traffic_sign_final.pth
```

Đổi tên không ảnh hưởng train, chỉ giúp tránh nhầm với GTSRB.

### 34.3. Cân nhắc giảm epoch

Nếu sau vài epoch val/test đã rất cao, có thể:

```python
'epochs': 15
'warmup_epochs': 2
'patience': 5
```

Không nhất thiết train 50 epoch.

### 34.4. Kiểm tra ảnh ngoài

Chỉ số nội bộ cao chưa đủ. Nên test ảnh ngoài:

```text
ảnh thực tế -> crop biển báo -> predict
```

Nếu ảnh ngoài kém, cần thêm dữ liệu thực tế hoặc tăng augment hợp lý.

### 34.5. Nếu muốn xử lý ảnh nguyên cảnh

Cần thêm detector:

```text
YOLO/SSD/Faster R-CNN -> crop biển báo -> MobileNetV2 classifier
```

Hoặc train một model detection end-to-end.

## 35. Kết quả tự kiểm tra tài liệu

Tài liệu này đã kiểm tra các điểm sau:

1. Có giải thích phạm vi classification, không nhầm detection.
2. Có giải thích cấu trúc dữ liệu và vai trò `SplitData`.
3. Có giải thích toàn bộ tham số chính trong `CONFIG`.
4. Có giải thích normalize là Z-score, không phải min-max.
5. Có giải thích augment online và augment lưu file khác nhau.
6. Có giải thích từng thành phần MobileNetV2.
7. Có công thức depthwise separable convolution và lý do tiết kiệm tham số.
8. Có công thức cross entropy, softmax, label smoothing.
9. Có giải thích SGD, momentum, weight decay, warmup, cosine.
10. Có giải thích mixed precision, gradient clipping, checkpoint, resume.
11. Có giải thích cách đọc log và dấu hiệu overfit.
12. Có phần phản biện về data leakage, val/test cao, crop, ảnh ngoài.
13. Có checklist trước và sau train.

Điểm cần nhớ nhất khi trình bày:

```text
Mô hình không phát hiện biển báo trong ảnh lớn. Mô hình phân loại ảnh biển báo đã crop.
```

Và:

```text
Kết quả cao trên test set nội bộ là hợp lý nếu data crop sạch, nhưng vẫn cần kiểm tra ảnh ngoài để đánh giá khả năng ứng dụng thực tế.
```
