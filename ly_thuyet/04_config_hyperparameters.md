# Phần 4: Config và hyperparameters

## 1. Mục tiêu

Phần này giải thích toàn bộ `CONFIG` trong Cell 3:

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

Hyperparameter là tham số do người huấn luyện chọn trước khi train. Nó khác với parameter của model:

```text
parameter: trọng số model học được, ví dụ weight của convolution
hyperparameter: cấu hình do mình đặt, ví dụ learning rate, batch size, epochs
```

Nếu chọn hyperparameter sai, model có thể:

```text
train chậm
không hội tụ
overfit
underfit
OOM GPU
kết quả không ổn định
```

## 2. `img_size`

```python
'img_size': 224
```

Đây là kích thước ảnh đầu vào mà ta muốn model xử lý.

Ảnh input vào model có shape:

```text
[batch_size, 3, img_size, img_size]
```

Với `img_size = 224` và `batch_size = 32`:

```text
[32, 3, 224, 224]
```

### 2.1. Ảnh hưởng của `img_size`

`img_size` lớn:

```text
giữ nhiều chi tiết hơn
có lợi cho biển tốc độ vì chữ số rõ hơn
tốn VRAM hơn
train chậm hơn
```

`img_size` nhỏ:

```text
train nhanh hơn
ít tốn VRAM hơn
có thể mất chi tiết nhỏ
```

Với biển báo đã crop, `224x224` là lựa chọn an toàn.

### 2.2. Vì sao không dùng 96 nữa?

Trước đó notebook từng dùng `96x96`. Nhưng dữ liệu hiện đã resize sẵn `224x224`. Nếu ép xuống `96x96`, các chi tiết như số `30`, `40`, `50`, `60` có thể mờ hơn.

Vì vậy cấu hình hiện tại:

```python
'img_size': 224
```

hợp với dữ liệu hơn.

## 3. `resize_enabled`

```python
'resize_enabled': 0
```

Ý nghĩa:

```text
0: không resize lại trong notebook
1: resize ảnh về resize_size trong transform
```

Vì ảnh trong `SplitData` đã là `224x224`, để `0` giúp tránh resize thừa.

Nếu dataset mới có ảnh khác kích thước, phải bật:

```python
'resize_enabled': 1
```

Nếu không, DataLoader có thể lỗi khi gom batch:

```text
stack expects each tensor to be equal size
```

## 4. `resize_size`

```python
'resize_size': 224
```

Đây là kích thước dùng khi `resize_enabled = 1`.

Notebook lấy:

```python
IMG_SIZE = CONFIG.get('resize_size', CONFIG['img_size'])
```

Vì vậy `resize_size` cũng ảnh hưởng đến:

```text
transform
dummy input test model
predict ảnh ngoài
```

Nếu ảnh đã chuẩn bị sẵn `224x224`, đặt:

```python
'resize_enabled': 0,
'resize_size': 224
```

Nếu ảnh đã chuẩn bị sẵn `128x128`, đặt:

```python
'resize_enabled': 0,
'resize_size': 128
```

## 5. `augment_enabled`

```python
'augment_enabled': 0
```

Ý nghĩa:

```text
0: không augment online
1: augment online khi train
```

Online augment không tạo file mới. Nó biến đổi ảnh tạm thời khi DataLoader đọc ảnh.

Khi vừa chạy lại split từ `RGBData`, train không còn ảnh augment sẵn. Khi đó có thể bật:

```python
'augment_enabled': 1
```

Nếu train đã có ảnh augment lưu file, nên để:

```python
'augment_enabled': 0
```

để tránh augment chồng augment.

## 6. `batch_size`

```python
'batch_size': 32
```

Batch size là số ảnh xử lý trong một lần forward/backward.

Với train set 8578 ảnh:

```text
8578 / 32 ≈ 269 batch mỗi epoch
```

### 6.1. Batch size ảnh hưởng gì?

Batch lớn:

```text
gradient ổn định hơn
tận dụng GPU tốt hơn
mỗi epoch có ít step hơn
tốn VRAM hơn
```

Batch nhỏ:

```text
đỡ OOM
gradient nhiễu hơn
có thể generalize tốt
train có thể lâu hơn
```

### 6.2. Khi nào giảm batch size?

Khi gặp:

```text
CUDA out of memory
```

giảm:

```python
'batch_size': 16
```

hoặc:

```python
'batch_size': 8
```

Sau OOM nặng, nên restart runtime vì GPU memory có thể bị giữ lại.

## 7. `epochs`

```python
'epochs': 50
```

Một epoch là model đi qua toàn bộ train set một lần.

Nếu:

```text
train = 8578 ảnh
batch_size = 32
```

thì epoch 1 đã có khoảng 269 lần cập nhật weight. Vì vậy accuracy epoch 1 cao không nhất thiết lạ.

### 7.1. Epoch quá ít

Model chưa học đủ:

```text
train loss còn cao
val acc còn tăng mạnh
```

### 7.2. Epoch quá nhiều

Có thể:

```text
tốn thời gian
overfit
val loss tăng
```

Vì notebook có early stopping qua `patience`, đặt `epochs = 50` không có nghĩa chắc chắn train đủ 50 epoch.

## 8. `lr`

```python
'lr': 0.01
```

Learning rate là độ lớn bước cập nhật trọng số.

Cập nhật đơn giản:

```text
W_new = W_old - lr * gradient
```

Learning rate quá cao:

```text
loss dao động
không hội tụ
có thể NaN
```

Learning rate quá thấp:

```text
train chậm
dễ kẹt ở kết quả chưa tốt
```

Notebook không dùng `lr` cố định. Nó dùng warmup + cosine scheduler.

## 9. `momentum`

```python
'momentum': 0.9
```

Momentum giúp SGD có quán tính.

Trực giác:

```text
Nếu gradient nhiều batch liên tục cùng hướng, momentum giúp đi nhanh hơn.
Nếu gradient nhiễu, momentum giúp đường đi mượt hơn.
```

Với SGD, `momentum = 0.9` là giá trị phổ biến.

## 10. `weight_decay`

```python
'weight_decay': 1e-4
```

Weight decay phạt trọng số lớn.

Loss hiệu dụng:

```text
loss_total = loss_data + λ * ||W||²
```

Trong đó:

```text
λ = 1e-4
```

Tác dụng:

```text
giảm overfit
làm trọng số nhỏ và mượt hơn
```

Nếu quá lớn, model có thể underfit vì bị ép weight quá mạnh.

## 11. `num_classes`

```python
'num_classes': 43
```

Trong config ban đầu có `43`, nhưng Cell 4 cập nhật lại:

```python
CONFIG['num_classes'] = len(CLASS_NAMES)
```

Với dataset hiện tại 12 lớp:

```text
num_classes = 12
```

Tại sao cần đúng số lớp?

Classifier cuối:

```python
nn.Linear(last_channels, num_classes)
```

Nếu số output không bằng số class, loss sẽ sai hoặc mapping class sai.

## 12. `width_mult`

```python
'width_mult': 1.0
```

`width_mult` điều chỉnh độ rộng MobileNetV2.

```text
1.0: bản chuẩn
0.75: ít channel hơn, nhẹ hơn
1.4: nhiều channel hơn, mạnh hơn nhưng tốn hơn
```

Số channel được nhân với `width_mult`, rồi làm tròn bằng `_make_divisible`.

Nếu OOM hoặc muốn model nhẹ:

```python
'width_mult': 0.75
```

Nếu dữ liệu khó và GPU đủ:

```python
'width_mult': 1.2
```

nhưng notebook hiện tại chưa cần.

## 13. `warmup_epochs`

```python
'warmup_epochs': 5
```

Warmup tăng learning rate từ thấp lên `lr` chính.

Với:

```python
lr = 0.01
warmup_epochs = 5
```

LR xấp xỉ:

```text
epoch 1: 0.002
epoch 2: 0.004
epoch 3: 0.006
epoch 4: 0.008
epoch 5: 0.010
```

Tác dụng:

```text
training đầu ổn định hơn
tránh cập nhật quá mạnh lúc weight còn random
```

Nếu model học quá nhanh, có thể giảm:

```python
'warmup_epochs': 2
```

## 14. `label_smoothing`

```python
'label_smoothing': 0.1
```

Thay vì target tuyệt đối:

```text
class đúng = 1
class sai = 0
```

label smoothing làm target mềm hơn:

```text
class đúng ≈ 0.9
class sai chia phần nhỏ còn lại
```

Tác dụng:

```text
giảm overconfidence
cải thiện generalization
giúp model không quá chắc chắn tuyệt đối
```

Nếu smoothing quá cao, model khó học rõ class.

## 15. `dropout`

```python
'dropout': 0.2
```

Dropout tắt ngẫu nhiên một phần neuron khi train.

Trong classifier:

```python
nn.Dropout(p=dropout)
nn.Linear(last_channels, num_classes)
```

Tác dụng:

```text
giảm overfit
buộc model không phụ thuộc vào một feature duy nhất
```

Khi evaluate:

```python
model.eval()
```

dropout tự tắt.

## 16. `grad_clip`

```python
'grad_clip': 5.0
```

Gradient clipping giới hạn norm của gradient:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
```

Nếu gradient quá lớn:

```text
weight update quá mạnh
loss nhảy
training bất ổn
```

Clipping giúp ổn định, nhất là khi train từ đầu.

## 17. `patience`

```python
'patience': 10
```

Early stopping:

```text
nếu val_acc không cải thiện trong 10 epoch liên tiếp -> dừng
```

Tác dụng:

```text
tránh train quá lâu
giảm overfit
tiết kiệm GPU
```

Nếu dataset học rất nhanh, có thể giảm:

```python
'patience': 5
```

## 18. `save_every`

```python
'save_every': 10
```

Cứ mỗi 10 epoch lưu checkpoint định kỳ:

```text
checkpoint_epoch_10.pth
checkpoint_epoch_20.pth
```

Ngoài ra notebook luôn lưu:

```text
best_model.pth
checkpoint_latest.pth
```

## 19. `seed`

```python
'seed': 42
```

Seed giúp kết quả tái lập hơn:

```python
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
random.seed(CONFIG['seed'])
```

Tuy vậy GPU có thể vẫn có một số phép toán không hoàn toàn deterministic.

## 20. `resume`

```python
'resume': False
```

Nếu `False`:

```text
xóa checkpoint/log cũ
train mới
```

Nếu `True`:

```text
load checkpoint_latest.pth
train tiếp
```

Cẩn thận: nếu chỉ muốn test model đã train, không cần bật resume. Chỉ cần tạo model rồi chạy cell evaluation để load `best_model.pth`.

## 21. `checkpoint_dir` và `log_dir`

```python
'checkpoint_dir': '/content/drive/MyDrive/mobilenetv2_gtsrb/checkpoints/'
'log_dir': '/content/drive/MyDrive/mobilenetv2_gtsrb/logs/'
```

Lưu trên Drive để không mất khi runtime Colab tắt.

Checkpoint gồm:

```text
model_state_dict
optimizer_state_dict
scheduler_state_dict
epoch
best_val_acc
training_log
```

Log gồm:

```text
training_log.csv
training_history.png
confusion_matrix.png
sample images
```

## 22. Quan hệ giữa các hyperparameter

Hyperparameter không độc lập hoàn toàn.

### 22.1. `img_size`, `batch_size`, VRAM

Ảnh lớn hơn làm tensor và feature map lớn hơn.

Nếu tăng:

```python
img_size: 224 -> 320
```

có thể phải giảm:

```python
batch_size: 32 -> 16
```

### 22.2. `lr` và `batch_size`

Batch lớn thường chịu được LR lớn hơn. Batch nhỏ gradient nhiễu hơn, đôi khi cần LR nhỏ hơn.

### 22.3. `epochs`, `patience`, `warmup_epochs`

Nếu `epochs` thấp mà `warmup_epochs` quá cao, phần lớn training chỉ ở giai đoạn warmup.

Ví dụ:

```text
epochs = 5
warmup_epochs = 5
```

thì chưa có giai đoạn cosine decay. Nếu train ngắn, nên giảm warmup.

### 22.4. `augment_enabled` và overfit

Bật augment giúp giảm overfit nhưng có thể làm train khó hơn.

Nếu augment quá mạnh:

```text
train acc thấp
val có thể không tăng
model học chậm
```

Nếu không augment và dữ liệu ít:

```text
train acc cao nhanh
val/test có thể kém
```

## 23. Cấu hình khuyến nghị cho dataset hiện tại

Nếu dùng `SplitData` đã chạy lại từ `RGBData`, chưa augment lưu file:

```python
'img_size': 224,
'resize_enabled': 0,
'resize_size': 224,
'augment_enabled': 1,
'batch_size': 32,
'epochs': 20,
'lr': 0.01,
'warmup_epochs': 2,
'patience': 5,
```

Nếu bị OOM:

```python
'batch_size': 16
```

Nếu train quá nhanh và val đã bão hòa:

```python
'epochs': 15
'patience': 4
```

## 24. Cấu hình an toàn khi debug

Để test pipeline nhanh:

```python
'batch_size': 16,
'epochs': 2,
'warmup_epochs': 1,
'patience': 2,
```

Mục tiêu không phải đạt accuracy cao, mà kiểm tra:

```text
data load được
model forward được
loss giảm
checkpoint lưu được
evaluation chạy được
```

## 25. Cách trình bày trong báo cáo

Có thể viết:

```text
Các siêu tham số chính gồm kích thước ảnh đầu vào 224x224, batch size 32, learning rate ban đầu 0.01, optimizer SGD với momentum 0.9 và weight decay 1e-4. Mô hình sử dụng label smoothing 0.1, dropout 0.2, gradient clipping 5.0, warmup learning rate trong các epoch đầu và cosine annealing cho các epoch còn lại. Checkpoint được lưu trên Google Drive để có thể khôi phục quá trình train khi Colab bị ngắt.
```

## 26. Câu hỏi phản biện thường gặp

### 26.1. Vì sao batch size không để 128?

Với ảnh 224x224 và MobileNetV2 custom, batch 128 dễ OOM trên GPU Colab 14GB. Batch 32 cân bằng giữa tốc độ và VRAM.

### 26.2. Vì sao `num_classes` trong config là 43 nhưng data có 12 lớp?

Giá trị 43 là mặc định cũ. Cell 4 tự cập nhật:

```python
CONFIG['num_classes'] = len(CLASS_NAMES)
```

Do đó model thực tế dùng 12 output nếu dataset có 12 class.

### 26.3. Vì sao cần seed?

Seed giúp split, shuffle, init weight và augment ngẫu nhiên ổn định hơn giữa các lần chạy.

### 26.4. Vì sao lưu checkpoint lên Drive?

Vì `/content` là bộ nhớ tạm. Nếu Colab reset, checkpoint ở `/content` mất. Drive bền hơn.

## 27. Checklist config trước khi train

```text
[ ] img_size đúng với ảnh
[ ] resize_enabled phù hợp
[ ] resize_size đúng
[ ] augment_enabled đúng với trạng thái data
[ ] batch_size không quá lớn
[ ] epochs không quá dài
[ ] warmup_epochs không quá lớn so với epochs
[ ] num_classes được cập nhật sau Cell 4
[ ] checkpoint_dir/log_dir nằm trên Drive
[ ] resume đúng mục đích
```

## 28. Kết luận phần 4

`CONFIG` là trung tâm điều khiển notebook. Nó quyết định:

```text
ảnh vào model như thế nào
model lớn cỡ nào
train nhanh hay chậm
có dễ OOM không
có regularization không
có resume/checkpoint không
```

Hiểu `CONFIG` giúp bạn không chỉ chạy notebook mà còn biết vì sao kết quả thay đổi khi chỉnh từng tham số.
