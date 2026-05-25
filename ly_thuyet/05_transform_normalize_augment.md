# Phần 5: Transform, Normalize và Augmentation

## 1. Mục tiêu

Phần này giải thích sâu Cell 4 và Cell 5 liên quan tới:

```text
ToTensor
Normalize
Resize
Augmentation
Train transform
Val/test transform
Inverse normalize để hiển thị
```

Đây là phần rất quan trọng vì model không nhìn ảnh `.jpg` trực tiếp. Model chỉ nhìn tensor sau transform.

## 2. Transform là gì?

Transform là chuỗi xử lý ảnh trước khi đưa vào model.

Ví dụ:

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
```

Input:

```text
PIL Image
```

Output:

```text
torch.Tensor
```

Với ảnh RGB 224x224, output có shape:

```text
[3, 224, 224]
```

## 3. Vì sao cần transform?

Model PyTorch không train trực tiếp trên file ảnh. Nó cần tensor số.

Transform giúp:

```text
đưa ảnh về tensor
đưa pixel về thang giá trị phù hợp
chuẩn hóa phân phối input
tạo biến thể ảnh train nếu augment
đảm bảo train/val/test cùng preprocessing cần thiết
```

Nếu transform sai, model có thể:

```text
train rất chậm
accuracy thấp
test ảnh ngoài sai
shape mismatch
hiển thị ảnh sai màu
```

## 4. `ToTensor`

`ToTensor` làm hai việc chính:

1. Đổi pixel từ `0..255` sang `0..1`.
2. Đổi shape từ `H x W x C` sang `C x H x W`.

Ảnh ban đầu:

```text
height x width x channel
224 x 224 x 3
```

Sau `ToTensor`:

```text
3 x 224 x 224
```

Pixel:

```text
0 -> 0.0
255 -> 1.0
128 -> 0.5019
```

Về mặt toán học:

```text
x_tensor = x_raw / 255
```

Đây giống một dạng min-max cố định cho ảnh 8-bit:

```text
(x - 0) / (255 - 0)
```

## 5. Normalize trong notebook là loại gì?

Notebook dùng:

```python
transforms.Normalize(MEAN, STD)
```

Đây là **Z-score normalization theo từng kênh RGB**.

Công thức:

```text
x_norm = (x - mean) / std
```

Trong đó:

```text
x đã nằm trong 0..1 sau ToTensor
mean là trung bình kênh màu tính từ train set
std là độ lệch chuẩn kênh màu tính từ train set
```

Với RGB:

```python
MEAN = [mean_R, mean_G, mean_B]
STD = [std_R, std_G, std_B]
```

Mỗi kênh được normalize riêng.

## 6. Chứng minh Z-score đưa mean về 0, std về 1

Gọi pixel một kênh là biến ngẫu nhiên `X`, có:

```text
E[X] = μ
Std(X) = σ
```

Normalize:

```text
Z = (X - μ) / σ
```

Kỳ vọng:

```text
E[Z] = E[(X - μ) / σ]
     = (E[X] - μ) / σ
     = (μ - μ) / σ
     = 0
```

Phương sai:

```text
Var(Z) = Var((X - μ) / σ)
       = Var(X) / σ²
       = σ² / σ²
       = 1
```

Vậy sau normalize, dữ liệu xấp xỉ có:

```text
mean = 0
std = 1
```

Điều này giúp model train ổn định hơn.

## 7. Vì sao tính mean/std từ train set?

Notebook tính:

```python
stats_dataset = ImageFolder(TRAIN_SOURCE_DIR, transform=stats_transform)
```

Chỉ dùng train để tính mean/std.

Lý do:

```text
val/test phải đóng vai trò dữ liệu chưa biết
```

Nếu dùng cả val/test để tính mean/std, ta đã dùng thông tin thống kê từ tập đánh giá. Đây là rò rỉ nhẹ.

Quy tắc sạch:

```text
fit preprocessing trên train
apply preprocessing cho train/val/test
```

## 8. Công thức tính mean/std trong code

Notebook dùng:

```python
channel_sum += images.sum(dim=[0, 2, 3])
channel_sq_sum += (images ** 2).sum(dim=[0, 2, 3])
pixel_count += images.size(0) * images.size(2) * images.size(3)
```

Mean:

```text
mean = sum(x) / N
```

Std:

```text
std = sqrt(E[x²] - E[x]²)
```

Vì:

```text
Var(X) = E[X²] - E[X]²
Std(X) = sqrt(Var(X))
```

## 9. Vì sao lấy mẫu tối đa 3000 ảnh?

Notebook có:

```python
MEAN_STD_MAX_IMAGES = 3000
```

Nếu train set rất lớn, tính toàn bộ ảnh có thể lâu. Lấy mẫu giúp nhanh hơn.

Ưu điểm:

```text
tính nhanh
đủ chính xác nếu sample đại diện tốt
```

Nhược điểm:

```text
mean/std là ước lượng
```

Nếu muốn chính xác tuyệt đối hơn, có thể tăng hoặc bỏ giới hạn.

## 10. Train transform và val/test transform khác nhau thế nào?

Notebook có hai transform chính:

```text
train_transform
val_transform
```

Nếu augment tắt:

```text
train: ToTensor + Normalize
val/test: ToTensor + Normalize
```

Nếu augment bật:

```text
train: augment + ToTensor + Normalize
val/test: ToTensor + Normalize
```

Val/test không augment để metric ổn định và công bằng.

## 11. Vì sao không augment val/test?

Val/test dùng để đánh giá. Nếu augment val/test ngẫu nhiên:

```text
mỗi lần chạy metric có thể khác
dữ liệu đánh giá không cố định
khó so sánh model
```

Hơn nữa, validation cần phản ánh dữ liệu thật trong phạm vi bài toán.

Do đó:

```text
augment chỉ dành cho train
```

## 12. Augmentation là gì?

Augmentation là tạo biến thể của ảnh train để model học robust hơn.

Mục tiêu:

```text
không học thuộc ảnh y nguyên
học đặc trưng bền vững của biển báo
chịu được lệch, sáng tối, mờ nhẹ, góc nhìn nhẹ
```

Augment không thay đổi label nếu biến đổi vẫn giữ class nhận ra được.

Ví dụ:

```text
xoay nhẹ biển stop vẫn là stop_sign
làm tối nhẹ biển cấm vào vẫn là no_entry
crop hơi lệch nhưng còn đủ biển vẫn là class cũ
```

## 13. Các augment trong notebook

Khi:

```python
'augment_enabled': 1
```

notebook dùng:

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

## 14. `RandomCrop`

```python
transforms.RandomCrop((IMG_SIZE, IMG_SIZE))
```

RandomCrop cắt một vùng kích thước `IMG_SIZE x IMG_SIZE`.

Trong notebook, nếu resize bật, trước đó có:

```python
maybe_resize(IMG_SIZE, extra_pixels=8)
```

Nghĩa là resize lên `IMG_SIZE + 8`, rồi crop về `IMG_SIZE`. Điều này tạo lệch nhẹ.

Tác dụng:

```text
model chịu được biển báo không nằm đúng giữa ảnh
```

Rủi ro:

```text
nếu crop làm mất ký hiệu chính, label trở nên nhiễu
```

## 15. `RandomRotation`

```python
transforms.RandomRotation(20)
```

Xoay ảnh ngẫu nhiên trong khoảng:

```text
-20 độ đến +20 độ
```

Tác dụng:

```text
chịu được camera nghiêng
chịu được biển báo không thẳng tuyệt đối
```

Nếu xoay quá mạnh, ảnh có thể không thực tế.

## 16. `RandomAffine`

```python
transforms.RandomAffine(
    degrees=0,
    translate=(0.15, 0.15),
    scale=(0.8, 1.2),
    shear=10
)
```

Affine gồm:

```text
translate: dịch ảnh ngang/dọc tối đa 15%
scale: zoom từ 0.8 đến 1.2
shear: nghiêng hình tối đa 10 độ
```

Tác dụng:

```text
biển báo có thể lệch vị trí
to/nhỏ khác nhau
góc chụp hơi nghiêng
```

## 17. `RandomPerspective`

```python
transforms.RandomPerspective(distortion_scale=0.3, p=0.5)
```

Biến dạng phối cảnh với xác suất 50%.

Tác dụng:

```text
mô phỏng ảnh chụp không chính diện
```

Nếu quá mạnh, hình biển báo có thể méo không thực tế.

## 18. `ColorJitter`

```python
transforms.ColorJitter(
    brightness=0.5,
    contrast=0.5,
    saturation=0.5,
    hue=0.1
)
```

Thay đổi:

```text
brightness: độ sáng
contrast: tương phản
saturation: độ bão hòa màu
hue: sắc thái màu
```

Tác dụng:

```text
chịu được ánh sáng khác nhau
camera khác nhau
thời tiết khác nhau
```

Rủi ro: biển báo phụ thuộc màu. Nếu đổi màu quá mạnh, class có thể bị nhiễu.

## 19. `RandomGrayscale`

```python
transforms.RandomGrayscale(p=0.05)
```

5% ảnh train được chuyển grayscale.

Tác dụng:

```text
giúp model không phụ thuộc tuyệt đối vào màu
```

Nhưng với biển báo, màu là thông tin quan trọng. Vì vậy xác suất thấp `0.05` là hợp lý hơn xác suất cao.

## 20. `GaussianBlur`

```python
transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
```

Làm mờ ảnh.

Tác dụng:

```text
mô phỏng ảnh rung nhẹ
camera kém nét
```

Nếu blur quá mạnh, số và ký hiệu có thể mất.

## 21. `RandomErasing`

```python
transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
```

Che ngẫu nhiên một vùng trên tensor sau `ToTensor`.

Tác dụng:

```text
model chịu được che khuất nhẹ
không phụ thuộc vào một vùng nhỏ duy nhất
```

Rủi ro:

```text
nếu che đúng ký hiệu chính, ảnh có thể mất thông tin
```

Xác suất `0.2` nghĩa là 20% ảnh bị erasing.

## 22. Thứ tự transform có quan trọng không?

Có.

Các transform hình học thường làm trên PIL image:

```text
Resize/Crop/Rotate/Affine/Perspective/ColorJitter/Blur
```

Sau đó:

```text
ToTensor
RandomErasing
Normalize
```

`RandomErasing` thường làm trên tensor, nên đặt sau `ToTensor`.

Normalize thường đặt cuối, vì các augment màu/hình nên làm trên ảnh chưa normalize.

## 23. Vì sao augment preview giống ảnh gốc khi augment tắt?

Nếu:

```python
'augment_enabled': 0
```

thì:

```python
base_augment_steps = maybe_resize(IMG_SIZE) + [transforms.ToTensor()]
```

Nếu resize cũng tắt, transform chỉ là `ToTensor`. Khi hiển thị lại, ảnh trông giống ảnh gốc.

Vì vậy hình `Augment 1..7` giống nhau là đúng khi augment tắt.

## 24. Augment online tạo bao nhiêu ảnh?

Không tạo file mới.

Mỗi lần DataLoader lấy ảnh, transform tạo một biến thể tạm thời.

Nếu train 20 epoch, mỗi ảnh gốc có thể được nhìn 20 lần, mỗi lần biến đổi khác nhau.

So sánh:

```text
augment lưu file: 1 ảnh -> 3-5 file mới
augment online: 1 ảnh -> biến thể ngẫu nhiên mỗi lần đọc, không lưu
```

## 25. Khi nào nên bật augment trong notebook?

Nếu train đã chạy lại từ `RGBData`, chưa augment sẵn:

```python
'augment_enabled': 1
```

Nếu train đã có ảnh augment lưu file:

```python
'augment_enabled': 0
```

Nếu accuracy train rất cao nhưng val/test hoặc ảnh ngoài kém, có thể bật augment để tăng robust.

## 26. Normalize khi predict ảnh ngoài

Ảnh ngoài phải dùng cùng transform chuẩn hóa:

```python
transforms.Resize((IMG_SIZE, IMG_SIZE))
transforms.ToTensor()
transforms.Normalize(MEAN, STD)
```

Nếu train dùng mean/std này nhưng predict ảnh ngoài không normalize, input distribution khác, model dễ sai.

Nguyên tắc:

```text
Train xử lý ảnh thế nào, inference cũng phải xử lý giống vậy.
```

## 27. Inverse normalize để hiển thị

Ảnh đã normalize không nên hiển thị trực tiếp. Giá trị có thể âm hoặc lớn hơn 1.

Notebook dùng:

```python
inv_normalize = transforms.Compose([
    transforms.Normalize(mean=[-m/s for m, s in zip(MEAN, STD)],
                         std=[1/s for s in STD])
])
```

Chứng minh:

Normalize ban đầu:

```text
y = (x - m) / s
```

Muốn lấy lại `x`:

```text
x = y*s + m
```

`transforms.Normalize(mean=a, std=b)` làm:

```text
z = (y - a) / b
```

Chọn:

```text
a = -m/s
b = 1/s
```

thì:

```text
z = (y + m/s) / (1/s)
  = y*s + m
  = x
```

## 28. Các lỗi thường gặp

### 28.1. Shape sai

Nếu ảnh không RGB:

```text
expected input to have 3 channels
```

Cần đảm bảo ảnh là RGB.

### 28.2. Batch stack lỗi

Nếu ảnh khác kích thước và không resize:

```text
stack expects each tensor to be equal size
```

Cách sửa:

```python
'resize_enabled': 1
```

### 28.3. Ảnh hiển thị sai màu

Do hiển thị ảnh đã normalize trực tiếp. Cần inverse normalize.

### 28.4. Augment quá mạnh làm model học chậm

Nếu train accuracy thấp bất thường, thử:

```python
'augment_enabled': 0
```

hoặc giảm độ mạnh augment.

## 29. Cách trình bày trong báo cáo

Có thể viết:

```text
Ảnh đầu vào được chuyển thành tensor bằng ToTensor, đưa pixel từ miền 0-255 về 0-1. Sau đó ảnh được chuẩn hóa bằng Z-score normalization theo từng kênh RGB với mean và std tính từ tập train. Đối với tập train, có thể áp dụng augmentation online như xoay, dịch chuyển, biến dạng phối cảnh, thay đổi sáng/tương phản, blur và random erasing. Validation và test không sử dụng augmentation để đảm bảo đánh giá ổn định.
```

## 30. Câu hỏi phản biện thường gặp

### 30.1. Normalize là min-max hay Z-score?

Cả hai bước đều có mặt:

```text
ToTensor: 0-255 -> 0-1, giống min-max cố định
Normalize: Z-score theo mean/std
```

Nếu hỏi chính normalize trong `transforms.Normalize`, trả lời:

```text
Z-score normalization
```

### 30.2. Vì sao không tính mean/std từ toàn bộ data?

Vì val/test phải là dữ liệu chưa biết. Chỉ tính từ train để tránh rò rỉ thông tin.

### 30.3. Vì sao augment không lưu ảnh?

Online augment tiết kiệm ổ cứng và tạo biến thể khác nhau qua các epoch.

### 30.4. Vì sao val/test không augment?

Để metric cố định, công bằng, phản ánh dữ liệu đánh giá thật.

## 31. Checklist transform

```text
[ ] Ảnh input là RGB
[ ] Ảnh cùng size hoặc resize_enabled=1
[ ] ToTensor trước Normalize
[ ] MEAN/STD tính từ train
[ ] Train dùng augment nếu cần
[ ] Val/test không augment
[ ] Predict ảnh ngoài dùng cùng MEAN/STD
[ ] Hiển thị ảnh dùng inverse normalize
```

## 32. Kết luận phần 5

Transform quyết định model thật sự nhìn thấy gì.

Điểm cốt lõi:

```text
ToTensor đưa ảnh về tensor 0..1.
Normalize là Z-score theo train mean/std.
Augment chỉ nên áp dụng cho train.
Val/test giữ transform ổn định.
Inference phải dùng preprocessing giống train.
```

Nếu transform nhất quán, kết quả model đáng tin hơn. Nếu transform train/test khác nhau, accuracy có thể giảm mạnh dù model architecture không đổi.
