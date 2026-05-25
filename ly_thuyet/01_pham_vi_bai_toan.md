# Phần 1: Phạm vi bài toán

## 1. Mục tiêu của phần này

Phần này trả lời câu hỏi quan trọng nhất trước khi học model:

```text
Bài toán này thật ra đang giải quyết điều gì?
```

Notebook `mobilenetv2_gtsrb_ver3_datatulam.ipynb` đang giải bài toán:

```text
Phân loại loại biển báo giao thông từ ảnh đã crop vùng biển báo.
```

Nói cách khác, model nhận một ảnh mà biển báo đã là chủ thể chính, rồi dự đoán ảnh đó thuộc lớp nào.

Ví dụ:

```text
ảnh biển cấm vào -> no_entry
ảnh biển stop -> stop_sign
ảnh biển giới hạn 50 -> speed_limit_50
ảnh biển công trường -> roadworks
```

Model hiện tại **không tự tìm vị trí biển báo trong ảnh đường phố nguyên cảnh**.

## 2. Bài toán classification là gì?

Classification, hay phân loại ảnh, là bài toán:

```text
input: một ảnh
output: một nhãn lớp
```

Với bài này:

```text
input: ảnh biển báo đã crop/resize/RGB
output: tên lớp biển báo
```

Ví dụ input/output:

```text
Input: ảnh biển báo "cấm vào"
Output: no_entry
```

```text
Input: ảnh biển báo "dừng lại"
Output: stop_sign
```

Mỗi ảnh chỉ được gán một nhãn chính. Model không cần trả về tọa độ, không cần vẽ bounding box, không cần biết trong ảnh có bao nhiêu biển báo.

## 3. Bài toán detection là gì?

Detection, hay phát hiện đối tượng, là bài toán phức tạp hơn:

```text
input: một ảnh nguyên cảnh
output: vị trí đối tượng + nhãn lớp
```

Ví dụ:

```text
Input: ảnh đường phố có xe, cây, nhà, trời, biển báo
Output:
  bounding box: x1, y1, x2, y2
  class: no_entry
```

Detection phải trả lời hai câu hỏi:

```text
1. Biển báo nằm ở đâu?
2. Biển báo đó thuộc loại nào?
```

Notebook hiện tại chỉ trả lời câu hỏi thứ hai, với giả định vùng biển báo đã được crop trước.

## 4. So sánh classification và detection

| Tiêu chí | Classification | Detection |
|---|---|---|
| Input | Ảnh đã chứa chủ thể chính | Ảnh nguyên cảnh |
| Output | Một class | Bounding box + class |
| Ví dụ model | MobileNetV2, ResNet, ViT classifier | YOLO, SSD, Faster R-CNN |
| Có cần crop trước không? | Có, nếu ảnh ngoài là nguyên cảnh | Không, detector tự tìm |
| Notebook hiện tại | Có | Không |

Tóm gọn:

```text
Classification = đây là biển gì?
Detection = biển ở đâu và là biển gì?
```

## 5. Vì sao notebook này cần ảnh đã crop?

Model được train trên dữ liệu đã qua pipeline:

```text
ảnh gốc
-> crop vùng biển báo
-> resize 224x224
-> chuyển RGB
-> chia train/val/test
-> train MobileNetV2 classifier
```

Vì vậy model học trên ảnh có dạng:

```text
biển báo nằm lớn ở trung tâm hoặc gần trung tâm ảnh
ít background thừa
kích thước ổn định
RGB 3 kênh
```

Nếu đưa ảnh nguyên cảnh vào model, ảnh sẽ có dạng khác:

```text
biển báo nhỏ
nhiều đường, xe, cây, nhà, trời
vị trí biển báo không cố định
background chiếm phần lớn ảnh
```

Hai kiểu input này khác nhau. Model đã học một kiểu nhưng bị test bằng kiểu khác thì rất dễ sai.

Đây gọi là khác phân phối dữ liệu:

```text
train distribution != test distribution
```

## 6. Ví dụ đúng phạm vi

Các input sau phù hợp với model:

```text
ảnh crop sát biển cấm vào
ảnh crop sát biển stop
ảnh crop sát biển giới hạn tốc độ
ảnh có hơi dư nền nhưng biển báo vẫn chiếm phần lớn ảnh
```

Ví dụ pipeline đúng khi test ảnh ngoài:

```text
ảnh ngoài nguyên tấm
-> kéo chuột crop vùng biển báo
-> resize/normalize giống train
-> MobileNetV2 dự đoán class
```

Cell dự đoán ảnh ngoài trong notebook làm đúng hướng này: cho upload ảnh, kéo chuột crop, rồi mới predict.

## 7. Ví dụ sai phạm vi

Các input sau không phù hợp nếu đưa thẳng vào classifier:

```text
ảnh đường phố nguyên cảnh chưa crop
ảnh có nhiều biển báo cùng lúc
ảnh biển báo quá nhỏ trong góc ảnh
ảnh nền chiếm 90% diện tích
ảnh chỉ thấy một phần rất nhỏ của biển báo
```

Nếu đưa thẳng ảnh như vậy vào MobileNetV2 classifier, model vẫn sẽ buộc phải dự đoán một class, nhưng dự đoán đó không đáng tin.

Lý do: classifier không có cơ chế tìm vị trí biển báo. Nó chỉ nhìn toàn ảnh sau resize và cố phân loại.

## 8. Crop sát hay crop dư?

Với classification, crop nên đủ chứa toàn bộ biển báo.

Crop tốt:

```text
biển báo đầy đủ
biển báo là chủ thể chính
có thể dư một ít nền xung quanh
```

Crop hơi dư vẫn chấp nhận được:

```text
có thêm ít trời, cây, đường
biển báo vẫn rõ và chiếm phần lớn ảnh
```

Crop xấu:

```text
mất ký hiệu chính
mất nhiều viền hoặc mất số
chỉ còn một góc biển báo
background quá nhiều
```

Nếu crop thiếu thông tin quan trọng, người cũng khó phân loại, nên model sai là hợp lý.

## 9. Vậy test set có cần crop không?

Có.

Nếu bài toán là classification từ ảnh crop, thì train/val/test nên cùng kiểu dữ liệu:

```text
train: ảnh crop biển báo
val: ảnh crop biển báo
test: ảnh crop biển báo
```

Nếu train crop mà test nguyên cảnh, bài toán test đã đổi thành detection/classification hỗn hợp, không còn đánh giá đúng classifier nữa.

Vì vậy test set hiện tại trong `SplitData/test` là ảnh đã crop/resize là đúng.

## 10. Vậy train set có nên có ảnh crop xấu không?

Có thể có, nhưng ở mức hợp lý.

Để model robust hơn ngoài thực tế, train nên có một số biến thể:

```text
crop hơi lệch
crop hơi dư nền
ảnh sáng/tối khác nhau
ảnh hơi mờ
ảnh hơi nghiêng
biển báo hơi nhỏ hơn một chút
```

Nhưng không nên có quá nhiều ảnh:

```text
mất nửa biển báo
che mất ký hiệu chính
crop chỉ còn viền
ảnh không còn đủ thông tin để phân loại
```

Nguyên tắc:

```text
Nếu con người nhìn crop đó vẫn phân loại được, model có thể học.
Nếu con người cũng không chắc, ảnh đó dễ làm nhiễu label.
```

## 11. Vai trò của `metaData`

Folder `metaData` chứa tọa độ crop:

```csv
image_name,folder,x1,x2,y1,y2
```

Nó thuộc giai đoạn tạo dữ liệu:

```text
ảnh gốc + tọa độ -> crop biển báo
```

Sau khi đã có `CropData`, `ResizeData`, `RGBData`, `SplitData`, notebook train classification không cần dùng tọa độ nữa.

Nói cách khác:

```text
metaData dùng để chuẩn bị dữ liệu
SplitData dùng để train model
```

## 12. Output của model là gì?

Model trả về vector logits có độ dài bằng số class.

Nếu có 12 class:

```text
output shape = [batch_size, 12]
```

Ví dụ một ảnh có logits:

```text
[1.2, -0.4, 0.8, ..., 3.1]
```

Sau softmax, logits thành xác suất:

```text
no_entry: 0.02
parking: 0.01
stop_sign: 0.94
...
```

Class có xác suất cao nhất là dự đoán cuối.

## 13. Vì sao tên file notebook còn `gtsrb`?

Tên file vẫn là:

```text
mobilenetv2_gtsrb_ver3_datatulam.ipynb
```

nhưng nội dung ver3 hiện đã chỉnh sang custom data. Vì vậy khi báo cáo không nên nói đang dùng GTSRB 43 lớp, mà nên nói:

```text
Dataset tự xây dựng gồm 12 lớp biển báo, đã crop/resize/RGB và chia train/val/test.
```

Nếu muốn sạch hơn, có thể đổi tên file sau này, nhưng không bắt buộc.

## 14. Cách viết trong báo cáo

Có thể viết:

```text
Trong đề tài này, nhóm tập trung vào bài toán phân loại biển báo giao thông từ ảnh đã được crop vùng biển báo. Mô hình không thực hiện phát hiện vị trí biển báo trên ảnh nguyên cảnh. Do đó, trước khi đưa ảnh vào mô hình, vùng chứa biển báo cần được crop thủ công hoặc bằng một bước tiền xử lý khác. Sau khi crop, ảnh được resize về 224x224, chuyển về RGB, chuẩn hóa và đưa vào MobileNetV2 để phân loại.
```

Nếu cần nói rõ giới hạn:

```text
Giới hạn của mô hình là chưa xử lý trực tiếp ảnh đường phố nguyên cảnh. Để áp dụng trong hệ thống thực tế, cần kết hợp thêm một mô hình object detection để tìm vùng biển báo trước, sau đó dùng mô hình MobileNetV2 đã huấn luyện để phân loại loại biển báo.
```

## 15. Cách nói khi thuyết trình

Có thể nói ngắn:

```text
Bài toán của em là classification, không phải detection. Nghĩa là model nhận ảnh biển báo đã được crop rồi dự đoán loại biển báo. Nếu ảnh ngoài là ảnh nguyên cảnh, em cần crop vùng biển báo trước khi đưa vào model. Việc tự động tìm vị trí biển báo là bài toán detection và nằm ngoài phạm vi notebook này.
```

Nếu thầy hỏi “sao không đưa ảnh nguyên tấm?”:

```text
Vì MobileNetV2 trong bài được train như classifier trên ảnh crop. Nếu đưa ảnh nguyên tấm thì input khác phân phối train, background quá nhiều và model không có module định vị biển báo. Muốn xử lý ảnh nguyên tấm cần thêm detector như YOLO hoặc SSD trước classifier.
```

## 16. Sai lầm dễ mắc

### 16.1. Nhầm classification với detection

Sai:

```text
Model MobileNetV2 của em phát hiện biển báo trong ảnh.
```

Đúng:

```text
Model MobileNetV2 của em phân loại ảnh biển báo đã crop.
```

### 16.2. Test bằng ảnh nguyên cảnh rồi kết luận model kém

Nếu test bằng ảnh nguyên cảnh chưa crop, kết quả kém không chứng minh classifier sai. Nó chứng minh pipeline test không đúng với phạm vi bài toán.

### 16.3. Nói crop làm gian lận

Crop không phải gian lận nếu phạm vi bài toán là classification. Crop là bước tiền xử lý xác định input cho classifier.

Chỉ cần trình bày rõ:

```text
Input của mô hình là ảnh crop vùng biển báo.
```

### 16.4. Dùng val/test không crop

Nếu train crop nhưng val/test không crop, kết quả đánh giá sẽ không phản ánh đúng khả năng classifier.

## 17. Liên hệ với notebook ver3

Trong notebook:

- Cell 2 tìm `SplitData`.
- Cell 4 dùng `ImageFolder` đọc `train/val/test`.
- Cell 5 hiển thị ảnh mẫu sau transform.
- Cell 17 cho crop ảnh ngoài bằng chuột trước khi predict.

Những cell này đều phù hợp với phạm vi classification từ ảnh crop.

## 18. Checklist phần phạm vi bài toán

Trước khi train hoặc báo cáo, kiểm tra:

```text
[ ] Có nói rõ bài toán là classification không?
[ ] Có nói rõ input là ảnh đã crop không?
[ ] Có phân biệt với detection không?
[ ] Có nói test ảnh ngoài cần crop không?
[ ] Có nói giới hạn: chưa phát hiện biển báo trong ảnh nguyên cảnh không?
[ ] Có giải thích vai trò metaData là để crop, không phải để train classifier không?
```

## 19. Kết luận phần 1

Phạm vi đúng của notebook là:

```text
Phân loại 12 loại biển báo giao thông từ ảnh đã crop vùng biển báo.
```

Pipeline đúng:

```text
ảnh ngoài -> crop biển báo -> resize/normalize -> MobileNetV2 -> class
```

Pipeline ngoài phạm vi:

```text
ảnh ngoài nguyên cảnh -> MobileNetV2 tự tìm biển báo
```

Nếu muốn pipeline ngoài phạm vi hoạt động, cần thêm mô hình detection trước MobileNetV2.
