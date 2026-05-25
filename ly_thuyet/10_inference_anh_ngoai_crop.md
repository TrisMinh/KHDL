# Phần 10 - Inference Ảnh Ngoài, Crop Và Giới Hạn Classification

## 1. Inference là gì?

Inference là giai đoạn dùng mô hình đã train để dự đoán ảnh mới.

Trong training:

```text
ảnh + label -> model học trọng số
```

Trong inference:

```text
ảnh mới -> model -> class dự đoán + độ tự tin
```

Ví dụ:

```text
input: ảnh biển báo crop
output: no_entry, confidence 99.3%
```

Inference không cập nhật trọng số. Mô hình chỉ dùng trọng số đã học để dự đoán.

---

## 2. Bài toán hiện tại là classification

Notebook hiện tại giải quyết bài toán:

```text
classification ảnh biển báo đã crop
```

Tức là input phải là ảnh mà biển báo đã nằm rõ trong khung hình.

Model được học từ dữ liệu kiểu:

```text
ảnh 224x224, biển báo nằm lớn ở giữa hoặc gần giữa ảnh
```

Vì vậy khi đưa ảnh mới vào, ảnh mới cũng nên giống điều kiện đó.

---

## 3. Classification khác detection thế nào?

### 3.1. Classification

Classification trả lời:

```text
Ảnh này thuộc class nào?
```

Input:

```text
một ảnh đã chứa đối tượng chính
```

Output:

```text
nhãn class
```

Ví dụ:

```text
ảnh crop một biển báo -> stop_sign
```

### 3.2. Detection

Detection trả lời:

```text
Trong ảnh có đối tượng nào, nằm ở đâu, thuộc class nào?
```

Input:

```text
ảnh nguyên cảnh
```

Output:

```text
bounding box + class + confidence
```

Ví dụ:

```text
x1, y1, x2, y2, stop_sign, 95%
```

### 3.3. Vì sao không đưa ảnh nguyên cảnh vào classifier?

Nếu ảnh ngoài là ảnh đường phố lớn:

```text
ảnh gốc: 1920x1080
biển báo chỉ chiếm 80x80 pixel
```

Khi resize về 224x224:

```text
biển báo bị thu rất nhỏ
nền đường, cây, xe, trời chiếm phần lớn ảnh
```

Classifier được train để nhìn biển báo lớn, nên nó có thể dự đoán sai.

Nói đơn giản:

```text
classifier không tự biết biển báo nằm ở đâu
```

Nó chỉ nhìn toàn bộ ảnh và chọn class.

---

## 4. Vì sao phải crop ảnh ngoài?

Vì training data là ảnh crop.

Mô hình học phân phối dữ liệu:

```text
biển báo lớn
nền ít
đối tượng chính rõ
```

Nếu inference dùng ảnh nguyên cảnh:

```text
biển báo nhỏ
nền nhiều
nhiều vật thể gây nhiễu
```

Đó là phân phối khác.

Hiện tượng này gọi là distribution shift.

Model học trong một kiểu dữ liệu, nhưng test trên kiểu dữ liệu khác, kết quả giảm là bình thường.

---

## 5. Crop đúng là crop như thế nào?

Crop đúng không có nghĩa là cắt sát mép biển báo tuyệt đối.

Crop tốt nên:

- chứa toàn bộ biển báo;
- không mất cạnh hoặc mất ký hiệu quan trọng;
- để lại một ít nền xung quanh;
- không để nền chiếm quá nhiều;
- giữ hình dạng biển báo không bị méo;
- gần giống kiểu ảnh train.

### 5.1. Crop quá sát

Ví dụ crop mất viền đỏ của biển:

```text
biển no_entry nhưng mất một phần vòng tròn đỏ
```

Mô hình có thể mất đặc trưng hình dạng.

### 5.2. Crop quá rộng

Ví dụ biển chỉ chiếm 10% ảnh:

```text
nền chiếm 90%
```

Sau resize, biển báo rất nhỏ. Mô hình khó nhận ra.

### 5.3. Crop lệch nhưng vẫn đủ biển

Nếu biển không nằm đúng giữa nhưng vẫn lớn và rõ, thường vẫn ổn hơn ảnh nguyên cảnh.

Augmentation nếu có random affine/perspective có thể giúp mô hình chịu được crop lệch nhẹ.

---

## 6. Test nội bộ có cần crop không?

Có, nếu train là ảnh crop.

Tập test nên cùng dạng với train:

```text
train: ảnh biển báo crop
test: ảnh biển báo crop
```

Nếu test dùng ảnh nguyên cảnh còn train dùng crop, bài toán đã thay đổi.

Lúc đó kết quả test không đánh giá đúng classifier nữa, mà đang đánh giá cả khả năng "tự tìm biển báo", trong khi model không được thiết kế cho việc đó.

---

## 7. Train có cần ảnh bị che, mất góc, lệch sáng không?

Tùy mục tiêu.

Nếu muốn mô hình dùng tốt hơn ngoài đời, train nên có một mức đa dạng:

- sáng/tối khác nhau;
- mờ nhẹ;
- xoay nhẹ;
- phối cảnh nhẹ;
- crop lệch nhẹ;
- nền khác nhau;
- biển bị che một phần nhỏ nếu ngoài đời có trường hợp đó.

Nhưng không nên augment quá mạnh làm sai bản chất class.

Ví dụ với biển tốc độ:

- làm mờ quá mạnh khiến số 50 không đọc được;
- crop mất số;
- xoay quá lớn không giống thực tế;
- đổi màu quá mạnh làm mất màu biển.

Augment phải mô phỏng nhiễu thật có khả năng gặp.

---

## 8. Pipeline inference chuẩn

Một pipeline inference cho ảnh crop nên như sau:

```text
1. Đọc ảnh
2. Chuyển sang RGB
3. Crop nếu ảnh chưa crop
4. Resize nếu config bật resize
5. ToTensor
6. Normalize bằng mean/std đã dùng khi train
7. Thêm batch dimension
8. model.eval()
9. torch.no_grad()
10. forward
11. softmax
12. lấy top-k prediction
```

---

## 9. Vì sao phải `model.eval()`?

Trong PyTorch, model có hai chế độ:

```python
model.train()
model.eval()
```

Khi train:

- dropout hoạt động;
- batch normalization dùng thống kê batch hiện tại.

Khi eval:

- dropout tắt;
- batch normalization dùng running mean/variance đã học.

Nếu quên `model.eval()` khi inference, kết quả có thể không ổn định.

---

## 10. Vì sao cần `torch.no_grad()`?

Trong inference, không cần tính gradient.

Dùng:

```python
with torch.no_grad():
    output = model(image)
```

Lợi ích:

- tiết kiệm VRAM;
- chạy nhanh hơn;
- tránh lưu computation graph;
- đúng mục đích inference.

---

## 11. Vì sao phải normalize giống train?

Trong train, ảnh sau `ToTensor()` có giá trị khoảng:

```text
[0, 1]
```

Sau normalize:

```text
(x - mean) / std
```

Model học trọng số trên input đã normalize.

Nếu inference không normalize, phân phối input khác hẳn.

Ví dụ model đã quen pixel đỏ sau normalize nằm quanh một khoảng nào đó. Nếu đưa raw tensor `[0, 1]` vào, activation các layer sẽ khác.

Kết quả có thể sai nhiều.

---

## 12. RGB và BGR

Đây là lỗi rất phổ biến.

PIL đọc ảnh theo RGB.

OpenCV đọc ảnh theo BGR.

Nếu train dùng PIL/RGB mà inference dùng OpenCV/BGR không chuyển lại, màu bị đảo.

Ví dụ:

```text
đỏ có thể bị hiểu thành xanh
```

Với biển báo, màu đỏ/xanh/vàng rất quan trọng. Sai kênh màu có thể làm model hỏng.

Nếu dùng OpenCV:

```python
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

Nếu dùng PIL:

```python
img = Image.open(path).convert("RGB")
```

---

## 13. Resize trong inference

Notebook có config:

```python
'resize_enabled': 0 hoặc 1
'resize_size': 224
```

Nếu ảnh trong dataset đã 224x224, có thể để `resize_enabled = 0`.

Nhưng khi inference ảnh ngoài, ảnh crop có thể không đúng 224x224.

Có hai cách:

### 13.1. Dùng resize trong transform inference

Cách này đơn giản:

```text
ảnh crop bất kỳ -> resize về 224x224
```

Nhược điểm: nếu ảnh không vuông, resize trực tiếp có thể làm méo biển.

### 13.2. Pad rồi resize

Nếu muốn giữ tỉ lệ tốt hơn:

```text
ảnh crop -> pad thành hình vuông -> resize 224x224
```

Cách này giảm méo hình.

Với biển báo, hình dạng tròn/tam giác/bát giác quan trọng, nên méo hình có thể ảnh hưởng.

---

## 14. Crop thủ công khi demo

Nếu làm demo đơn giản, có thể:

1. Chụp hoặc tải ảnh ngoài.
2. Crop vùng biển báo bằng công cụ chỉnh ảnh.
3. Lưu ảnh crop.
4. Chạy cell inference trong notebook.

Cách này phù hợp với phạm vi bài classification.

Trong báo cáo nên nói rõ:

> Ảnh ngoài được crop thủ công quanh vùng biển báo trước khi đưa vào mô hình phân loại.

---

## 15. Crop tự động bằng detection

Nếu muốn hệ thống đầy đủ hơn:

```text
ảnh nguyên cảnh -> detector -> crop biển báo -> classifier -> class
```

Detector có thể là:

- YOLO;
- Faster R-CNN;
- SSD;
- RetinaNet.

Pipeline:

```text
1. Detector tìm bounding box biển báo.
2. Cắt từng box.
3. Resize/normalize crop.
4. MobileNetV2 phân loại từng crop.
5. Vẽ box + nhãn lên ảnh gốc.
```

Nhưng đây là bài toán lớn hơn.

Notebook hiện tại chưa train detector.

---

## 16. Có thể dùng MobileNetV2 để detection không?

MobileNetV2 thường là backbone được dùng trong nhiều hệ detection nhẹ.

Nhưng bản notebook hiện tại là classification head:

```text
features -> pooling -> linear classifier
```

Nó không xuất bounding box.

Muốn detection cần kiến trúc khác:

```text
backbone -> detection head -> box regression + class prediction
```

Vì vậy không thể nói notebook hiện tại tự detect biển báo.

---

## 17. Input shape khi inference

PyTorch model thường nhận input:

```text
[batch_size, channels, height, width]
```

Một ảnh RGB sau transform có shape:

```text
[3, 224, 224]
```

Cần thêm batch dimension:

```python
x = x.unsqueeze(0)
```

Shape thành:

```text
[1, 3, 224, 224]
```

Nếu quên `unsqueeze(0)`, model có thể báo lỗi shape.

---

## 18. Output shape khi inference

Nếu có 12 class, output model có shape:

```text
[1, 12]
```

Đó là logits cho từng class.

Ví dụ:

```text
[-1.2, 0.5, 3.1, ...]
```

Sau softmax:

```text
[0.01, 0.04, 0.82, ...]
```

Class dự đoán:

```python
pred_idx = probs.argmax(dim=1).item()
```

Tên class:

```python
class_name = class_names[pred_idx]
```

---

## 19. Mapping class phải đúng

Khi dùng `ImageFolder`, class index được tạo theo thứ tự tên folder.

Ví dụ:

```text
0 -> no_entry
1 -> no_stopping
2 -> no_vehicles
...
```

Inference phải dùng đúng mapping lúc train.

Nếu mapping lệch, model vẫn xuất index đúng theo lúc train nhưng mình đọc sai tên class.

Ví dụ model dự đoán index 0, lúc train index 0 là `no_entry`, nhưng code inference đọc index 0 là `parking`, kết quả hiển thị sai.

Nên lưu `class_to_idx` hoặc `class_names` trong checkpoint.

---

## 20. Confidence cao nhưng sai

Model có thể tự tin cao dù sai.

Ví dụ ảnh ngoài không thuộc 12 class:

```text
một biển báo lạ
```

Classifier vẫn buộc phải chọn một trong 12 class.

Nó có thể trả:

```text
speed_limit_50: 97%
```

Không có nghĩa ảnh thật là speed_limit_50.

Classifier closed-set chỉ biết 12 class, không có class "unknown" nếu mình không train.

Muốn nhận biết ảnh không thuộc class nào, cần thêm:

- class unknown;
- threshold confidence;
- open-set recognition;
- kiểm tra entropy;
- hoặc detector/filter riêng.

---

## 21. Có nên đặt ngưỡng confidence không?

Có thể.

Ví dụ:

```text
nếu confidence < 70%:
    báo "không chắc"
```

Nhưng ngưỡng này cần thử nghiệm.

Nếu đặt quá cao:

- bỏ qua nhiều ảnh đúng nhưng confidence thấp.

Nếu đặt quá thấp:

- chấp nhận nhiều dự đoán sai.

Ngưỡng tốt nên được chọn dựa trên validation/test và ảnh ngoài.

---

## 22. Test ảnh ngoài cần ghi kết quả thế nào?

Nên tạo bảng:

```text
Ảnh | Label thật | Dự đoán | Confidence | Đúng/Sai | Ghi chú
```

Ví dụ:

```text
outside_01.jpg | no_entry | no_entry | 99.1% | đúng | crop rõ
outside_02.jpg | speed_limit_50 | speed_limit_60 | 61.4% | sai | ảnh mờ, số nhỏ
outside_03.jpg | stop_sign | stop_sign | 94.8% | đúng | lệch sáng
```

Ghi chú giúp giải thích lỗi.

---

## 23. Các kiểu ảnh ngoài nên thử

Để đánh giá thực tế hơn, nên thử:

- ảnh crop đẹp giống dataset;
- ảnh hơi tối;
- ảnh hơi sáng;
- ảnh hơi mờ;
- ảnh crop lệch;
- ảnh có nền nhiều hơn;
- ảnh biển hơi nghiêng;
- ảnh biển nhỏ hơn;
- ảnh khác nguồn trên internet;
- ảnh chụp bằng điện thoại.

Không nên chỉ chọn ảnh dễ.

Nếu chỉ test ảnh quá đẹp, kết quả ngoài vẫn có thể lạc quan.

---

## 24. Khi ảnh ngoài dự đoán sai thì làm gì?

Không nên vội sửa model ngay.

Đầu tiên kiểm tra:

- ảnh đã crop đúng chưa;
- có resize/normalize giống train chưa;
- ảnh có đúng RGB không;
- class đó có trong 12 class không;
- checkpoint load đúng không;
- mapping class đúng không;
- ảnh có quá mờ không;
- biển có bị che/mất số không.

Nếu pipeline đúng mà vẫn sai nhiều:

- thêm dữ liệu giống ảnh ngoài;
- bật augmentation phù hợp;
- fine-tune thêm;
- dùng model khác;
- thêm detection/crop tự động nếu ảnh nguyên cảnh.

---

## 25. Inference không nên dùng augmentation random

Không nên dùng random augmentation trong inference thường.

Vì cùng một ảnh có thể ra kết quả khác nhau mỗi lần chạy.

Nếu muốn dùng test-time augmentation, đó là kỹ thuật riêng:

```text
tạo nhiều biến thể nhẹ của ảnh
dự đoán từng biến thể
lấy trung bình xác suất
```

Nhưng với bài hiện tại, inference chuẩn nên deterministic:

```text
resize -> tensor -> normalize -> predict
```

---

## 26. Test-time augmentation là gì?

Test-time augmentation, viết tắt TTA, là dùng augmentation ở lúc test/inference.

Ví dụ:

```text
ảnh gốc
ảnh sáng hơn nhẹ
ảnh tối hơn nhẹ
ảnh crop lệch nhẹ
```

Model dự đoán nhiều lần rồi lấy trung bình xác suất.

Lợi ích:

- đôi khi tăng độ ổn định.

Nhược điểm:

- chậm hơn;
- phức tạp hơn;
- nếu augment không hợp lý có thể làm sai.

Notebook hiện tại không cần TTA để bắt đầu.

---

## 27. Demo pipeline nên nói thế nào?

Nếu thầy hỏi:

> Ảnh ngoài đời nền rộng thì sao?

Trả lời:

> Mô hình hiện tại là mô hình phân loại, nên giả định input là vùng ảnh đã chứa biển báo. Với ảnh nguyên cảnh, cần bước phát hiện hoặc crop biển báo trước. Trong demo hiện tại em crop vùng biển báo rồi đưa vào MobileNetV2 để phân loại. Nếu phát triển tiếp, em sẽ ghép thêm detector như YOLO để tự tìm bounding box trước khi phân loại.

Đây là câu trả lời đúng phạm vi, không phóng đại.

---

## 28. Sai lầm cần tránh khi demo

Không nên:

- đưa ảnh nguyên cảnh rộng vào rồi kết luận model kém;
- quên normalize khi test ngoài;
- dùng sai class mapping;
- test ảnh không thuộc 12 class rồi tin confidence;
- crop mất biển;
- resize làm méo quá mạnh;
- nói model "detect biển báo" khi chưa có detection.

---

## 29. Checklist inference ảnh ngoài

Trước khi dự đoán ảnh ngoài:

- ảnh có chứa đúng một biển báo chính không;
- biển báo thuộc 12 class đã train không;
- đã crop quanh biển báo chưa;
- crop có mất cạnh/mất số không;
- ảnh đã chuyển RGB chưa;
- resize có đúng config không;
- normalize dùng đúng mean/std không;
- model đã load best checkpoint chưa;
- model đang `eval()` chưa;
- inference có `torch.no_grad()` chưa;
- class mapping có đúng không;
- có in top-k confidence không.

---

## 30. Kết luận phần 10

Notebook hiện tại dùng MobileNetV2 để phân loại ảnh biển báo đã crop.

Vì vậy với ảnh ngoài:

```text
ảnh nguyên cảnh -> cần crop/detect trước
ảnh crop biển báo -> đưa vào classifier
```

Nếu muốn hệ thống hoàn chỉnh hơn:

```text
detector tìm biển báo -> crop -> MobileNetV2 phân loại
```

Điểm quan trọng khi báo cáo:

```text
Đây là bài toán classification, không phải detection.
```

Nói đúng phạm vi sẽ làm bài chắc hơn, vì mình không hứa một khả năng mà notebook chưa làm.
