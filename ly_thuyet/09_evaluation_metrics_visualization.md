# Phần 9 - Evaluation, Metrics Và Visualization

## 1. Vì sao phải đánh giá mô hình?

Train xong chưa có nghĩa là mô hình dùng được.

Trong bài toán phân loại biển báo, mô hình có thể đạt accuracy rất cao trên tập validation/test nội bộ, nhưng vẫn dự đoán kém trên ảnh ngoài đời nếu:

- train/val/test quá giống nhau;
- ảnh đã được crop quá sạch;
- dữ liệu test cũng lấy từ cùng nguồn ảnh;
- có data leakage do ảnh augment bị chia lẫn vào train và val/test;
- ảnh ngoài đời có góc chụp, ánh sáng, độ mờ, nền, kích thước khác;
- mô hình chỉ học đặc điểm nền hoặc màu tổng quát thay vì học biển báo thật.

Vì vậy evaluation không chỉ là nhìn một con số accuracy. Cần xem nhiều góc:

- loss;
- accuracy;
- confusion matrix;
- accuracy từng class;
- top class tốt nhất và tệ nhất;
- ảnh dự đoán sai;
- độ tự tin của mô hình;
- test ảnh ngoài;
- kiểm tra split data;
- kiểm tra có leakage hay không.

---

## 2. Các tập dữ liệu trong evaluation

Pipeline của mình thường có ba tập:

```text
train -> dùng để học trọng số
val   -> dùng để theo dõi trong lúc train, chọn checkpoint tốt nhất
test  -> dùng để đánh giá cuối cùng sau khi train
```

### 2.1. Train

Train là tập mô hình được phép nhìn trong quá trình học.

Trong mỗi epoch, mô hình đi qua toàn bộ ảnh train một lần.

Nếu train có 8578 ảnh và batch size là 32, số batch mỗi epoch xấp xỉ:

```text
8578 / 32 = 268.06
```

Tức là khoảng 269 batch cho một epoch.

### 2.2. Validation

Validation dùng để xem mô hình có đang học tốt không.

Mô hình không cập nhật trọng số trên val. Nó chỉ forward qua ảnh val, tính loss và accuracy.

Val thường dùng để:

- chọn best checkpoint;
- early stopping;
- phát hiện overfit;
- so sánh model/config.

### 2.3. Test

Test chỉ nên dùng sau cùng.

Nếu cứ chỉnh model theo test nhiều lần, test sẽ không còn là test khách quan nữa. Khi đó test vô tình trở thành validation thứ hai.

Nói khi báo cáo:

> Tập test được giữ riêng, không dùng để cập nhật trọng số hay chọn siêu tham số trong quá trình huấn luyện.

---

## 3. Accuracy là gì?

Accuracy là tỉ lệ dự đoán đúng trên tổng số mẫu.

Công thức:

```text
Accuracy = số mẫu dự đoán đúng / tổng số mẫu
```

Ví dụ:

```text
test có 1835 ảnh
mô hình dự đoán đúng 1828 ảnh

Accuracy = 1828 / 1835 = 0.9962 = 99.62%
```

Accuracy dễ hiểu nhưng có giới hạn.

Nếu dữ liệu mất cân bằng mạnh, accuracy có thể lừa mình.

Ví dụ cực đoan:

```text
Class A: 950 ảnh
Class B: 50 ảnh
```

Nếu mô hình luôn đoán A:

```text
Accuracy = 950 / 1000 = 95%
```

Nhìn rất cao, nhưng mô hình hoàn toàn không biết class B.

Bài của mình có 12 class và cần xem thêm per-class accuracy để tránh lỗi này.

---

## 4. Loss là gì?

Loss đo mức sai của mô hình theo xác suất.

Accuracy chỉ nhìn đúng/sai. Loss nhìn sâu hơn: mô hình đúng với tự tin bao nhiêu, sai với tự tin bao nhiêu.

Ví dụ cùng dự đoán đúng class `stop_sign`:

```text
Mẫu 1:
P(stop_sign) = 0.99

Mẫu 2:
P(stop_sign) = 0.41
```

Cả hai đều đúng nếu `stop_sign` là xác suất cao nhất.

Nhưng mẫu 1 tốt hơn vì mô hình tự tin hơn. Loss của mẫu 1 sẽ thấp hơn.

### 4.1. Vì sao accuracy tăng nhưng loss cũng có thể tăng?

Điều này có thể xảy ra.

Ví dụ:

- nhiều ảnh được dự đoán đúng hơn nên accuracy tăng;
- nhưng một số ảnh sai bị mô hình đoán sai với xác suất rất cao;
- các lỗi sai tự tin cao làm loss tăng.

Loss rất nhạy với xác suất.

Nếu ảnh đúng là class 3 nhưng mô hình dự đoán:

```text
P(class 3) = 0.001
```

CrossEntropy sẽ phạt rất nặng.

Vì vậy:

```text
accuracy tăng không bắt buộc loss luôn giảm
loss giảm không bắt buộc accuracy luôn tăng từng epoch
```

Xu hướng nhiều epoch mới quan trọng hơn một dòng đơn lẻ.

---

## 5. Train loss và val loss đọc thế nào?

### 5.1. Trường hợp tốt

```text
train loss giảm
val loss giảm
train acc tăng
val acc tăng
```

Đây là dấu hiệu mô hình đang học tốt.

### 5.2. Overfit

Dấu hiệu phổ biến:

```text
train loss tiếp tục giảm
train acc tiếp tục tăng
val loss bắt đầu tăng
val acc đứng yên hoặc giảm
```

Nghĩa là mô hình học quá sát train, nhưng không tổng quát tốt sang val.

### 5.3. Underfit

Dấu hiệu:

```text
train acc thấp
val acc thấp
train loss cao
val loss cao
```

Mô hình chưa học đủ hoặc cấu hình chưa phù hợp.

Nguyên nhân có thể:

- learning rate sai;
- model quá nhỏ;
- train quá ít epoch;
- transform sai;
- label sai;
- dữ liệu khó hoặc nhiễu;
- normalize sai;
- ảnh input bị resize/crop hỏng.

### 5.4. Val accuracy cao hơn train accuracy

Điều này không nhất thiết sai.

Trong notebook, train có thể khó hơn val vì:

- train có augmentation;
- train có dropout;
- train dùng batch có nhiễu;
- val không augment;
- val được đánh giá ở chế độ `model.eval()`.

Khi `model.train()`, dropout hoạt động. Khi `model.eval()`, dropout tắt.

Vì vậy val đôi khi cao hơn train.

---

## 6. Vì sao epoch 1 đã accuracy cao?

Nếu epoch 1 đã đạt 78% train accuracy hoặc val 98%, có vài khả năng:

### 6.1. Dữ liệu dễ

Ảnh biển báo đã crop sạch, nền ít gây nhiễu, class khác nhau rõ ràng.

Ví dụ:

- stop sign có hình bát giác đỏ;
- no entry là tròn đỏ có vạch trắng;
- speed limit có số rõ;
- parking có chữ P;
- roadworks có hình công trường.

Với crop sạch 224x224, bài toán phân loại 12 class có thể khá dễ.

### 6.2. Mô hình đủ mạnh

MobileNetV2 tuy nhẹ nhưng vẫn là CNN mạnh.

Nó học được:

- cạnh;
- màu;
- hình tròn/tam giác/bát giác;
- số;
- chữ;
- pattern bên trong biển báo.

### 6.3. Có thể dùng pretrained hoặc không

Cần kiểm tra code tạo model.

Nếu dùng trọng số pretrained ImageNet, epoch 1 cao là rất bình thường.

Nếu không pretrained, epoch 1 vẫn có thể cao nếu:

- dữ liệu dễ;
- class ít;
- ảnh sạch;
- learning rate phù hợp;
- split train/val cùng phân phối.

### 6.4. Có thể có leakage

Nếu ảnh augment từ cùng ảnh gốc bị chia sang cả train và val/test, kết quả cao bất thường.

Ví dụ:

```text
ảnh gốc A nằm train
ảnh augment của A nằm val
```

Khi đó val không thật sự độc lập.

Đây là lý do mình đã kiểm tra và chạy lại split từ `RGBData` để tránh dùng tập đã augment sẵn.

---

## 7. Overfit là gì?

Overfit là khi mô hình học quá sát dữ liệu train, dẫn đến kết quả train tốt nhưng dữ liệu mới kém.

Không phải cứ accuracy cao là overfit.

Overfit cần nhìn quan hệ giữa train, val, test và ảnh ngoài.

### 7.1. Không nên kết luận overfit chỉ vì test cao

Nếu:

```text
train acc = 99%
val acc   = 99%
test acc  = 99%
```

Thì có thể:

- mô hình tốt;
- bài toán dễ;
- dữ liệu test giống train;
- hoặc có leakage.

Chưa đủ để kết luận.

### 7.2. Dấu hiệu overfit rõ hơn

```text
train acc = 99.9%
val acc   = 85%
test acc  = 82%
```

Hoặc:

```text
train loss giảm liên tục
val loss tăng liên tục
```

Hoặc:

```text
test nội bộ 99%
ảnh ngoài đời crop thật chỉ đúng 60%
```

Lúc đó có khả năng overfit hoặc dataset shift.

---

## 8. Confusion matrix là gì?

Confusion matrix là ma trận cho biết class thật và class dự đoán.

Trục thường dùng:

```text
hàng: label thật
cột: label dự đoán
```

Ví dụ đơn giản có 3 class:

```text
                predicted
              A    B    C
true A       50    2    0
true B        1   45    4
true C        0    3   48
```

Đọc:

- 50 ảnh class A đoán đúng A;
- 2 ảnh class A bị đoán nhầm thành B;
- 4 ảnh class B bị đoán nhầm thành C;
- 3 ảnh class C bị đoán nhầm thành B.

Đường chéo chính là dự đoán đúng.

Ngoài đường chéo là nhầm lẫn.

### 8.1. Vì sao confusion matrix quan trọng?

Accuracy tổng thể không nói mô hình nhầm class nào.

Với biển báo, nhầm giữa các class có mức nghiêm trọng khác nhau.

Ví dụ:

- nhầm `speed_limit_50` thành `speed_limit_60` là sai nhưng cùng nhóm tốc độ;
- nhầm `stop_sign` thành `parking` nghiêm trọng hơn;
- nhầm `no_entry` thành `no_stopping` cũng nguy hiểm hơn tùy ứng dụng.

Confusion matrix giúp biết:

- class nào hay nhầm;
- cặp class nào dễ lẫn;
- có class nào quá ít mẫu;
- có class nào label sai;
- có class nào cần thêm dữ liệu.

---

## 9. Per-class accuracy

Per-class accuracy là accuracy tính riêng từng class.

Công thức:

```text
Accuracy(class k) = số ảnh class k đoán đúng / tổng ảnh class k
```

Ví dụ:

```text
Class speed_limit_50:
tổng ảnh test = 110
đoán đúng = 108

Per-class acc = 108 / 110 = 98.18%
```

### 9.1. Top-5 best và Top-5 worst

Trong notebook có phần in:

```text
Top-5 best classes
Top-5 worst classes
```

Mục đích:

- biết class nào mô hình làm tốt nhất;
- biết class nào cần kiểm tra thêm;
- ưu tiên xem ảnh sai ở các class worst.

Nếu worst class vẫn 98%, chưa chắc overfit. Có thể test nội bộ quá sạch hoặc bài toán thật sự dễ.

Nên tiếp tục kiểm tra ảnh ngoài.

---

## 10. Precision, Recall, F1-score

Accuracy chưa đủ nếu dữ liệu lệch class hoặc cần đánh giá nghiêm túc hơn.

### 10.1. Precision

Precision trả lời:

> Trong các ảnh mô hình dự đoán là class A, có bao nhiêu ảnh thật sự là A?

Công thức:

```text
Precision = TP / (TP + FP)
```

Trong đó:

- TP: true positive;
- FP: false positive.

Ví dụ class `stop_sign`:

```text
mô hình dự đoán stop_sign 100 ảnh
trong đó 95 ảnh thật sự là stop_sign

Precision = 95 / 100 = 95%
```

Precision thấp nghĩa là mô hình hay "gán nhầm" ảnh class khác thành class này.

### 10.2. Recall

Recall trả lời:

> Trong tất cả ảnh thật sự là class A, mô hình tìm đúng được bao nhiêu?

Công thức:

```text
Recall = TP / (TP + FN)
```

Trong đó:

- FN: false negative.

Ví dụ:

```text
tập test có 120 ảnh stop_sign
mô hình đoán đúng stop_sign 110 ảnh

Recall = 110 / 120 = 91.67%
```

Recall thấp nghĩa là mô hình bỏ sót class đó nhiều.

### 10.3. F1-score

F1 là trung bình điều hòa giữa precision và recall.

Công thức:

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

F1 hữu ích khi cần cân bằng giữa precision và recall.

### 10.4. Macro và weighted average

Macro average:

```text
tính trung bình đều trên các class
```

Mỗi class có trọng số như nhau, dù class nhiều hay ít ảnh.

Weighted average:

```text
tính trung bình có trọng số theo số ảnh mỗi class
```

Class nhiều ảnh ảnh hưởng nhiều hơn.

Trong báo cáo, nếu dataset khá cân bằng, accuracy và macro F1 thường gần nhau. Nếu dataset lệch, macro F1 quan trọng hơn.

---

## 11. Classification report

Classification report thường gồm:

```text
precision
recall
f1-score
support
```

`support` là số lượng mẫu thật của class đó trong tập test.

Ví dụ:

```text
class             precision   recall   f1-score   support
no_entry            1.00       0.99      0.99       150
stop_sign           0.99       1.00      0.99       120
```

Khi đọc report:

- nhìn class có F1 thấp nhất;
- kiểm tra support có quá ít không;
- so sánh precision và recall;
- xem class nào bị nhầm trong confusion matrix.

Nếu một class support quá ít, accuracy class đó không ổn định.

Ví dụ:

```text
class A có 5 ảnh test
đúng 5 ảnh -> 100%
```

Nhưng 100% này chưa mạnh bằng class có 200 ảnh test và đúng 198 ảnh.

---

## 12. Visualization trong notebook để làm gì?

Notebook thường có các biểu đồ:

- train loss theo epoch;
- val loss theo epoch;
- train accuracy theo epoch;
- val accuracy theo epoch;
- confusion matrix;
- sample predictions;
- ảnh dự đoán sai;
- augmentation example.

Mỗi biểu đồ trả lời một câu hỏi khác nhau.

### 12.1. Loss curve

Loss curve giúp xem mô hình học ổn không.

Tốt:

```text
train loss giảm
val loss giảm hoặc ổn định
```

Cảnh báo:

```text
train loss giảm
val loss tăng
```

### 12.2. Accuracy curve

Accuracy curve giúp xem mô hình đạt hiệu năng thế nào theo epoch.

Nếu accuracy nhảy rất cao từ epoch 1, cần kiểm tra:

- dataset dễ;
- pretrained;
- split có leakage không;
- val/test có ảnh augment cùng gốc không;
- ảnh crop quá sạch không.

### 12.3. Confusion matrix heatmap

Heatmap giúp nhìn nhanh class nào bị nhầm.

Nếu đường chéo chính rất đậm và ngoài đường chéo gần như trắng, nghĩa là mô hình phân biệt tốt trên tập đó.

Nếu có ô ngoài đường chéo đậm, đó là cặp class hay nhầm.

### 12.4. Misclassified images

Ảnh dự đoán sai cực kỳ quan trọng.

Nó giúp trả lời:

- ảnh bị mờ không;
- crop thiếu biển không;
- label có sai không;
- biển bị che không;
- class thật có dễ nhầm không;
- mô hình dự đoán sai nhưng hợp lý không.

Đôi khi lỗi không nằm ở model mà nằm ở dữ liệu.

---

## 13. Độ tự tin của mô hình

Output của model là logits, chưa phải xác suất.

Sau `softmax`, ta có xác suất từng class:

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

Trong đó:

- `z_i` là logit class i;
- `p_i` là xác suất class i.

Mô hình chọn class có xác suất cao nhất:

```text
pred = argmax(p)
```

### 13.1. Top-1 confidence

Top-1 confidence là xác suất của class được chọn.

Ví dụ:

```text
prediction: stop_sign
confidence: 99.2%
```

Không có nghĩa chắc chắn ngoài đời 99.2%. Nó chỉ là mức tự tin nội bộ của mô hình.

Mô hình deep learning có thể quá tự tin khi gặp ảnh lạ.

### 13.2. Top-k predictions

Top-k giúp xem các class gần nhất.

Ví dụ:

```text
1. speed_limit_50: 80%
2. speed_limit_60: 15%
3. speed_limit_40: 3%
```

Nếu top-1 và top-2 gần nhau, mô hình đang phân vân.

Nếu ảnh ngoài đời:

```text
1. no_entry: 40%
2. no_stopping: 38%
```

Thì không nên quá tin kết quả, dù top-1 vẫn có tên class.

---

## 14. Evaluation với dữ liệu đã crop và ảnh ngoài

Dataset hiện tại là bài toán classification trên ảnh crop.

Điều đó nghĩa là test nội bộ cũng nên crop tương tự train.

Nếu đưa ảnh nguyên cảnh rộng vào classifier:

```text
ảnh đường phố lớn -> resize thành 224x224 -> biển báo rất nhỏ
```

Mô hình sẽ không thấy rõ biển.

Kết quả sai là bình thường, không phải vì classifier hỏng.

Muốn xử lý ảnh nguyên cảnh cần thêm bước detection hoặc crop thủ công trước.

---

## 15. Cần crop test như thế nào?

Với test nội bộ:

- crop nên bao quanh biển báo;
- không crop mất phần quan trọng;
- không dư nền quá nhiều;
- kích thước/kiểu crop tương tự train;
- không chỉnh ảnh test bằng augmentation.

Với ảnh ngoài:

- crop biển báo trước;
- để lại một ít viền/nền quanh biển;
- tránh crop sát quá làm mất cạnh biển;
- tránh crop quá rộng làm biển quá nhỏ;
- resize theo pipeline của notebook.

Nếu train toàn ảnh crop sạch nhưng test ngoài crop lệch nhiều, kết quả sẽ giảm.

---

## 16. Evaluation có cần augmentation không?

Không.

Validation/test không nên dùng random augmentation.

Lý do:

- val/test phải ổn định;
- mỗi lần chạy evaluation phải cho kết quả giống nhau;
- val/test dùng để đo khả năng tổng quát, không dùng để tạo thêm dữ liệu học.

Transform cho val/test thường chỉ gồm:

```text
resize nếu cần
ToTensor
Normalize
```

Train transform mới có thể thêm:

```text
random rotation
random affine
color jitter
random perspective
blur nhẹ
```

---

## 17. Dùng checkpoint nào để test?

Nên dùng checkpoint tốt nhất trên validation.

Trong training loop, khi val accuracy tốt hơn best trước đó, notebook lưu best checkpoint.

Logic:

```text
nếu val_acc hiện tại > best_val_acc:
    lưu best_model
```

Khi test, load best checkpoint thay vì dùng model ở epoch cuối.

Vì epoch cuối có thể overfit hơn best epoch.

Ví dụ:

```text
epoch 12: val acc 99.5% -> best
epoch 30: val acc 98.7%
```

Nếu lấy epoch 30 thì kém hơn.

---

## 18. Đánh giá khi class rất cao 100%

Nếu top best hoặc worst đều gần 100%, cần bình tĩnh.

Có ba khả năng:

### 18.1. Kết quả tốt thật

Dataset crop sạch, class rõ, model đủ mạnh.

### 18.2. Test quá giống train

Dù không leakage, ảnh cùng nguồn có thể rất giống.

Ví dụ cùng camera, cùng cách crop, cùng điều kiện ánh sáng.

### 18.3. Có leakage

Ảnh augment hoặc ảnh trùng đã lọt qua split.

Cần kiểm tra:

- split từ ảnh gốc chưa augment;
- hash ảnh trùng giữa train/val/test;
- tên file có pattern augment không;
- ảnh gần giống nhau giữa split không.

---

## 19. Kiểm tra leakage bằng hash ảnh

Một cách nghiêm túc là hash nội dung file ảnh.

Ý tưởng:

```text
hash(file ảnh train)
hash(file ảnh val)
hash(file ảnh test)
```

Nếu cùng hash xuất hiện ở nhiều split, có ảnh trùng y hệt.

Tuy nhiên ảnh augment nhẹ sẽ khác hash dù cùng gốc.

Vì vậy hash chỉ bắt được trùng tuyệt đối, không bắt hết near-duplicate.

### 19.1. Pseudocode

```python
from pathlib import Path
import hashlib

def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

train_hashes = set(md5(p) for p in train_images)
val_hashes = set(md5(p) for p in val_images)

overlap = train_hashes & val_hashes
print(len(overlap))
```

Nếu `overlap > 0`, cần xem lại split.

---

## 20. Kiểm tra near-duplicate

Near-duplicate là ảnh gần giống nhau nhưng không trùng byte.

Ví dụ:

- cùng ảnh gốc nhưng xoay 2 độ;
- tăng sáng nhẹ;
- crop lệch vài pixel;
- blur nhẹ.

Hash MD5 không bắt được.

Có thể dùng perceptual hash như:

- average hash;
- pHash;
- dHash.

Ý tưởng:

```text
ảnh giống nhau -> perceptual hash gần nhau
```

Nếu phát hiện nhiều ảnh gần giống giữa train và val/test, kết quả evaluation sẽ lạc quan quá mức.

---

## 21. Đọc kết quả kiểu nào cho báo cáo?

Không nên viết:

> Mô hình đạt 99% nên rất tốt.

Nên viết chắc hơn:

> Trên tập test đã được crop theo cùng quy trình với tập huấn luyện, mô hình đạt accuracy X%. Kết quả này cho thấy mô hình phân biệt tốt 12 lớp biển báo trong điều kiện dữ liệu nội bộ. Tuy nhiên, do bài toán hiện tại là phân loại ảnh crop, khi áp dụng cho ảnh đường phố nguyên cảnh cần có bước phát hiện hoặc crop biển báo trước khi đưa vào classifier.

Nếu có ảnh ngoài:

> Ngoài đánh giá trên test set, mô hình được kiểm tra thêm bằng một số ảnh ngoài tập dữ liệu. Các ảnh này được crop quanh biển báo trước khi dự đoán nhằm phù hợp với phạm vi bài toán classification.

---

## 22. Khi nào nên dừng train?

Dừng train khi:

- val accuracy không tăng nữa;
- val loss tăng hoặc dao động xấu;
- early stopping kích hoạt;
- mô hình đã đạt mục tiêu;
- train thêm không cải thiện test ngoài.

Không nên train chỉ để đạt 100% train accuracy.

Train accuracy 100% không phải mục tiêu chính.

Mục tiêu là generalization.

---

## 23. Early stopping liên quan evaluation thế nào?

Early stopping theo dõi validation.

Ví dụ `patience = 10`:

```text
nếu 10 epoch liên tiếp không cải thiện val acc
thì dừng train
```

Điều này giúp:

- tiết kiệm thời gian;
- tránh train quá lâu;
- giảm nguy cơ overfit.

Nhưng early stopping chỉ tốt nếu validation đáng tin.

Nếu validation bị leakage, early stopping cũng bị lừa.

---

## 24. Đánh giá sai do preprocessing không khớp

Một lỗi rất thường gặp:

```text
train dùng normalize mean/std A
test ngoài không normalize
```

Hoặc:

```text
train ảnh RGB
test đọc nhầm BGR
```

Hoặc:

```text
train resize 224
test resize khác
```

Khi preprocessing không khớp, model có thể giảm mạnh.

Pipeline test phải giống train ở các bước deterministic:

- chuyển RGB;
- resize nếu bật;
- ToTensor;
- Normalize cùng mean/std;
- shape input `[1, 3, H, W]`.

---

## 25. Những câu hỏi phản biện thường gặp

### 25.1. Tại sao accuracy cao vậy?

Trả lời:

> Vì bài toán hiện tại là phân loại trên ảnh biển báo đã crop, số lớp là 12 và các lớp có đặc trưng hình dạng/màu sắc khá rõ. Ngoài ra train/val/test được lấy cùng nguồn dữ liệu nên phân phối tương đối giống nhau. Vì vậy accuracy cao trên test nội bộ là hợp lý, nhưng để khẳng định khả năng dùng thực tế cần kiểm tra thêm ảnh ngoài hoặc thêm bước detection.

### 25.2. Có overfit không?

Trả lời:

> Không kết luận chỉ từ accuracy cao. Cần so sánh train loss, val loss, train acc, val acc, test acc và kiểm tra ảnh dự đoán sai. Nếu train tiếp tục tốt lên nhưng val/test giảm, đó là dấu hiệu overfit. Nếu train/val/test đều cao và loss ổn định, chưa đủ bằng chứng nói overfit.

### 25.3. Tại sao cần confusion matrix?

Trả lời:

> Accuracy tổng thể chỉ cho biết đúng bao nhiêu phần trăm, còn confusion matrix cho biết mô hình nhầm class nào với class nào. Trong biển báo giao thông, một số nhầm lẫn nguy hiểm hơn các nhầm lẫn khác nên cần xem chi tiết theo class.

### 25.4. Test ảnh nguyên cảnh được không?

Trả lời:

> Với model hiện tại thì không nên đưa trực tiếp ảnh nguyên cảnh vì model là classifier, không phải detector. Cần crop vùng biển báo trước hoặc dùng một mô hình detection để tìm biển báo, sau đó mới phân loại.

---

## 26. Checklist evaluation đúng

Trước khi tin kết quả, kiểm tra:

- train/val/test có split từ dữ liệu gốc chưa augment không;
- val/test không dùng random augmentation;
- số ảnh từng split đúng khoảng 70/15/15 không;
- class mapping train/val/test có giống nhau không;
- model test bằng best checkpoint chưa;
- preprocessing test giống train chưa;
- confusion matrix có class nào nhầm nhiều không;
- có xem ảnh dự đoán sai chưa;
- có thử ảnh ngoài được crop chưa;
- có ghi rõ phạm vi là classification ảnh crop chưa.

---

## 27. Kết luận phần 9

Evaluation là bước trả lời câu hỏi:

> Mô hình học thật không, học tốt class nào, yếu class nào, và kết quả có đáng tin trong phạm vi bài toán không?

Với bài toán hiện tại:

- accuracy cao không tự động là sai;
- accuracy cao cũng không tự động chứng minh dùng tốt ngoài đời;
- cần đọc cùng loss, confusion matrix, per-class accuracy, ảnh sai và test ngoài;
- test nội bộ phải crop giống train;
- ảnh nguyên cảnh cần detection/crop trước.

Điểm quan trọng nhất khi báo cáo là nói đúng phạm vi:

```text
Mô hình MobileNetV2 trong notebook này giải quyết bài toán phân loại 12 loại biển báo từ ảnh đã crop.
```

Không nên trình bày như một hệ thống tự phát hiện biển báo hoàn chỉnh nếu chưa có detector.
