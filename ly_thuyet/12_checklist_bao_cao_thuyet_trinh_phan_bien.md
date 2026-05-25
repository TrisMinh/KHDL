# Phần 12 - Checklist Báo Cáo, Thuyết Trình Và Phản Biện

## 1. Mục tiêu phần này

Phần này giúp chuẩn bị để:

- hiểu toàn bộ notebook;
- trình bày đúng phạm vi bài toán;
- giải thích từng bước pipeline;
- trả lời câu hỏi của thầy;
- tránh nói quá khả năng của mô hình;
- biết điểm yếu và hướng phát triển.

Nếu cần ôn nhanh trước khi thuyết trình, đọc phần này sau khi đã đọc các phần 1 đến 11.

---

## 2. Một câu mô tả đúng bài làm

Câu nên dùng:

> Đề tài xây dựng mô hình MobileNetV2 để phân loại 12 loại biển báo giao thông từ ảnh đã được crop, sử dụng pipeline tiền xử lý, augmentation tùy chọn, huấn luyện bằng CrossEntropy Loss và SGD, sau đó đánh giá bằng accuracy, loss, confusion matrix và per-class metrics.

Câu này đúng vì có đủ:

- model: MobileNetV2;
- bài toán: classification;
- dữ liệu: ảnh crop;
- số class: 12;
- train: loss/optimizer;
- evaluation: metrics.

---

## 3. Câu không nên nói

Không nên nói:

> Mô hình của em phát hiện biển báo trong ảnh đường phố.

Vì notebook hiện tại chưa có detector.

Không nên nói:

> Accuracy 99% chứng minh mô hình chắc chắn dùng tốt ngoài thực tế.

Vì test nội bộ có thể giống train và ảnh đã crop sạch.

Không nên nói:

> Augmentation tạo thêm ảnh thật trong folder train.

Vì notebook dùng online augmentation, không lưu thêm ảnh ra disk.

Không nên nói:

> Normalize chỉ là chia 255.

Vì `ToTensor()` mới là bước đưa về `[0, 1]`, còn `Normalize(mean, std)` là z-score theo kênh.

---

## 4. Dàn ý thuyết trình đề xuất

Một dàn ý gọn nhưng đủ sâu:

1. Giới thiệu bài toán.
2. Phạm vi: classification ảnh crop, chưa phải detection.
3. Dữ liệu và cấu trúc folder.
4. Split train/val/test 70/15/15.
5. Tiền xử lý ảnh: RGB, resize tùy chọn, ToTensor, Normalize.
6. Augmentation cho train.
7. Kiến trúc MobileNetV2.
8. Loss, optimizer, scheduler, warmup.
9. Training loop, checkpoint, early stopping.
10. Evaluation: accuracy, loss, confusion matrix, per-class.
11. Test ảnh ngoài và yêu cầu crop.
12. Hạn chế và hướng phát triển.

---

## 5. Phần giới thiệu bài toán

Nên trình bày:

```text
Input: ảnh biển báo đã crop
Output: một trong 12 lớp biển báo
```

Ví dụ class:

- no_entry;
- no_stopping;
- no_vehicles;
- parking;
- priority_road;
- roadworks;
- roundabout;
- speed_limit_30;
- speed_limit_40;
- speed_limit_50;
- speed_limit_60;
- stop_sign.

Giải thích:

> Do dữ liệu huấn luyện là ảnh biển báo đã được crop, mô hình học nhiệm vụ phân loại đối tượng chính trong ảnh. Nếu dùng ảnh nguyên cảnh, cần thêm bước phát hiện hoặc crop trước.

---

## 6. Phần dữ liệu

Nói rõ folder:

```text
SplitData/
  train/
  val/
  test/
```

Mỗi split có folder con theo class.

Với `ImageFolder`, label được lấy từ tên folder:

```text
SplitData/train/no_entry/*.jpg -> class no_entry
```

Sau khi split lại từ dữ liệu gốc, số lượng:

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

Điểm quan trọng:

> Split phải thực hiện trên dữ liệu gốc, tránh để ảnh augment từ cùng một ảnh gốc rơi vào nhiều split gây leakage.

---

## 7. Phần load data từ Drive

Nói ngắn:

> Dữ liệu được nén trên Google Drive. Khi chạy Colab, file nén được copy về `/content` rồi giải nén để train nhanh hơn, vì đọc nhiều ảnh nhỏ trực tiếp từ Drive thường chậm.

Giải thích sâu hơn nếu bị hỏi:

```text
Drive mount tiện lưu trữ nhưng I/O nhiều file nhỏ chậm.
/content là ổ local tạm thời của Colab, đọc nhanh hơn nhưng mất khi runtime tắt.
```

Vì vậy:

```text
Drive dùng để lưu file nén và checkpoint quan trọng.
/content dùng để train trong session hiện tại.
```

---

## 8. Phần resize

Config có:

```python
'resize_enabled': 0 hoặc 1
'resize_size': 224
```

Giải thích:

> Vì ảnh trong dataset đã được resize sẵn, em để resize thành tùy chọn. Nếu ảnh đầu vào chưa đồng nhất kích thước, bật resize để đưa về kích thước model yêu cầu.

Nếu thầy hỏi vì sao không luôn resize:

> Resize lại ảnh đã đúng kích thước có thể không cần thiết. Để config giúp kiểm soát rõ khi dữ liệu đã chuẩn hoặc khi dùng ảnh ngoài.

---

## 9. Phần normalize

Cần nói đúng:

```text
ToTensor: đưa pixel từ [0, 255] về [0, 1]
Normalize: chuẩn hóa z-score theo từng kênh màu
```

Công thức:

```text
x' = (x - mean) / std
```

Ý nghĩa:

- đưa dữ liệu về phân phối ổn định hơn;
- giúp optimizer học dễ hơn;
- giảm ảnh hưởng do thang giá trị pixel;
- phù hợp với input distribution mà model được train.

Nếu dùng pretrained ImageNet:

> Nên dùng mean/std ImageNet vì trọng số pretrained đã được học với chuẩn đó.

Nếu train from scratch:

> Có thể dùng mean/std tính từ chính train set.

---

## 10. Phần augmentation

Nói:

> Augmentation chỉ áp dụng cho train set để tạo các biến đổi ngẫu nhiên trong lúc load ảnh, giúp mô hình bớt phụ thuộc vào một kiểu ảnh cố định.

Nhấn mạnh:

```text
val/test không dùng random augmentation
online augmentation không làm tăng số file ảnh trên disk
```

Các kiểu augment có thể gồm:

- xoay nhẹ;
- affine/perspective nhẹ;
- thay đổi sáng/tương phản;
- blur nhẹ;
- crop/scale nhẹ nếu phù hợp.

Không nên augment quá mạnh:

- mất số trên biển tốc độ;
- mất viền/hình dạng;
- đổi màu sai bản chất;
- xoay quá không giống thực tế.

---

## 11. Phần MobileNetV2

Câu mô tả:

> MobileNetV2 là kiến trúc CNN nhẹ, dùng depthwise separable convolution và inverted residual block để giảm số tham số và chi phí tính toán nhưng vẫn giữ khả năng trích xuất đặc trưng ảnh.

Ba ý chính phải nhớ:

### 11.1. Depthwise separable convolution

Thay vì convolution thường học không gian và trộn kênh cùng lúc, nó tách thành:

```text
depthwise conv: lọc không gian riêng từng kênh
pointwise conv 1x1: trộn thông tin giữa các kênh
```

Lợi ích:

```text
giảm tham số và FLOPs
```

### 11.2. Inverted residual

Block MobileNetV2 thường:

```text
expand -> depthwise -> project
```

Nó mở rộng số kênh ở giữa để học đặc trưng rồi nén lại.

### 11.3. Linear bottleneck

Layer cuối block không dùng ReLU mạnh sau projection để tránh mất thông tin trong không gian chiều thấp.

---

## 12. Phần stride

Nói:

> Stride quyết định feature map có bị giảm kích thước không. Với ảnh 224x224, dùng stride 2 ở layer đầu giúp giảm kích thước feature map từ 224x224 xuống 112x112, tiết kiệm VRAM đáng kể.

Công thức diện tích:

```text
224 * 224 = 50176
112 * 112 = 12544
```

Giảm 4 lần số vị trí không gian.

Nếu thầy hỏi vì sao ảnh hưởng memory:

> Khi train, PyTorch phải lưu activation để backpropagation. Feature map càng lớn thì activation càng tốn VRAM.

---

## 13. Phần loss

Loss chính:

```text
CrossEntropyLoss
```

Nó kết hợp:

```text
LogSoftmax + Negative Log Likelihood
```

Ý nghĩa:

- nếu mô hình gán xác suất cao cho class đúng, loss thấp;
- nếu gán xác suất thấp cho class đúng, loss cao;
- sai với độ tự tin cao bị phạt nặng.

Nếu có label smoothing:

> Label smoothing không để target là one-hot tuyệt đối, mà phân một phần xác suất nhỏ sang class khác để giảm overconfidence.

---

## 14. Phần optimizer

Optimizer có thể là SGD momentum/Nesterov.

Nói:

> Optimizer cập nhật trọng số dựa trên gradient để giảm loss. Momentum giúp hướng cập nhật ổn định hơn bằng cách tích lũy xu hướng gradient qua các bước.

Các thông số:

- `lr`: bước học;
- `momentum`: quán tính cập nhật;
- `weight_decay`: regularization giảm overfit;
- `grad_clip`: giới hạn gradient quá lớn.

---

## 15. Phần warmup và scheduler

Warmup:

```text
tăng learning rate từ nhỏ lên lr chính trong vài epoch đầu
```

Lý do:

- đầu training trọng số chưa ổn định;
- lr quá lớn ngay từ đầu có thể làm update mạnh;
- warmup giúp khởi động mềm hơn.

Cosine scheduler:

```text
giảm learning rate dần theo dạng cosine
```

Ý nghĩa:

- đầu train học nhanh;
- cuối train tinh chỉnh nhỏ hơn.

Nếu model epoch 1 đã cao:

> Warmup vẫn không sai. Nó chỉ điều khiển learning rate, không bắt buộc mô hình phải học chậm. Nếu dữ liệu dễ, model vẫn có thể đạt accuracy cao ngay epoch đầu.

---

## 16. Phần training loop

Một epoch gồm:

```text
1. model.train()
2. duyệt từng batch train
3. forward
4. tính loss
5. backward
6. optimizer.step()
7. validate trên val set
8. scheduler cập nhật lr
9. lưu checkpoint nếu tốt hơn
```

Mỗi epoch đi qua toàn bộ train set một lần.

Nếu train có 8578 ảnh:

```text
1 epoch = mô hình đã được train qua 8578 ảnh đó
```

Không phải mỗi epoch chỉ train một phần rất nhỏ, trừ khi code cố ý giới hạn.

---

## 17. Phần checkpoint

Checkpoint lưu:

- trọng số model;
- optimizer state;
- scheduler state;
- epoch;
- best val acc;
- class names/config nếu có.

Best checkpoint:

```text
lưu model có val tốt nhất
```

Khi test cuối:

```text
nên load best checkpoint, không nhất thiết dùng epoch cuối
```

Nếu dừng train giữa chừng:

> Có thể chạy cell sau nếu best checkpoint trước đó đã được lưu.

---

## 18. Phần evaluation

Nói:

> Em đánh giá bằng accuracy/loss tổng thể, confusion matrix và per-class accuracy để biết mô hình nhầm class nào.

Accuracy:

```text
số dự đoán đúng / tổng số ảnh
```

Confusion matrix:

```text
hàng là class thật, cột là class dự đoán
```

Per-class:

```text
đánh giá riêng từng class
```

Không nên chỉ nhìn accuracy tổng.

---

## 19. Phần overfit

Câu trả lời chắc:

> Không kết luận overfit chỉ vì accuracy cao. Overfit cần nhìn train loss, val loss, train acc, val acc, test acc và kiểm tra ảnh ngoài. Nếu train tiếp tục tăng nhưng val/test giảm hoặc val loss tăng rõ, đó mới là dấu hiệu overfit.

Dấu hiệu overfit:

```text
train acc cao
val acc thấp hơn nhiều hoặc giảm
train loss giảm
val loss tăng
```

Nếu train/val/test đều cao:

- có thể model tốt;
- có thể dataset dễ;
- có thể test quá giống train;
- cần kiểm tra leakage và ảnh ngoài.

---

## 20. Phần ảnh ngoài

Nói rõ:

> Vì model là classifier nên ảnh ngoài cần được crop quanh biển báo trước khi dự đoán. Nếu đưa ảnh nguyên cảnh, biển báo có thể quá nhỏ và nền chiếm nhiều, model không được thiết kế để tự tìm vị trí biển báo.

Pipeline ảnh ngoài:

```text
ảnh ngoài -> crop biển báo -> resize/normalize -> MobileNetV2 -> class
```

Hướng phát triển:

```text
ảnh nguyên cảnh -> detector -> crop -> classifier
```

---

## 21. Các câu hỏi phản biện và trả lời

### 21.1. Vì sao dùng MobileNetV2?

> Vì MobileNetV2 nhẹ, phù hợp bài toán phân loại ảnh với chi phí tính toán thấp. Nó dùng depthwise separable convolution để giảm tham số và inverted residual block để giữ khả năng biểu diễn đặc trưng.

### 21.2. Tại sao không dùng ResNet?

> ResNet cũng là lựa chọn tốt, nhưng thường nặng hơn. MobileNetV2 phù hợp khi muốn mô hình nhẹ hơn, dễ chạy trên GPU hạn chế hoặc thiết bị tài nguyên thấp.

### 21.3. Attention/Transformer có dùng không?

> Transformer và attention là hướng kiến trúc khác, mạnh trong việc mô hình hóa quan hệ toàn cục. Tuy nhiên notebook hiện tại dùng CNN MobileNetV2. Attention không chỉ là một "mẹo cải thiện" gắn tùy tiện vào mọi model, mà thường cần thiết kế kiến trúc phù hợp. Có thể xem là hướng mở rộng sau.

### 21.4. Vì sao không train detector?

> Vì phạm vi hiện tại là classification ảnh đã crop. Detection cần dữ liệu có bounding box và kiến trúc khác như YOLO/Faster R-CNN. Nếu mở rộng hệ thống cho ảnh nguyên cảnh, em sẽ bổ sung detector trước classifier.

### 21.5. Accuracy cao có đáng tin không?

> Đáng tin trong phạm vi test set nội bộ nếu split đúng và không leakage. Nhưng để kết luận thực tế cần test thêm ảnh ngoài, đặc biệt ảnh khác điều kiện chụp.

### 21.6. Vì sao val cao hơn train?

> Train có augmentation/dropout và model ở chế độ train, còn val không augmentation và model ở eval mode. Vì vậy val đôi khi cao hơn train, nhất là khi dữ liệu val sạch hơn.

### 21.7. Vì sao loss tăng nhưng accuracy vẫn cao?

> Accuracy chỉ đo đúng/sai, còn loss đo cả xác suất. Một vài mẫu sai với confidence rất cao có thể làm loss tăng dù accuracy tổng vẫn tăng.

### 21.8. Vì sao phải normalize?

> Normalize đưa dữ liệu về phân phối ổn định hơn theo từng kênh màu, giúp quá trình tối ưu dễ hơn và khớp với input distribution mà model được train.

### 21.9. Augmentation có làm tăng số ảnh không?

> Với online augmentation thì không tăng số file ảnh. Nó tạo biến đổi ngẫu nhiên khi load ảnh trong từng epoch.

### 21.10. Tại sao train lâu từ Drive?

> Vì đọc nhiều file ảnh nhỏ từ Drive mount chậm. Copy file nén về `/content` rồi giải nén local giúp đọc dữ liệu nhanh hơn.

---

## 22. Checklist trước khi chạy train

Kiểm tra:

- file nén data đã nằm trên Drive;
- đường dẫn `DRIVE_ARCHIVE_PATH` đúng;
- giải nén ra đúng `SplitData`;
- `train/val/test` tồn tại;
- mỗi split có folder class;
- số class đúng 12;
- số ảnh split khoảng 70/15/15;
- `batch_size` phù hợp GPU;
- `img_size` đúng;
- `resize_enabled` đúng với dữ liệu;
- `augment_enabled` đúng mục đích;
- `num_classes` được cập nhật theo dataset;
- checkpoint cũ có cần xóa hoặc resume không.

---

## 23. Checklist trong lúc train

Theo dõi:

- train loss;
- train acc;
- val loss;
- val acc;
- learning rate;
- thời gian mỗi epoch;
- trạng thái best checkpoint;
- dấu hiệu OOM;
- dấu hiệu overfit.

Nếu OOM:

```text
giảm batch_size -> restart runtime -> chạy lại
```

Nếu val không tăng lâu:

```text
early stopping hoặc xem lại lr/augment/data
```

---

## 24. Checklist sau khi train

Làm:

- load best checkpoint;
- chạy test set;
- xem confusion matrix;
- xem per-class accuracy;
- xem ảnh sai;
- test vài ảnh ngoài đã crop;
- ghi lại config cuối;
- lưu kết quả/charts/checkpoint.

Không nên chỉ chụp mỗi dòng accuracy.

---

## 25. Checklist báo cáo kết quả

Nên có:

- mô tả dataset;
- số class;
- số ảnh train/val/test;
- mô tả preprocessing;
- mô tả augmentation;
- kiến trúc model;
- thông số train;
- biểu đồ loss/accuracy;
- confusion matrix;
- bảng per-class;
- ví dụ dự đoán đúng/sai;
- hạn chế;
- hướng phát triển.

---

## 26. Hạn chế nên tự nói trước

Nên chủ động nêu:

1. Mô hình hiện tại chỉ classification ảnh crop.
2. Chưa tự phát hiện vị trí biển báo trong ảnh nguyên cảnh.
3. Test nội bộ có thể cùng phân phối với train nên chưa phản ánh đầy đủ ngoài đời.
4. Nếu ảnh ngoài khác điều kiện nhiều, cần thêm dữ liệu hoặc augmentation phù hợp.
5. Chưa xử lý class unknown nếu ảnh không thuộc 12 class.

Tự nói hạn chế không làm bài yếu đi. Ngược lại, nó cho thấy mình hiểu bài.

---

## 27. Hướng phát triển

Có thể đề xuất:

- thêm detector như YOLO để tự crop biển báo;
- thu thập thêm ảnh ngoài đời đa dạng;
- thêm class unknown;
- so sánh MobileNetV2 với ResNet/EfficientNet/ViT;
- thử pretrained ImageNet và fine-tuning;
- thử attention module phù hợp;
- đánh giá trên tập dữ liệu ngoài nguồn;
- triển khai demo realtime/webcam.

---

## 28. Cách nói khi kết quả quá cao

Nói:

> Kết quả cao do dữ liệu là ảnh crop, số class 12 và các class có đặc trưng khá rõ. Em không kết luận mô hình đã giải quyết hoàn toàn bài toán nhận diện biển báo ngoài đời, mà kết luận mô hình phân loại tốt trên tập ảnh crop nội bộ. Để dùng ngoài đời cần kiểm tra thêm ảnh ngoài và bổ sung detection.

Đây là câu rất chắc.

---

## 29. Cách nói về việc data đã augment trước đó

Nói:

> Ban đầu tập train có dấu hiệu chứa ảnh augment sẵn nên số lượng train lớn hơn bất thường. Để tránh leakage và đánh giá khách quan hơn, em split lại từ dữ liệu gốc `RGBData` theo tỉ lệ 70/15/15. Augmentation sau đó được xử lý online trong train transform nếu bật config.

---

## 30. Cách nói về checkpoint khi bị dừng train

Nói:

> Notebook lưu best checkpoint theo validation. Nếu dừng train sau khi đã có checkpoint, có thể load checkpoint tốt nhất để đánh giá tiếp. Nếu dừng trước khi lưu checkpoint nào thì cần train lại.

---

## 31. Cách nói về Colab session

Nói:

> Mỗi session Colab có ổ `/content` tạm thời. Trong session thì train bình thường, nhưng khi runtime tắt hoặc reset, dữ liệu giải nén trong `/content` mất. Vì vậy file nén và checkpoint quan trọng nên lưu trên Drive.

---

## 32. Tự kiểm tra hiểu bài

Nếu trả lời được các câu này là đã nắm khá vững:

1. Bài toán hiện tại là classification hay detection?
2. Vì sao ảnh ngoài cần crop?
3. `ToTensor()` khác `Normalize()` thế nào?
4. Normalize trong notebook là min-max hay z-score?
5. Online augmentation có tạo thêm file ảnh không?
6. Vì sao val/test không dùng random augmentation?
7. MobileNetV2 nhẹ nhờ kỹ thuật gì?
8. Depthwise conv khác conv thường thế nào?
9. Inverted residual là gì?
10. CrossEntropyLoss phạt dự đoán sai tự tin cao thế nào?
11. Learning rate ảnh hưởng gì?
12. Warmup để làm gì?
13. Weight decay giúp gì?
14. Batch size ảnh hưởng VRAM thế nào?
15. Vì sao stride 2 tiết kiệm memory?
16. Checkpoint best khác checkpoint epoch cuối thế nào?
17. Accuracy cao có chắc không overfit không?
18. Confusion matrix đọc thế nào?
19. Nếu test ngoài sai thì kiểm tra gì trước?
20. Nếu muốn ảnh nguyên cảnh thì cần thêm gì?

---

## 33. Bản tóm tắt 60 giây

Nếu chỉ có 60 giây để trình bày:

> Em xây dựng mô hình MobileNetV2 để phân loại 12 loại biển báo giao thông từ ảnh đã crop. Dữ liệu được chia train/validation/test theo tỉ lệ khoảng 70/15/15 từ dữ liệu gốc. Ảnh được chuyển RGB, resize tùy chọn, đưa về tensor và normalize theo mean/std. Train set có thể dùng augmentation online để tăng tính đa dạng, còn val/test không dùng augmentation ngẫu nhiên. MobileNetV2 được chọn vì nhẹ, dùng depthwise separable convolution và inverted residual block. Mô hình được train bằng CrossEntropyLoss, SGD momentum, có scheduler/warmup, checkpoint best theo validation. Sau train, em đánh giá bằng accuracy, loss, confusion matrix và per-class accuracy. Vì đây là classifier nên khi test ảnh ngoài cần crop vùng biển báo trước; nếu muốn xử lý ảnh nguyên cảnh cần bổ sung detector.

---

## 34. Kết luận phần 12

Để bài này vững, cần giữ ba điều:

1. Nói đúng phạm vi: classification ảnh crop.
2. Hiểu đúng pipeline: data -> transform -> model -> train -> evaluate -> inference.
3. Không thần thánh hóa accuracy: đọc cùng loss, confusion matrix, leakage và test ngoài.

Khi nắm ba điều này, phần code không còn là những cell rời rạc nữa, mà là một hệ thống train model có logic rõ ràng từ đầu đến cuối.
