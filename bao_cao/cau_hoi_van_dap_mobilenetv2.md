# Câu hỏi vấn đáp MobileNetV2

Tài liệu này tổng hợp các câu hỏi có thể gặp khi bảo vệ phần MobileNetV2. Nội dung tập trung vào mô hình, kiến trúc, cơ chế học, SE Attention, huấn luyện và đánh giá. Không đi sâu vào các phần phụ như cách load dữ liệu, giải nén file hoặc chia thư mục.

---

## 1. Thuật toán chính của đề tài là gì?

Thuật toán chính được sử dụng là **MobileNetV2 kết hợp SE Attention** để phân loại biển báo giao thông.

MobileNetV2 là một kiến trúc mạng nơ-ron tích chập nhẹ, được thiết kế để đạt độ chính xác tốt nhưng vẫn có số tham số và chi phí tính toán thấp. Trong đề tài này, MobileNetV2 được huấn luyện để nhận ảnh biển báo đầu vào kích thước `224x224` và phân loại vào `12` lớp biển báo.

Ở phiên bản v7, mô hình được cải tiến thêm **SE Attention**. SE giúp mô hình tự học channel đặc trưng nào quan trọng hơn, từ đó tập trung tốt hơn vào các thông tin như màu sắc, viền biển báo, ký hiệu, chữ số hoặc hình dạng.

---

## 2. Vì sao chọn MobileNetV2 thay vì CNN thường?

MobileNetV2 được chọn vì đây là mô hình CNN nhẹ, phù hợp với bài toán phân loại ảnh nhưng không quá nặng như các kiến trúc CNN lớn.

Nếu dùng convolution thường, số phép tính và số tham số sẽ lớn vì mỗi filter xử lý đồng thời cả không gian và toàn bộ channel đầu vào. MobileNetV2 sử dụng **depthwise separable convolution**, tách convolution thành hai bước:

```text
Depthwise convolution: xử lý không gian riêng từng channel
Pointwise convolution: dùng 1x1 conv để trộn thông tin giữa các channel
```

Cách tách này giúp giảm mạnh chi phí tính toán nhưng vẫn giữ khả năng trích xuất đặc trưng ảnh. Với bài toán biển báo giao thông, đặc trưng thường là màu sắc, hình dạng, viền, ký hiệu và chữ số, nên MobileNetV2 đủ mạnh để học mà vẫn nhẹ.

---

## 3. MobileNetV2 khác MobileNetV1 ở điểm nào?

MobileNetV1 chủ yếu dựa vào **depthwise separable convolution** để giảm chi phí tính toán.

MobileNetV2 kế thừa ý tưởng đó nhưng bổ sung hai kỹ thuật quan trọng:

- **Inverted residual block**
- **Linear bottleneck**

Trong MobileNetV2, dữ liệu đi qua block theo hướng:

```text
Input channel hẹp
-> mở rộng channel bằng 1x1 convolution
-> depthwise convolution
-> nén lại bằng 1x1 convolution tuyến tính
```

Điểm này khác residual block truyền thống, vì residual block thường đi từ rộng sang hẹp rồi rộng lại. MobileNetV2 làm ngược lại: từ hẹp mở rộng ra rồi nén về hẹp, nên gọi là **inverted residual**.

---

## 4. Depthwise separable convolution là gì?

Depthwise separable convolution là kỹ thuật tách convolution thường thành hai phần:

```text
Depthwise convolution
Pointwise convolution
```

**Depthwise convolution** dùng một kernel riêng cho từng channel. Nó học đặc trưng không gian như cạnh, góc, vùng màu hoặc texture trong từng channel.

**Pointwise convolution** là convolution `1x1`, dùng để kết hợp thông tin giữa các channel.

Nếu convolution thường vừa học không gian vừa trộn channel trong một bước, thì depthwise separable convolution chia hai việc này ra. Nhờ vậy, số phép tính giảm đáng kể.

---

## 5. Vì sao depthwise separable convolution nhẹ hơn convolution thường?

Giả sử input có `M` channel, output có `N` channel, kernel size là `K x K`, feature map có kích thước `H x W`.

Convolution thường có chi phí xấp xỉ:

```text
H x W x M x N x K x K
```

Depthwise separable convolution gồm:

```text
Depthwise: H x W x M x K x K
Pointwise: H x W x M x N
```

Tổng chi phí nhỏ hơn nhiều so với convolution thường, đặc biệt khi `N` lớn. Đây là lý do MobileNetV2 có thể giữ độ chính xác tốt nhưng vẫn nhẹ.

---

## 6. Inverted residual block là gì?

Inverted residual block là khối chính của MobileNetV2.

Luồng xử lý trong block:

```text
Input
-> 1x1 expansion convolution
-> 3x3 depthwise convolution
-> SE Attention trong bản v7
-> 1x1 projection convolution
-> skip connection nếu đủ điều kiện
```

Gọi là inverted residual vì block bắt đầu từ tensor có số channel hẹp, mở rộng channel lên ở giữa, rồi nén lại ở cuối. Điều này ngược với residual block truyền thống.

---

## 7. Expansion layer dùng để làm gì?

Expansion layer là convolution `1x1` dùng để tăng số channel trước khi thực hiện depthwise convolution.

Ví dụ nếu input có `C` channel và expansion ratio là `t = 6`, số channel ở giữa block sẽ là:

```text
hidden_dim = C x 6
```

Mục đích của expansion là tạo không gian biểu diễn rộng hơn để mô hình học được nhiều đặc trưng hơn. Nếu không mở rộng channel, tensor bottleneck quá hẹp có thể làm mất thông tin khi qua activation và convolution.

---

## 8. Linear bottleneck là gì?

Linear bottleneck là phần projection cuối block, nơi MobileNetV2 dùng convolution `1x1` để nén channel về số lượng nhỏ hơn nhưng **không dùng ReLU ở lớp cuối này**.

Lý do không dùng ReLU ở bottleneck cuối là vì tensor lúc này có số chiều hẹp. Nếu dùng ReLU, các giá trị âm bị đưa về 0, có thể làm mất thông tin quan trọng. Vì vậy MobileNetV2 để lớp projection cuối là tuyến tính, giúp giữ thông tin tốt hơn.

---

## 9. Skip connection trong MobileNetV2 hoạt động khi nào?

Skip connection được dùng khi input và output của block có cùng shape.

Điều kiện thường là:

```text
stride = 1
in_channels = out_channels
```

Khi đó, output của block được tính:

```text
output = input + block(input)
```

Skip connection giúp gradient lan truyền tốt hơn, giảm mất mát thông tin và giúp mô hình dễ huấn luyện hơn.

---

## 10. Vì sao có block không dùng skip connection?

Một số block không dùng skip connection vì shape input và output không giống nhau.

Ví dụ khi block dùng `stride = 2`, kích thước feature map bị giảm, nên không thể cộng trực tiếp input với output. Ngoài ra, nếu số channel đầu vào và đầu ra khác nhau thì cũng không cộng trực tiếp được.

Do đó, MobileNetV2 chỉ dùng skip connection ở các block giữ nguyên kích thước và số channel.

---

## 11. ReLU6 là gì và vì sao MobileNetV2 dùng ReLU6?

ReLU6 là biến thể của ReLU, được định nghĩa:

```text
ReLU6(x) = min(max(0, x), 6)
```

Nó giới hạn đầu ra trong khoảng từ `0` đến `6`. MobileNetV2 dùng ReLU6 vì nó phù hợp với các thiết bị có tài nguyên thấp và tính toán lượng tử hóa. Ngoài ra, ReLU6 giúp kiểm soát biên độ activation, làm quá trình huấn luyện ổn định hơn.

Trong notebook, ReLU6 được dùng ở các tầng expansion và depthwise convolution, nhưng không dùng ở projection cuối của linear bottleneck.

---

## 12. SE Attention là gì?

SE là viết tắt của **Squeeze-and-Excitation**. Đây là attention theo channel, tức mô hình học xem channel đặc trưng nào quan trọng hơn.

Một feature map có dạng:

```text
C x H x W
```

Trong đó mỗi channel có thể học một loại đặc trưng khác nhau. Ví dụ với ảnh biển báo, có channel học vùng màu đỏ, channel học viền trắng, channel học chữ số, channel học hình tròn hoặc tam giác.

SE Attention giúp mô hình gán trọng số cho từng channel. Channel hữu ích được giữ mạnh hơn, channel ít hữu ích hoặc nhiễu bị giảm ảnh hưởng.

---

## 13. SE Attention hoạt động như thế nào?

SE gồm ba bước:

```text
Squeeze
Excitation
Reweight
```

**Squeeze** dùng global average pooling để gom thông tin không gian:

```text
C x H x W -> C x 1 x 1
```

Mỗi channel được rút gọn thành một giá trị trung bình, đại diện cho mức độ kích hoạt tổng quát của channel đó.

**Excitation** đưa vector channel qua một mạng nhỏ:

```text
C -> C/reduction -> C
```

Mạng nhỏ này học quan hệ giữa các channel và tạo ra trọng số attention.

**Reweight** nhân trọng số vừa học vào feature map ban đầu:

```text
output = feature_map x attention_weight
```

Nhờ vậy, mô hình tự điều chỉnh channel nào nên nhấn mạnh và channel nào nên giảm.

---

## 14. Vì sao thêm SE Attention vào MobileNetV2?

SE Attention được thêm vào để cải thiện khả năng chọn lọc đặc trưng của mô hình.

MobileNetV2 vốn nhẹ vì dùng depthwise separable convolution. Tuy nhiên, depthwise convolution xử lý từng channel riêng nên việc chọn channel quan trọng là vấn đề đáng chú ý. SE Attention bổ sung cơ chế học trọng số channel, giúp mô hình tập trung tốt hơn vào các đặc trưng có ích cho phân loại.

Với biển báo giao thông, các đặc trưng quan trọng thường là:

- màu sắc chính;
- viền biển báo;
- hình dạng biển báo;
- chữ số;
- ký hiệu bên trong.

SE giúp mô hình học xem đặc trưng nào đáng tin hơn trong từng ảnh.

---

## 15. SE Attention được đặt ở đâu trong block?

Trong mô hình v7, SE Attention được đặt sau depthwise convolution và trước projection convolution.

Luồng block:

```text
Input
-> Expansion 1x1
-> Depthwise 3x3
-> SE Attention
-> Projection 1x1
-> Skip connection nếu đủ điều kiện
```

Đặt SE sau depthwise convolution là hợp lý vì lúc này mỗi channel đã chứa đặc trưng không gian riêng. SE có thể đánh giá channel nào quan trọng trước khi projection nén tensor về bottleneck hẹp.

---

## 16. Reduction trong SE là gì?

Reduction là hệ số giảm chiều trong mạng nhỏ của SE.

Nếu số channel là `C` và `reduction = 4`, SE sẽ dùng:

```text
C -> C/4 -> C
```

Ví dụ nếu hidden channel là `96`:

```text
96 -> 24 -> 96
```

Reduction giúp giảm số tham số của SE. Nếu không giảm chiều, mạng attention sẽ nặng hơn. Trong đề tài, `reduction = 4` giúp SE đủ khả năng học quan hệ channel nhưng vẫn giữ mô hình tương đối nhẹ.

---

## 17. Thêm SE có làm mô hình nặng hơn không?

Có. SE Attention làm tăng số tham số vì mỗi SE block có thêm hai convolution `1x1`:

```text
C -> C/reduction
C/reduction -> C
```

Trong kết quả của mô hình v7:

```text
Total parameters: 4,504,292
Model size: khoảng 17.2 MB FP32
```

Số tham số cao hơn MobileNetV2 gốc vì có thêm SE Attention. Tuy nhiên, kích thước khoảng 17 MB vẫn là nhẹ so với nhiều mô hình CNN lớn. Vì vậy đánh đổi này chấp nhận được nếu SE giúp tăng độ chính xác.

---

## 18. Input của mô hình là bao nhiêu?

Input của mô hình là ảnh RGB kích thước:

```text
224 x 224 x 3
```

Ảnh được resize/crop về `224x224`, chuyển thành tensor và normalize trước khi đưa vào mô hình.

Kích thước `224x224` phù hợp với MobileNetV2 chuẩn và giữ được nhiều chi tiết của biển báo như viền, ký hiệu, chữ số và mũi tên.

---

## 19. Vì sao first convolution dùng stride 2?

Với input `224x224`, first convolution dùng stride `2` để giảm kích thước feature map từ:

```text
224x224 -> 112x112
```

Điều này giúp giảm chi phí tính toán ngay từ đầu nhưng vẫn giữ đủ thông tin không gian. Nếu giữ stride `1`, feature map lớn quá lâu, làm tăng thời gian train và VRAM.

Vì input đang là `224x224`, stride `2` là hợp lý và đúng với cấu hình MobileNetV2 chuẩn.

---

## 20. Feature map thay đổi như thế nào trong mô hình?

Luồng spatial size chính:

```text
Input: 224x224
Conv đầu: 112x112
Stage 1: 112x112
Stage 2: 56x56
Stage 3: 28x28
Stage 4: 14x14
Stage 5: 14x14
Stage 6: 7x7
Stage 7: 7x7
Global average pooling: 1x1
```

Spatial size giảm dần, trong khi số channel tăng dần. Các tầng đầu học đặc trưng đơn giản như cạnh và màu sắc, các tầng sâu học đặc trưng trừu tượng hơn như hình dạng biển báo, ký hiệu và chữ số.

---

## 21. Vì sao dùng global average pooling?

Global average pooling dùng để gom feature map cuối về vector đặc trưng.

Trước pooling, tensor cuối có dạng:

```text
1280 x 7 x 7
```

Sau global average pooling:

```text
1280 x 1 x 1
```

Nếu flatten trực tiếp `1280 x 7 x 7`, classifier sẽ có rất nhiều tham số. Global average pooling giúp giảm số tham số, giảm overfitting và làm classifier gọn hơn.

---

## 22. Classifier cuối hoạt động như thế nào?

Sau global average pooling, mô hình có vector đặc trưng `1280` chiều. Vector này được đưa vào fully connected layer để tạo logits cho `12` lớp.

Classifier có dạng:

```text
Dropout
Linear(1280, 12)
```

Số tham số classifier:

```text
1280 x 12 + 12 = 15,372
```

Classifier nhỏ vì phần trích xuất đặc trưng đã được backbone MobileNetV2-SE đảm nhiệm.

---

## 23. Logits là gì?

Logits là đầu ra thô của mô hình trước softmax.

Nếu có `12` lớp, mô hình tạo ra vector:

```text
[z1, z2, ..., z12]
```

Mỗi giá trị là điểm dự đoán cho một class. Khi cần tính xác suất, logits được đưa qua softmax. Tuy nhiên khi train bằng `CrossEntropyLoss` trong PyTorch, không cần thêm softmax trong `forward` vì `CrossEntropyLoss` đã xử lý phần này bên trong.

---

## 24. Vì sao không đặt softmax trong model forward?

Không đặt softmax trong `forward` vì PyTorch `CrossEntropyLoss` nhận logits trực tiếp.

`CrossEntropyLoss` đã bao gồm:

```text
LogSoftmax + Negative Log Likelihood
```

Nếu thêm softmax trước, việc tính loss có thể kém ổn định hơn và không đúng cách dùng chuẩn. Softmax chỉ nên dùng khi cần hiển thị xác suất dự đoán sau khi model đã inference.

---

## 25. Loss function được dùng là gì?

Mô hình dùng:

```text
CrossEntropyLoss
```

Đây là loss phù hợp cho bài toán phân loại nhiều lớp. Mỗi ảnh thuộc một class đúng, mô hình tạo logits cho 12 class, và CrossEntropyLoss đo mức độ sai khác giữa dự đoán và nhãn thật.

Công thức tổng quát:

```text
L = - sum(y_i log(p_i))
```

Với bài toán một nhãn, nếu class đúng là `k`, công thức rút gọn:

```text
L = -log(p_k)
```

Nếu mô hình dự đoán xác suất cao cho class đúng thì loss thấp. Nếu xác suất class đúng thấp thì loss cao.

---

## 26. Label smoothing là gì?

Label smoothing là kỹ thuật làm mềm nhãn thật.

Thay vì nhãn đúng có xác suất tuyệt đối là `1` và các nhãn sai là `0`, label smoothing phân phối một phần nhỏ xác suất sang các class còn lại.

Trong đề tài:

```text
label_smoothing = 0.05
```

Kỹ thuật này giúp mô hình không quá tự tin vào một class duy nhất, từ đó giảm overfitting và cải thiện khả năng tổng quát hóa.

---

## 27. Optimizer được dùng là gì?

Optimizer được sử dụng là:

```text
SGD + momentum + Nesterov
```

Cấu hình:

```text
learning rate = 0.003
momentum = 0.9
weight_decay = 1e-4
nesterov = True
```

SGD cập nhật trọng số dựa trên gradient. Momentum giúp hướng cập nhật ổn định hơn bằng cách tích lũy xu hướng từ các bước trước. Weight decay đóng vai trò regularization, hạn chế trọng số quá lớn và giảm overfitting. Nesterov momentum giúp cải thiện cách cập nhật bằng cách ước lượng trước hướng di chuyển.

---

## 28. Learning rate scheduler là gì?

Learning rate scheduler là cơ chế thay đổi learning rate trong quá trình huấn luyện.

Mô hình dùng:

```text
Warmup + Cosine Annealing
```

Trong `5` epoch đầu, learning rate tăng dần từ nhỏ lên giá trị chính. Đây là warmup, giúp mô hình ổn định khi trọng số ban đầu còn ngẫu nhiên.

Sau warmup, learning rate giảm dần theo cosine. Giai đoạn này giúp mô hình tinh chỉnh trọng số nhẹ hơn ở cuối quá trình huấn luyện.

---

## 29. Vì sao validation accuracy có lúc cao hơn train accuracy?

Điều này có thể xảy ra và không nhất thiết là lỗi.

Lý do:

- train set có augmentation nên ảnh train khó hơn;
- train có dropout nên mô hình bị regularize;
- validation không dùng augmentation ngẫu nhiên;
- label smoothing làm train loss/accuracy có thể không quá cao tuyệt đối.

Vì vậy nếu validation accuracy cao hơn train accuracy một chút, đó có thể là dấu hiệu regularization đang hoạt động, không nhất thiết là overfitting.

---

## 30. Có bị overfitting không?

Để kết luận overfitting, cần nhìn đồng thời train loss, validation loss, train accuracy và validation accuracy.

Dấu hiệu overfitting thường là:

```text
train accuracy tăng
validation accuracy giảm hoặc đứng yên
train loss giảm
validation loss tăng
```

Trong kết quả hiện tại, validation accuracy và test metrics đều cao, validation loss không tăng mạnh bất thường, nên chưa thấy dấu hiệu overfitting rõ ràng trên tập test nội bộ.

Tuy nhiên, cần lưu ý mô hình được test trên dữ liệu đã crop/tiền xử lý cùng kiểu với train. Khi đưa ảnh ngoài đời có background phức tạp, mô hình có thể kém hơn nếu không crop đúng vùng biển báo.

---

## 31. Kết quả test 99.4% có ý nghĩa gì?

Kết quả test khoảng `99.4%` nghĩa là mô hình dự đoán đúng phần lớn ảnh trong tập test nội bộ.

Các chỉ số:

```text
Accuracy
Precision
Recall
F1-score
```

đều khoảng `99.4%`, cho thấy mô hình không chỉ đúng nhiều ảnh mà còn khá cân bằng giữa các class.

Tuy nhiên, đây là kết quả trên test set đã được chuẩn bị cùng phân phối với train. Không nên hiểu rằng mô hình chắc chắn đạt 99.4% trên mọi ảnh ngoài đời. Nếu ảnh ngoài đời chưa crop biển báo, bị mờ, quá nhỏ hoặc background phức tạp, độ chính xác có thể giảm.

---

## 32. Confusion matrix dùng để làm gì?

Confusion matrix cho biết mô hình nhầm class nào sang class nào.

Trong confusion matrix:

```text
Hàng: nhãn thật
Cột: nhãn dự đoán
Đường chéo chính: dự đoán đúng
Ngoài đường chéo: dự đoán sai
```

Normalized confusion matrix hiển thị theo phần trăm, giúp dễ xem mỗi class được dự đoán đúng bao nhiêu phần trăm và bị nhầm sang class nào.

Overall metrics cho biết mô hình tốt tổng thể, còn confusion matrix cho biết lỗi cụ thể nằm ở đâu.

---

## 33. Classification report là gì?

Classification report là bảng đánh giá chi tiết theo từng class.

Các cột chính:

```text
precision
recall
f1-score
support
```

Precision cho biết trong các ảnh được dự đoán là class đó, có bao nhiêu ảnh đúng. Recall cho biết trong các ảnh thật thuộc class đó, mô hình tìm đúng được bao nhiêu. F1-score là trung bình điều hòa giữa precision và recall. Support là số ảnh thật của class đó trong tập test.

Classification report giúp kiểm tra class nào mạnh, class nào yếu, thay vì chỉ nhìn accuracy tổng.

---

## 34. Missed và false positive là gì?

Với một class cụ thể:

**Missed** là ảnh thật thuộc class đó nhưng mô hình dự đoán sang class khác.

```text
True = no_entry
Predicted = no_stopping
```

Với class `no_entry`, đây là missed.

**False positive** là ảnh thật thuộc class khác nhưng mô hình lại dự đoán thành class đang xét.

```text
True = no_stopping
Predicted = no_entry
```

Với class `no_entry`, đây là false positive.

---

## 35. Vì sao mô hình cần crop ảnh ngoài đời trước khi dự đoán?

Mô hình là **classifier**, không phải detector.

Classifier chỉ học phân loại khi vùng biển báo đã rõ trong ảnh. Nếu đưa nguyên ảnh đường phố có nhiều background, mô hình có thể bị nhiễu vì phần lớn ảnh không phải biển báo.

Do đó khi dùng ảnh ngoài đời, cần crop vùng biển báo trước rồi mới đưa vào mô hình. Nếu muốn xử lý ảnh nguyên cảnh tự động, cần bổ sung một mô hình detection để phát hiện vị trí biển báo trước.

---

## 36. Mô hình này có phát hiện vị trí biển báo không?

Không. Mô hình hiện tại chỉ làm classification.

Nó trả lời câu hỏi:

```text
Ảnh crop này thuộc loại biển báo nào?
```

Nó không trả lời câu hỏi:

```text
Biển báo nằm ở đâu trong ảnh?
```

Muốn phát hiện vị trí, cần dùng object detection như YOLO, SSD hoặc Faster R-CNN. Sau đó mới đưa vùng crop vào MobileNetV2 để phân loại.

---

## 37. Vì sao tham số mô hình là 4.5 triệu, có quá nhiều không?

Mô hình v7 có khoảng:

```text
4,504,292 parameters
```

Con số này cao hơn MobileNetV2 gốc vì đã thêm SE Attention. SE thêm các lớp nhỏ để học trọng số channel trong nhiều block.

Tuy nhiên, kích thước model khoảng `17.2 MB` ở FP32 vẫn là nhẹ. So với nhiều CNN truyền thống, mô hình vẫn phù hợp cho bài toán cần tốc độ và kích thước nhỏ.

---

## 38. Parameters per layer nghĩa là gì?

Trong log:

```text
features: 4,488,920 params
classifier: 15,372 params
```

`features` là phần backbone trích xuất đặc trưng, gồm convolution đầu, các inverted residual block, SE Attention và convolution cuối. Đây là nơi chứa phần lớn tham số.

`classifier` là phần phân loại cuối, gồm dropout và linear layer từ `1280` đặc trưng sang `12` class. Vì classifier chỉ là một lớp nhỏ nên số tham số rất ít.

---

## 39. Dropout dùng để làm gì?

Dropout là kỹ thuật regularization. Trong quá trình train, dropout ngẫu nhiên tắt một phần neuron để mô hình không phụ thuộc quá mạnh vào một số đặc trưng cụ thể.

Trong mô hình, dropout được đặt trước classifier:

```text
Dropout(0.2)
Linear(1280, 12)
```

Dropout giúp giảm overfitting và cải thiện khả năng tổng quát hóa.

---

## 40. Nếu thầy hỏi mô hình học đặc trưng gì từ biển báo thì trả lời sao?

Mô hình học đặc trưng theo nhiều mức.

Các tầng đầu học đặc trưng đơn giản:

- cạnh;
- góc;
- vùng màu đỏ, xanh, trắng;
- viền tròn, viền tam giác.

Các tầng giữa học bộ phận của biển báo:

- chữ số;
- mũi tên;
- gạch ngang;
- hình ký hiệu.

Các tầng sâu kết hợp đặc trưng để phân biệt class:

- no entry;
- no stopping;
- speed limit;
- roadworks;
- stop sign.

SE Attention giúp mô hình chọn channel đặc trưng quan trọng hơn cho từng ảnh.

---

## 41. Điểm mạnh của mô hình là gì?

Điểm mạnh:

- nhẹ hơn nhiều CNN lớn;
- phù hợp classification ảnh;
- dùng depthwise separable convolution để giảm chi phí;
- có inverted residual và linear bottleneck giúp giữ thông tin tốt;
- thêm SE Attention giúp chọn lọc channel đặc trưng;
- đạt kết quả test cao trên dữ liệu đã chuẩn bị;
- kích thước model vẫn nhỏ, khoảng 17 MB.

---

## 42. Hạn chế của mô hình là gì?

Hạn chế:

- chỉ phân loại ảnh đã crop, không tự phát hiện vị trí biển báo;
- có thể giảm độ chính xác nếu ảnh ngoài đời nhiều background;
- có thể nhầm nếu biển báo nhỏ, mờ, bị che khuất hoặc góc chụp xấu;
- kết quả test cao chủ yếu phản ánh tập test cùng phân phối với train;
- thêm SE Attention làm tăng số tham số so với MobileNetV2 gốc.

---

## 43. Nếu muốn cải thiện tiếp thì làm gì?

Một số hướng cải thiện:

- bổ sung object detection để tự phát hiện biển báo trong ảnh nguyên cảnh;
- tăng dữ liệu ngoài đời với nhiều background, ánh sáng và góc chụp khác nhau;
- thử fine-tune pretrained MobileNetV2;
- thử các attention nhẹ khác như ECA hoặc CBAM;
- thử quantization để giảm kích thước khi triển khai;
- đánh giá thêm trên tập ảnh ngoài đời không cùng phân phối.

---

## 44. Nếu hỏi vì sao không dùng pretrained thì trả lời sao?

Có thể trả lời:

Trong đề tài này, mô hình được huấn luyện từ đầu để kiểm soát toàn bộ kiến trúc và quan sát rõ quá trình học trên dữ liệu biển báo custom. Việc train from scratch giúp đánh giá trực tiếp khả năng học của MobileNetV2-SE trên tập dữ liệu đã xây dựng.

Tuy nhiên, nếu muốn cải thiện khả năng tổng quát hóa, đặc biệt khi dữ liệu không lớn, có thể dùng pretrained weights trên ImageNet rồi fine-tune lại cho bài toán biển báo.

---

## 45. Nếu hỏi vì sao thêm SE mà validation tăng nhanh thì có đáng nghi không?

Validation tăng nhanh không nhất thiết là sai.

Trong dữ liệu biển báo, nếu ảnh đã crop rõ và các class có đặc trưng màu/hình dạng khác nhau, mô hình có thể học rất nhanh. Thêm SE Attention giúp mô hình chọn channel tốt hơn nên validation có thể tăng nhanh hơn baseline.

Tuy nhiên vẫn cần kiểm tra:

- train/val/test không bị trùng ảnh;
- test set tách riêng;
- không dùng augmentation ngẫu nhiên cho val/test;
- đánh giá thêm trên ảnh ngoài đời.

Nếu các điều kiện trên đúng, việc validation tăng nhanh là có thể chấp nhận.

---

## 46. Tóm tắt ngắn nhất để trả lời khi bị hỏi tổng quan

Đề tài sử dụng MobileNetV2 kết hợp SE Attention để phân loại 12 loại biển báo giao thông. MobileNetV2 là CNN nhẹ dùng depthwise separable convolution, inverted residual block và linear bottleneck để giảm chi phí tính toán nhưng vẫn giữ khả năng trích xuất đặc trưng. SE Attention được thêm vào sau depthwise convolution để mô hình học trọng số quan trọng cho từng channel đặc trưng. Mô hình nhận ảnh RGB `224x224`, trích xuất đặc trưng qua backbone, dùng global average pooling và classifier để tạo logits cho 12 lớp. Quá trình huấn luyện dùng CrossEntropyLoss, SGD momentum, label smoothing và scheduler Warmup + Cosine Annealing. Kết quả test đạt khoảng `99.4%`, nhưng mô hình hiện là classifier nên ảnh ngoài đời cần crop vùng biển báo trước khi dự đoán.
