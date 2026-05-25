# 2.4.3. Loss Function, Optimizer và Learning Rate

Trong quá trình huấn luyện, mô hình MobileNetV2 cần ba thành phần chính: hàm mất mát để đo mức độ dự đoán sai, optimizer để cập nhật trọng số, và learning rate scheduler để điều chỉnh tốc độ học theo từng epoch.

## Loss Function

Mô hình sử dụng `CrossEntropyLoss` vì bài toán là phân loại nhiều lớp. Đầu ra của mô hình là `logits`, tương ứng với điểm dự đoán cho từng lớp biển báo. `CrossEntropyLoss` sẽ so sánh các logits này với nhãn thật để tính lỗi.

Trong notebook:

```python
criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
```

`label_smoothing = 0.05` được dùng để giảm hiện tượng mô hình quá tự tin vào một lớp duy nhất. Thay vì ép nhãn đúng có xác suất tuyệt đối là 1, label smoothing làm nhãn mềm hơn một chút. Điều này giúp mô hình tổng quát tốt hơn, đặc biệt khi dữ liệu có nhiễu hoặc một số ảnh biển báo khá giống nhau.

## Optimizer

Optimizer được sử dụng là `SGD` kết hợp với momentum và Nesterov:

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=CONFIG['lr'],
    momentum=CONFIG['momentum'],
    weight_decay=CONFIG['weight_decay'],
    nesterov=True
)
```

Các tham số chính:

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `lr` | `0.003` | tốc độ cập nhật trọng số |
| `momentum` | `0.9` | giúp hướng cập nhật ổn định hơn |
| `weight_decay` | `1e-4` | giảm overfitting bằng regularization |
| `nesterov` | `True` | cải thiện cách cập nhật dựa trên momentum |

SGD phù hợp với bài toán phân loại ảnh vì thường cho khả năng tổng quát tốt. Momentum giúp quá trình học bớt dao động, còn weight decay hạn chế việc mô hình học quá khớp với tập train.

## Learning Rate Scheduler

Notebook không giữ learning rate cố định trong suốt quá trình train. Thay vào đó, mô hình dùng scheduler dạng:

```text
Warmup -> Cosine Annealing
```

Trong 5 epoch đầu, learning rate được tăng dần từ nhỏ lên giá trị chính. Giai đoạn này gọi là warmup. Warmup giúp mô hình ổn định hơn khi trọng số ban đầu còn ngẫu nhiên.

Sau warmup, learning rate giảm dần theo đường cosine. Ở giai đoạn đầu, learning rate còn đủ lớn để mô hình học các đặc trưng quan trọng. Ở giai đoạn cuối, learning rate nhỏ hơn giúp mô hình tinh chỉnh trọng số nhẹ nhàng hơn.

Thông số trong notebook:

| Thành phần | Giá trị |
|---|---:|
| Tổng số epoch | `20` |
| Warmup epoch | `5` |
| Learning rate cao nhất | `0.003` |
| Scheduler | Warmup + Cosine Annealing |

Tóm lại, `CrossEntropyLoss` giúp đo lỗi phân loại, `SGD + momentum` cập nhật trọng số ổn định, còn `Warmup + Cosine Annealing` giúp quá trình học vừa ổn định ở đầu, vừa tinh chỉnh tốt ở cuối.
