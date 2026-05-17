# CELL 10: ĐÁNH GIÁ TRÊN TEST SET

## 1. Nội dung

```python
model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
```

## 2. Tại sao load best_model?
Training loop lưu model có val_acc cao nhất → đó là model tốt nhất. Epoch cuối cùng KHÔNG nhất thiết là tốt nhất (có thể đã bắt đầu overfit).

## 3. Metrics

### Accuracy
```
Accuracy = Số dự đoán đúng / Tổng số samples
```
Đơn giản nhưng bị ảnh hưởng bởi class imbalance. Nếu 90% data là class A → model chỉ cần đoán A luôn = 90% accuracy.

### Precision
```
Precision = TP / (TP + FP) = "Khi model nói X, đúng bao nhiêu %?"
```
Quan trọng khi false positive nguy hiểm (model nói "Stop" nhưng thực ra là "Yield" → xe dừng sai chỗ).

### Recall
```
Recall = TP / (TP + FN) = "Trong tất cả X thực, model tìm được bao nhiêu %?"
```
Quan trọng khi false negative nguy hiểm (model bỏ lỡ biển "Stop" → tai nạn).

### F1-Score
```
F1 = 2 × Precision × Recall / (Precision + Recall)
```
Trung bình hài hòa — cân bằng giữa precision và recall. F1 cao = cả hai đều cao.

## 4. Macro vs Weighted Average

| | Macro | Weighted |
|:---|:---|:---|
| Cách tính | Trung bình đều 43 classes | Trung bình có trọng số theo số samples |
| Class imbalance | Mỗi class đóng góp bằng nhau | Class nhiều ảnh đóng góp nhiều hơn |
| Dùng khi | Quan tâm mọi class đều | Quan tâm hiệu suất tổng thể |

## 5. Tác dụng
Đánh giá THỰC SỰ trên test set (data model chưa bao giờ thấy). Đây là kết quả chính thức của đề tài.
