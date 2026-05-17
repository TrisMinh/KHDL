# CELL 11: CONFUSION MATRIX & VISUALIZATION

## 1. Confusion Matrix là gì?

Ma trận 43×43 — hàng = True label, cột = Predicted label.

```
         Pred: Stop  Pred: Yield  Pred: 50km
True Stop:    58         2           0
True Yield:    1        87           0
True 50km:     0         0          95
```

## 2. Cách đọc

- **Đường chéo chính** (diagonal): Dự đoán ĐÚNG → càng đậm càng tốt
- **Ngoài đường chéo**: Dự đoán SAI → tìm cell đậm ngoài diagonal = model hay nhầm cặp nào
- **Hàng sum ≠ cột sum:** Class imbalance trong test set

## 3. Phát hiện nhầm lẫn thường gặp

GTSRB có nhiều cặp biển giống nhau:
- Speed limit 30 vs 80 (số giống nhau khi mờ)
- Speed limit 50 vs 60 (số 5 vs 6)
- Turn left vs Go straight or left (mũi tên giống)

Confusion matrix cho thấy CHÍNH XÁC model nhầm cặp nào → hướng cải thiện.

## 4. Per-class Accuracy Bar Chart
```python
per_class_acc = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
```
- Bar chart 43 cột, mỗi cột = accuracy 1 class
- Màu: Đỏ < 85%, Vàng 85-95%, Xanh > 95%
- Nhanh chóng thấy class nào yếu nhất → ưu tiên cải thiện

## 5. Tác dụng
Confusion matrix là công cụ phân tích SÂU nhất — không chỉ biết "đúng bao nhiêu %" mà biết "sai ở đâu, sai thế nào". Quan trọng cho báo cáo và cải thiện model.
