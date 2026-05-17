# CELL 15: DỰ ĐOÁN ẢNH TỪ BÊN NGOÀI

## 1. Pipeline dự đoán

```
Ảnh bất kỳ (jpg/png, bất kỳ size)
    ↓ Resize(96, 96)
    ↓ ToTensor()
    ↓ Normalize(MEAN, STD)
    ↓ model(tensor)
    ↓ softmax → xác suất
    ↓ top-5 classes + confidence
Kết quả hiển thị
```

## 2. Crop Box (Optional)

```python
predict_image(filepath, model, device, crop_box=(left, top, right, bottom))
```

Nếu ảnh có nhiều background → crop chỉ vùng biển báo trước khi đưa vào model. Model chỉ biết phân loại, KHÔNG biết detect.

## 3. Softmax

```python
probs = torch.softmax(outputs, dim=1)
```

Chuyển logits (giá trị bất kỳ) thành xác suất [0,1], tổng = 1:
```
Logits:  [2.5, 1.0, 0.3, ...]
Softmax: [0.65, 0.14, 0.07, ...]  → 65% class 0, 14% class 1, ...
```

## 4. Hạn chế quan trọng

1. **Domain shift:** Model train trên biển Đức (GTSRB) → accuracy thấp trên biển Việt Nam
2. **Cần crop sẵn:** Model phân loại, không detect → ảnh phải chỉ chứa biển báo
3. **Confidence thấp = không chắc:** Nếu <50% → model không quen loại ảnh này

## 5. Tác dụng
Cell này cho phép demo model trên ảnh thực tế — quan trọng cho bảo vệ đề tài (live demo).
