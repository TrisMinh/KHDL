# Traffic Sign Classifier Web

Landing page web cho mô hình MobileNetV2 + SE Attention.

## Chạy web

```bash
cd LandingPage
pip install -r requirements.txt
python app.py
```

Mở trình duyệt tại:

```text
http://localhost:7860
```

## Model

Mặc định web load model tại:

```text
best_model.pth
```

Nếu muốn dùng model khác:

```bash
MODEL_PATH=/path/to/model.pth python app.py
```

## Cách dùng

1. Upload ảnh có biển báo.
2. Kéo crop đúng vùng biển báo.
3. Bấm Predict.
4. Web trả về class dự đoán và top-5 confidence.

Lưu ý: mô hình là classifier, nên ảnh ngoài đời cần crop vùng biển báo trước khi dự đoán.
