# CELL 12: SAMPLE PREDICTIONS

## 1. Nội dung
Hiển thị grid ảnh từ test set với:
- Ảnh gốc (sau inverse normalize)
- Dự đoán của model (class + confidence %)
- True label
- ✅ đúng / ❌ sai

## 2. Tại sao cần hiển thị ảnh?
- Metrics (accuracy, F1) chỉ cho **con số tổng** → không thấy model sai ở đâu
- Hiển thị ảnh cho thấy **trực quan**: ảnh sai trông như thế nào? Mờ? Bị che? Class giống nhau?
- Giúp xác định: model sai do yếu kém hay do ảnh quá khó (ambiguous)?

## 3. Confidence Analysis
- **Confidence cao (>95%) + đúng:** Model tự tin và chính xác → tốt
- **Confidence cao + sai:** Model tự tin nhưng sai → dangerous, cần cải thiện
- **Confidence thấp (<50%) + sai:** Model không chắc chắn → có thể cải thiện bằng thêm data
- **Confidence thấp + đúng:** Model "may mắn" đoán đúng → cần thêm training

## 4. Tác dụng
Qualitative evaluation — đánh giá chất lượng bằng mắt, bổ sung cho quantitative metrics. Trong báo cáo, ảnh minh họa dự đoán rất thuyết phục.
