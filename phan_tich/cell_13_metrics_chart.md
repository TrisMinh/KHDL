# CELL 13: BIỂU ĐỒ TỔNG HỢP METRICS

## 1. Nội dung
Bar chart so sánh 4 metrics: Accuracy, Precision, Recall, F1-Score trên toàn bộ test set.

## 2. Ý nghĩa từng metric trong bài toán biển báo

| Metric | Câu hỏi | Ứng dụng thực tế |
|:---|:---|:---|
| **Accuracy** | "Đúng bao nhiêu % tổng thể?" | Đánh giá chung |
| **Precision** | "Khi model nói X, đúng bao nhiêu %?" | Tránh false alarm (báo sai) |
| **Recall** | "Bao nhiêu % X thực tế được tìm ra?" | Tránh bỏ sót (miss) |
| **F1** | "Cân bằng giữa P và R?" | Đánh giá tổng hợp |

## 3. Khi nào các metrics khác nhau?
- **Precision cao, Recall thấp:** Model "khắt khe" — chỉ dự đoán khi rất chắc → ít sai nhưng bỏ sót nhiều
- **Recall cao, Precision thấp:** Model "dễ dãi" — dự đoán nhiều → bắt được hết nhưng hay false alarm
- **Cả hai cao:** Model tốt ✅

## 4. Mục tiêu
Cả 4 metrics > 95% → model đạt chất lượng cao cho bài toán phân loại biển báo.
