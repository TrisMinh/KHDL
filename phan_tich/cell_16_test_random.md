# CELL 16: TEST 10 ẢNH NGẪU NHIÊN TỪ TEST SET

## 1. Nội dung
Chọn ngẫu nhiên 10 ảnh từ GTSRB test set → predict → hiển thị kết quả.

## 2. Tại sao ngẫu nhiên?
- Tránh cherry-picking (chọn ảnh dễ để khoe)
- Mỗi lần chạy cho 10 ảnh khác nhau → thấy model hoạt động tổng quát
- Thuyết phục hơn so với chọn ảnh thủ công

## 3. Thông tin hiển thị
- Ảnh gốc (sau inverse normalize)
- ✅/❌ + Predicted class + Confidence %
- True class
- Tổng: X/10 đúng

## 4. Khi nào dùng?
- Sau khi train xong → kiểm tra nhanh model có hoạt động không
- Chạy nhiều lần → nếu luôn 9-10/10 đúng → model ổn
- Nếu có ảnh sai → xem ảnh đó có gì đặc biệt (mờ, bị che, giống class khác?)

## 5. Tác dụng
Quick sanity check — kiểm tra nhanh model trước khi chạy full evaluation.
