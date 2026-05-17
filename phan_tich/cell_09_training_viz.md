# CELL 9: VISUALIZATION TRAINING HISTORY

## 1. Nội dung
Vẽ 4 biểu đồ theo epoch: (1) Loss, (2) Accuracy, (3) Learning Rate, (4) Time/epoch.

## 2. Biểu đồ Loss (Train vs Val)
- **Train loss giảm đều:** Model đang học → tốt
- **Val loss giảm theo:** Model generalize → không overfit
- **Val loss tăng trong khi train loss giảm:** Overfit → cần stop sớm hơn
- **Cả hai dao động mạnh:** LR quá lớn hoặc batch_size quá nhỏ

## 3. Biểu đồ Accuracy (Train vs Val)
- **Val > Train:** Augmentation đang hoạt động tốt (train khó hơn val)
- **Train >> Val (gap lớn):** Overfit → cần thêm regularization
- **Cả hai ~ngang:** Model đã hội tụ → stop

## 4. Biểu đồ Learning Rate
- Thấy rõ: Warmup (tăng dần) → Cosine Decay (giảm mượt)
- Kiểm tra LR schedule đúng như thiết kế

## 5. Biểu đồ Time per Epoch
- Epoch đầu chậm hơn (GPU cần warmup, compile kernel)
- Các epoch sau ổn định ~90s mỗi epoch
- Nếu thời gian tăng dần → memory leak, cần kiểm tra

## 6. Tác dụng
Visualization giúp **chẩn đoán** training: overfit? underfit? LR đúng? convergence ổn? Đây là kỹ năng quan trọng của ML engineer.
