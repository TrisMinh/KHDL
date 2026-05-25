# Mục Lục Và Lộ Trình Học MobileNetV2 Ver3

## 1. Mục tiêu bộ tài liệu

Bộ tài liệu này dùng để hiểu toàn bộ notebook:

```text
version/mobilenetv2_gtsrb_ver3_datatulam.ipynb
```

Mục tiêu không chỉ là biết bấm chạy cell, mà phải hiểu:

- bài toán đang giải quyết là gì;
- dữ liệu đi từ đâu đến đâu;
- vì sao phải load từ Drive rồi giải nén local;
- vì sao cần split train/val/test;
- transform, normalize, augmentation làm gì;
- MobileNetV2 hoạt động thế nào;
- loss, optimizer, scheduler ảnh hưởng gì;
- training loop lưu checkpoint ra sao;
- evaluation đọc thế nào;
- ảnh ngoài phải crop vì sao;
- lỗi thường gặp xử lý thế nào;
- khi thuyết trình nên nói gì và tránh nói gì.

---

## 2. Thứ tự đọc đề xuất

Đọc theo thứ tự này:

1. `01_pham_vi_bai_toan.md`
2. `02_cau_truc_du_lieu_va_pipeline.md`
3. `03_load_data_drive_giai_nen_colab.md`
4. `04_config_hyperparameters.md`
5. `05_transform_normalize_augment.md`
6. `06_mobilenetv2_architecture.md`
7. `07_loss_optimizer_scheduler.md`
8. `08_training_loop_checkpoint_resume.md`
9. `09_evaluation_metrics_visualization.md`
10. `10_inference_anh_ngoai_crop.md`
11. `11_loi_thuong_gap_troubleshooting.md`
12. `12_checklist_bao_cao_thuyet_trinh_phan_bien.md`

File lớn tổng hợp:

```text
mobileNetV2_ver3_ly_thuyet_chi_tiet.md
```

File lớn dùng khi muốn đọc một bản gom toàn bộ. Các file nhỏ dùng để học từng phần kỹ hơn, dễ ôn và dễ mở lại đúng chủ đề.

---

## 3. To-do học hiểu notebook

Đánh dấu từng mục khi đã hiểu:

- [ ] Hiểu bài toán hiện tại là classification ảnh crop, không phải detection.
- [ ] Hiểu vì sao ảnh ngoài cần crop trước khi đưa vào model.
- [ ] Hiểu cấu trúc `SplitData/train`, `SplitData/val`, `SplitData/test`.
- [ ] Hiểu vì sao split phải từ dữ liệu gốc, tránh ảnh augment lọt qua nhiều split.
- [ ] Hiểu vì sao copy file nén từ Drive về `/content` rồi giải nén sẽ nhanh hơn train trực tiếp từ Drive.
- [ ] Hiểu config `img_size`, `resize_enabled`, `resize_size`, `batch_size`, `epochs`, `lr`.
- [ ] Hiểu `ToTensor()` khác `Normalize()` thế nào.
- [ ] Hiểu normalize trong notebook là z-score theo kênh, không chỉ là chia 255.
- [ ] Hiểu augmentation online không tạo thêm file ảnh trên disk.
- [ ] Hiểu val/test không dùng random augmentation.
- [ ] Hiểu MobileNetV2 dùng depthwise separable convolution để nhẹ hơn convolution thường.
- [ ] Hiểu inverted residual và linear bottleneck ở mức giải thích được.
- [ ] Hiểu CrossEntropyLoss phạt dự đoán sai tự tin cao.
- [ ] Hiểu learning rate, momentum, weight decay, warmup, scheduler.
- [ ] Hiểu một epoch là đi qua toàn bộ train set một lần.
- [ ] Hiểu checkpoint best theo validation khác epoch cuối.
- [ ] Hiểu CUDA OOM thường sửa bằng giảm batch size và restart runtime.
- [ ] Hiểu đọc loss/accuracy curve.
- [ ] Hiểu confusion matrix và per-class accuracy.
- [ ] Hiểu accuracy cao không tự động là overfit, nhưng cũng chưa chứng minh dùng tốt ngoài đời.
- [ ] Hiểu test ảnh ngoài cần đúng crop, RGB, resize, normalize, class mapping.
- [ ] Chuẩn bị được câu trả lời khi thầy hỏi về detection, attention/transformer, overfit, augmentation và normalize.

---

## 4. Checklist chạy notebook trước khi train

Trước khi train, kiểm tra:

- [ ] File nén data đã nằm trên Google Drive.
- [ ] `DRIVE_ARCHIVE_PATH` đúng tên file và đúng đuôi `.rar`, `.zip`, `.tar`, `.tar.gz` hoặc `.tgz`.
- [ ] Cell load data đã mount Drive, copy archive về `/content`, giải nén và tìm được `SplitData`.
- [ ] `SplitData` có đủ `train`, `val`, `test`.
- [ ] Mỗi split có đủ folder class.
- [ ] Số class đúng 12.
- [ ] Số ảnh sau split xấp xỉ 70/15/15.
- [ ] `batch_size` phù hợp GPU, nên bắt đầu từ 32 hoặc thấp hơn nếu OOM.
- [ ] `augment_enabled` đúng ý định.
- [ ] `resize_enabled` đúng với dữ liệu đã resize hay chưa.
- [ ] Nếu vừa bị OOM, đã restart runtime trước khi train lại.

---

## 5. Checklist sau khi train

Sau khi train, kiểm tra:

- [ ] Best checkpoint đã được lưu.
- [ ] Test bằng best checkpoint.
- [ ] Có accuracy/loss trên test set.
- [ ] Có confusion matrix.
- [ ] Có per-class accuracy.
- [ ] Có xem ảnh dự đoán sai.
- [ ] Có test một vài ảnh ngoài đã crop.
- [ ] Có ghi rõ trong báo cáo: mô hình phân loại ảnh crop, chưa phải detector.

---

## 6. Câu nhớ nhanh

Nếu chỉ nhớ một câu, nhớ câu này:

> Notebook ver3 huấn luyện MobileNetV2 để phân loại 12 loại biển báo từ ảnh đã crop; dữ liệu được split train/val/test, train có thể dùng augmentation online, input được tensor hóa và normalize, model được đánh giá bằng accuracy/loss/confusion matrix, còn ảnh ngoài cần crop hoặc detect trước khi phân loại.
