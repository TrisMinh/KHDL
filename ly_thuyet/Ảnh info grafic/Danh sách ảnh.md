# Danh sách ảnh cần vẽ để trực quan hóa phần lý thuyết MobileNetV2

Tài liệu này tổng hợp toàn bộ các hình minh họa nên có cho báo cáo/thuyết trình dự án MobileNetV2 phân loại biển báo giao thông.

Các hình được nhóm theo đúng flow lý thuyết trong notebook:

```text
Bài toán
→ Pipeline dữ liệu
→ Transform & Augmentation
→ MobileNetV2
→ Loss/Optimizer/Scheduler
→ Training loop
→ Evaluation
→ Inference ảnh ngoài
```

---

# PHẦN 1 — BÀI TOÁN VÀ TỔNG QUAN

## 1.1. Hình tổng quan bài toán classification

### Nội dung hình

Flow:

```text
Ảnh biển báo đã crop
→ MobileNetV2
→ output class
```

Ví dụ:

```text
ảnh stop sign → stop_sign
ảnh no_entry → no_entry
```

### Mục đích

Giải thích:

* bài toán hiện tại là classification;
* model chỉ phân loại;
* không detect vị trí biển báo.

---

## 1.2. Hình classification vs detection

### Nội dung hình

So sánh 2 cột:

#### Classification

```text
Input: ảnh đã crop biển báo
Output: class
```

#### Detection

```text
Input: ảnh nguyên cảnh
Output: bounding box + class
```

### Nên vẽ

* ảnh đường phố;
* box quanh biển báo;
* classifier output;
* detector output.

### Mục đích

Giải thích vì sao phải crop ảnh trước khi predict.

---

## 1.3. Hình distribution shift

### Nội dung hình

Hai nhóm:

```text
Train distribution:
ảnh crop sạch
```

```text
Inference distribution:
ảnh nguyên cảnh nhiều nền
```

### Mục đích

Giải thích:

* vì sao ảnh ngoài dễ sai;
* model học trên ảnh crop.

---

# PHẦN 2 — PIPELINE DỮ LIỆU

## 2.1. Hình pipeline dữ liệu tổng quát

### Nội dung hình

```text
Data
→ FilterData
→ CropData
→ ResizeData
→ RGBData
→ SplitData
→ Train
```

### Nên vẽ

Mỗi bước có:

* icon folder;
* ví dụ ảnh.

### Mục đích

Cho thấy toàn bộ quá trình preprocessing.

---

## 2.2. Hình crop từ ảnh gốc

### Nội dung hình

Bên trái:

* ảnh đường phố gốc.

Bên phải:

* ảnh crop biển báo.

Có bounding box minh họa.

### Mục đích

Giải thích metadata crop.

---

## 2.3. Hình metadata bounding box

### Nội dung hình

Một bảng:

```text
image_name | x1 | y1 | x2 | y2
```

Kèm hình minh họa tọa độ trên ảnh.

### Mục đích

Giải thích CSV metadata dùng để crop.

---

## 2.4. Hình cấu trúc SplitData

### Nội dung hình

Tree folder:

```text
SplitData/
  train/
  val/
  test/
```

Bên trong mỗi split:

```text
no_entry/
stop_sign/
...
```

### Mục đích

Giải thích cách ImageFolder đọc label.

---

## 2.5. Hình ImageFolder mapping class

### Nội dung hình

```text
Folder name → class index
```

Ví dụ:

```text
no_entry → 0
stop_sign → 1
parking → 2
```

### Mục đích

Giải thích mapping class_to_idx.

---

## 2.6. Hình train/val/test split

### Nội dung hình

Pie chart:

```text
70% train
15% val
15% test
```

### Mục đích

Giải thích tỷ lệ split.

---

## 2.7. Hình data leakage

### Nội dung hình

Ảnh gốc:

```text
A.jpg
```

Augment:

```text
A_0001.jpg
A_0002.jpg
```

Sai:

```text
train chứa A_0001
val chứa A_0002
```

### Mục đích

Giải thích leakage do split sau augment.

---

## 2.8. Hình quy trình augment đúng vs sai

### Nội dung hình

#### Đúng

```text
split
→ augment train only
```

#### Sai

```text
augment all
→ split
```

### Mục đích

Giải thích vì sao phải split trước augment.

---

# PHẦN 3 — TRANSFORM, NORMALIZE, AUGMENT

## 3.1. Hình transform pipeline

### Nội dung hình

```text
PIL Image
→ Resize
→ ToTensor
→ Normalize
→ Tensor [3,224,224]
```

### Mục đích

Giải thích transform flow.

---

## 3.2. Hình ToTensor

### Nội dung hình

#### Trước

```text
H x W x C
0..255
```

#### Sau

```text
C x H x W
0..1
```

### Mục đích

Giải thích tensor conversion.

---

## 3.3. Hình normalize Z-score

### Nội dung hình

Công thức:

```text
x_norm = (x - mean) / std
```

Có biểu đồ:

* trước normalize;
* sau normalize.

### Mục đích

Giải thích mean/std.

---

## 3.4. Hình train transform vs val transform

### Nội dung hình

Hai pipeline song song:

#### Train

```text
augment
→ tensor
→ normalize
```

#### Val/Test

```text
tensor
→ normalize
```

### Mục đích

Giải thích val/test không augment.

---

## 3.5. Hình augmentation gallery

### Nội dung hình

Grid:

```text
original
rotation
crop
perspective
blur
brightness
random erasing
```

### Mục đích

Trực quan hóa augment.

---

## 3.6. Hình RandomCrop

### Nội dung hình

* ảnh gốc;
* nhiều crop khác nhau.

### Mục đích

Giải thích crop lệch nhẹ.

---

## 3.7. Hình RandomRotation

### Nội dung hình

* xoay -20°;
* xoay +20°.

### Mục đích

Giải thích robustness với camera nghiêng.

---

## 3.8. Hình RandomAffine

### Nội dung hình

Ví dụ:

* translate;
* scale;
* shear.

### Mục đích

Giải thích affine transform.

---

## 3.9. Hình RandomPerspective

### Nội dung hình

Ảnh trước và sau perspective distortion.

### Mục đích

Mô phỏng góc nhìn chéo.

---

## 3.10. Hình ColorJitter

### Nội dung hình

Một hàng ảnh:

* sáng hơn;
* tối hơn;
* contrast cao;
* saturation khác.

### Mục đích

Giải thích robustness ánh sáng.

---

## 3.11. Hình GaussianBlur

### Nội dung hình

* ảnh nét;
* ảnh blur.

### Mục đích

Mô phỏng rung/mờ camera.

---

## 3.12. Hình RandomErasing

### Nội dung hình

Ảnh bị che một vùng.

### Mục đích

Giải thích chống phụ thuộc cục bộ.

---

# PHẦN 4 — MOBILE NET V2

## 4.1. Hình CNN nhìn ảnh như thế nào

### Nội dung hình

Nhiều layer:

```text
cạnh
→ hình dạng
→ ký hiệu biển báo
→ class
```

### Mục đích

Giải thích feature hierarchy.

---

## 4.2. Hình convolution thường

### Nội dung hình

Kernel 3x3 trượt trên ảnh.

### Mục đích

Giải thích convolution.

---

## 4.3. Hình regular conv vs depthwise separable conv

### Nội dung hình

#### Regular conv

```text
3x3 conv trực tiếp
```

#### MobileNet

```text
Depthwise
→ Pointwise 1x1
```

### Mục đích

Giải thích MobileNet nhẹ hơn.

---

## 4.4. Hình depthwise convolution

### Nội dung hình

Mỗi channel có kernel riêng.

### Mục đích

Giải thích groups=hidden_dim.

---

## 4.5. Hình pointwise convolution 1x1

### Nội dung hình

Mix channel bằng conv 1x1.

### Mục đích

Giải thích trộn thông tin giữa channel.

---

## 4.6. Hình Conv → BN → ReLU6

### Nội dung hình

Block:

```text
Conv2D
→ BatchNorm
→ ReLU6
```

### Mục đích

Giải thích ConvBNReLU6.

---

## 4.7. Hình ReLU vs ReLU6

### Nội dung hình

Hai đồ thị activation.

### Mục đích

Giải thích clamp tại 6.

---

## 4.8. Hình inverted residual block

### Nội dung hình

```text
narrow
→ expand
→ depthwise
→ project
→ narrow
```

### Mục đích

Đây là hình rất quan trọng.

---

## 4.9. Hình ResNet residual vs inverted residual

### Nội dung hình

#### ResNet

```text
wide → narrow → wide
```

#### MobileNetV2

```text
narrow → wide → narrow
```

### Mục đích

Giải thích chữ “inverted”.

---

## 4.10. Hình expansion layer

### Nội dung hình

Ví dụ:

```text
32 channels
→ 192 channels
```

### Mục đích

Giải thích expand ratio.

---

## 4.11. Hình linear bottleneck

### Nội dung hình

Projection layer không có ReLU.

### Mục đích

Giải thích tránh mất thông tin.

---

## 4.12. Hình skip connection

### Nội dung hình

```text
x
↘
  F(x) + x
↗
```

### Mục đích

Giải thích gradient flow.

---

## 4.13. Hình feature map size reduction

### Nội dung hình

```text
224
→ 112
→ 56
→ 28
→ 14
→ 7
```

Kèm số channel tăng.

### Mục đích

Giải thích stride.

---

## 4.14. Hình adaptive average pooling

### Nội dung hình

```text
[B,C,H,W]
→
[B,C,1,1]
```

### Mục đích

Giải thích pooling cuối.

---

## 4.15. Hình classifier cuối

### Nội dung hình

```text
Feature vector
→ Dropout
→ Linear
→ logits
```

### Mục đích

Giải thích output class.

---

# PHẦN 5 — LOSS, OPTIMIZER, SCHEDULER

## 5.1. Hình logits → softmax → probabilities

### Nội dung hình

```text
[2.1, -0.4, 5.2]
→ softmax
→ [0.04, 0.01, 0.95]
```

### Mục đích

Giải thích logits.

---

## 5.2. Hình cross entropy

### Nội dung hình

Class đúng:

```text
loss thấp nếu p đúng cao
```

Class sai:

```text
loss cao nếu p đúng thấp
```

### Mục đích

Giải thích loss function.

---

## 5.3. Hình label smoothing

### Nội dung hình

#### One-hot

```text
[0,0,1,0]
```

#### Smoothed

```text
[0.03,0.03,0.9,0.03]
```

### Mục đích

Giải thích giảm overconfidence.

---

## 5.4. Hình SGD update

### Nội dung hình

Mũi tên gradient descent trên bề mặt loss.

### Mục đích

Giải thích optimizer.

---

## 5.5. Hình momentum

### Nội dung hình

Ví dụ quả bóng lăn xuống valley.

### Mục đích

Giải thích quán tính.

---

## 5.6. Hình weight decay

### Nội dung hình

So sánh:

* weight lớn;
* weight regularized.

### Mục đích

Giải thích chống overfit.

---

## 5.7. Hình learning rate warmup

### Nội dung hình

Biểu đồ LR:

```text
0.002
0.004
0.006
0.008
0.01
```

### Mục đích

Giải thích warmup.

---

## 5.8. Hình cosine annealing

### Nội dung hình

Đường cosine giảm dần.

### Mục đích

Giải thích scheduler.

---

# PHẦN 6 — TRAINING LOOP

## 6.1. Hình training loop tổng quát

### Nội dung hình

```text
for epoch:
    train
    validate
    save checkpoint
```

### Mục đích

Cho thấy flow training.

---

## 6.2. Hình một batch training

### Nội dung hình

```text
images
→ forward
→ loss
→ backward
→ optimizer.step
```

### Mục đích

Giải thích batch update.

---

## 6.3. Hình forward vs backward

### Nội dung hình

* forward direction;
* backward gradient flow.

### Mục đích

Giải thích backpropagation.

---

## 6.4. Hình mixed precision

### Nội dung hình

```text
FP32 vs FP16
```

VRAM giảm.

### Mục đích

Giải thích autocast + GradScaler.

---

## 6.5. Hình gradient clipping

### Nội dung hình

Gradient lớn bị clip.

### Mục đích

Giải thích ổn định training.

---

## 6.6. Hình checkpoint system

### Nội dung hình

```text
best_model.pth
checkpoint_latest.pth
checkpoint_epoch_10.pth
```

### Mục đích

Giải thích save/resume.

---

## 6.7. Hình early stopping

### Nội dung hình

Val acc không tăng nhiều epoch → stop.

### Mục đích

Giải thích patience.

---

# PHẦN 7 — EVALUATION

## 7.1. Hình accuracy formula

### Nội dung hình

```text
correct / total
```

### Mục đích

Giải thích accuracy.

---

## 7.2. Hình train vs val curves

### Nội dung hình

Biểu đồ:

* train loss;
* val loss;
* train acc;
* val acc.

### Mục đích

Giải thích learning dynamics.

---

## 7.3. Hình overfit

### Nội dung hình

```text
train loss giảm
val loss tăng
```

### Mục đích

Giải thích overfitting.

---

## 7.4. Hình underfit

### Nội dung hình

```text
train acc thấp
val acc thấp
```

### Mục đích

Giải thích underfitting.

---

## 7.5. Hình confusion matrix

### Nội dung hình

Heatmap confusion matrix.

### Mục đích

Giải thích class confusion.

---

## 7.6. Hình per-class accuracy

### Nội dung hình

Bar chart accuracy từng class.

### Mục đích

So sánh class mạnh/yếu.

---

## 7.7. Hình precision/recall/F1

### Nội dung hình

TP / FP / FN diagram.

### Mục đích

Giải thích metric.

---

## 7.8. Hình misclassified samples

### Nội dung hình

Gallery ảnh dự đoán sai.

### Mục đích

Phân tích lỗi model.

---

## 7.9. Hình top-k prediction

### Nội dung hình

```text
1. no_entry 95%
2. no_stopping 3%
3. parking 1%
```

### Mục đích

Giải thích confidence.

---

# PHẦN 8 — INFERENCE ẢNH NGOÀI

## 8.1. Hình inference pipeline

### Nội dung hình

```text
Ảnh ngoài
→ crop
→ resize
→ normalize
→ model
→ prediction
```

### Mục đích

Giải thích inference flow.

---

## 8.2. Hình crop đúng vs crop sai

### Nội dung hình

#### Đúng

* biển lớn;
* đủ biển.

#### Sai

* crop quá sát;
* crop quá rộng;
* mất biển.

### Mục đích

Giải thích crop hợp lý.

---

## 8.3. Hình resize nguyên cảnh gây lỗi

### Nội dung hình

```text
1920x1080
→ resize 224x224
→ biển quá nhỏ
```

### Mục đích

Giải thích vì sao classifier sai trên ảnh nguyên cảnh.

---

## 8.4. Hình RGB vs BGR

### Nội dung hình

So sánh màu bị đảo.

### Mục đích

Giải thích lỗi OpenCV.

---

## 8.5. Hình confidence cao nhưng sai

### Nội dung hình

Ví dụ:

```text
unknown sign
→ predicted speed_limit_50 97%
```

### Mục đích

Giải thích closed-set classification.

---

# PHẦN 9 — HÌNH CHO THUYẾT TRÌNH / BẢO VỆ

## 9.1. Hình kiến trúc tổng hệ thống

### Nội dung hình

```text
Dataset
→ Preprocessing
→ MobileNetV2
→ Evaluation
→ Inference
```

### Mục đích

Hình tổng quan mở đầu báo cáo.

---

## 9.2. Hình so sánh ảnh gốc và ảnh augment

### Nội dung hình

Before/After.

### Mục đích

Trình bày preprocessing.

---

## 9.3. Hình best predictions

### Nội dung hình

Các ảnh predict đúng với confidence cao.

### Mục đích

Demo model hoạt động tốt.

---

## 9.4. Hình failure cases

### Nội dung hình

Các ảnh khó:

* mờ;
* tối;
* lệch;
* biển nhỏ.

### Mục đích

Phân tích giới hạn mô hình.

---

## 9.5. Hình bảng kết quả cuối

### Nội dung hình

Bảng:

```text
Accuracy
Precision
Recall
F1-score
```

### Mục đích

Tổng hợp kết quả báo cáo.

---

# DANH SÁCH HÌNH QUAN TRỌNG NHẤT NÊN ƯU TIÊN

Nếu thời gian ít, ưu tiên vẽ trước:

1. Classification vs Detection
2. Pipeline dữ liệu
3. Train/Val/Test split
4. Data leakage
5. Transform pipeline
6. Augmentation gallery
7. Depthwise separable convolution
8. Inverted residual block
9. Skip connection
10. Feature map reduction
11. Cross entropy + softmax
12. Warmup + cosine scheduler
13. Training loop
14. Overfit curve
15. Confusion matrix
16. Inference pipeline
17. Crop đúng vs crop sai
18. Kiến trúc tổng hệ thống

---

# GỢI Ý PHONG CÁCH VẼ

## Nên dùng

* PowerPoint
* Figma
* Canva
* draw.io
* Excalidraw

## Style nên thống nhất

* màu xanh cho pipeline đúng;
* màu đỏ cho lỗi;
* icon folder cho dataset;
* icon GPU/model cho training;
* mũi tên rõ ràng;
* ít chữ, nhiều trực quan.

---

# GỢI Ý CHIA HÌNH THEO CHƯƠNG BÁO CÁO

## Chương dữ liệu

* pipeline data;
* crop;
* split;
* leakage.

## Chương preprocessing

* transform;
* normalize;
* augment.

## Chương mô hình

* convolution;
* depthwise;
* inverted residual;
* skip connection.

## Chương training

* optimizer;
* scheduler;
* training loop;
* checkpoint.

## Chương đánh giá

* curves;
* confusion matrix;
* per-class accuracy;
* ảnh sai.

## Chương demo

* inference pipeline;
* crop đúng/sai;
* ảnh ngoài.
