# 2. Nền Tảng Lý Thuyết

---

## 2.1. Tích Chập Truyền Thống (Standard Convolution)

### Định nghĩa

Tích chập tiêu chuẩn áp dụng một tập kernel lên toàn bộ không gian **và** toàn bộ kênh cùng lúc. Với đầu vào kích thước **H × W × C_in**, kernel **k × k**, ta thu được đầu ra **H × W × C_out**.

### Công thức tính MACs

$$\text{MACs}_{\text{std}} = k^2 \times C_{in} \times C_{out} \times H \times W$$

> **MACs** (Multiply-Accumulate Operations) = số phép nhân + cộng — thước đo chi phí tính toán phổ biến trong deep learning.

### Ví dụ minh họa

| Tham số        | Giá trị       |
|----------------|---------------|
| Kernel size    | 3 × 3         |
| Input          | 112 × 112 × 32|
| Output         | 112 × 112 × 64|
| **MACs**       | **3² × 32 × 64 × 112 × 112 ≈ 231 triệu** |

### Trực quan hoá

```
Input:  [H × W × C_in]
         ↓  ↓  ↓  ↓  ↓   ← mỗi vị trí (h,w) nhìn toàn bộ C_in kênh
       ┌───────────────┐
       │  kernel k×k   │  × C_out bộ lọc
       └───────────────┘
Output: [H × W × C_out]
```

Mỗi neuron đầu ra phụ thuộc vào **k² × C_in** giá trị → rất tốn kém khi C_in, C_out lớn.

### Nhược điểm

- Chi phí tính toán **O(k² · C_in · C_out · H · W)** — tăng bậc hai khi tăng số kênh.
- Với mạng di động cần C_out lớn (64, 128, 256…), chi phí nhanh chóng vượt ngưỡng thiết bị nhúng.

---

## 2.2. Depthwise Separable Convolution

Ý tưởng cốt lõi: **tách** tích chập tiêu chuẩn thành 2 bước nhỏ hơn rất nhiều.

---

### Bước 1 — Depthwise Convolution (DW Conv)

Mỗi kênh đầu vào được xử lý **độc lập** bởi một kernel k × k riêng.

```
C_in kênh đầu vào:
  Kênh 1 → kernel₁ (3×3) → Feature map 1
  Kênh 2 → kernel₂ (3×3) → Feature map 2
     ⋮              ⋮              ⋮
  Kênh C → kernelC (3×3) → Feature map C

Output: H × W × C_in  (số kênh KHÔNG đổi)
```

**MACs của DW Conv:**

$$\text{MACs}_{\text{DW}} = k^2 \times C_{in} \times H \times W$$

> Không có chiều C_out ở đây — mỗi kênh chỉ dùng đúng 1 kernel.

---

### Bước 2 — Pointwise Convolution (PW Conv / 1×1 Conv)

Dùng kernel **1 × 1** để trộn thông tin giữa các kênh, tạo ra C_out kênh mới.

```
Input: H × W × C_in
  ↓  kernel 1×1 (× C_out bộ)
Output: H × W × C_out
```

**MACs của PW Conv:**

$$\text{MACs}_{\text{PW}} = 1^2 \times C_{in} \times C_{out} \times H \times W = C_{in} \times C_{out} \times H \times W$$

---

### So sánh tổng MACs

| Loại Conv          | Công thức MACs                           |
|--------------------|------------------------------------------|
| Standard Conv      | $k^2 \cdot C_{in} \cdot C_{out} \cdot H \cdot W$ |
| Depthwise Separable| $k^2 \cdot C_{in} \cdot H \cdot W + C_{in} \cdot C_{out} \cdot H \cdot W$ |

**Tỉ lệ giảm:**

$$\frac{\text{MACs}_{\text{DS}}}{\text{MACs}_{\text{Std}}} = \frac{1}{C_{out}} + \frac{1}{k^2}$$

Với k = 3, C_out = 64:

$$= \frac{1}{64} + \frac{1}{9} \approx 0.127 \quad \Rightarrow \text{ giảm } \approx \mathbf{8\text{–}9 \times}$$

> **Insight**: Khi C_out đủ lớn, số hạng 1/C_out nhỏ không đáng kể, mức giảm chủ yếu đến từ 1/k² ≈ 1/9.

---

## 2.3. Residual Connection (Skip Connection)

### Nguồn gốc

Được giới thiệu trong **ResNet** (He et al., 2016, "Deep Residual Learning for Image Recognition"). Ý tưởng: thay vì học trực tiếp H(x), mạng học **phần dư** F(x) = H(x) − x, sau đó cộng lại:

$$\text{Output} = F(x) + x$$

### Minh họa

```
    x ──────────────────────────┐
    │                           │  (skip / identity)
    ↓                           │
  Conv → BN → ReLU             │
    ↓                           │
  Conv → BN                    │
    ↓                           │
  (+) ←──────────────────────────┘
    ↓
  ReLU
  Output = F(x) + x
```

### Tại sao cần skip connection?

| Vấn đề                  | Giải thích                                                                 |
|-------------------------|----------------------------------------------------------------------------|
| **Vanishing gradient**  | Khi mạng rất sâu, gradient bị nhân liên tiếp với số < 1 → tiệm cận 0     |
| **Giải pháp skip**      | Gradient lan ngược qua đường tắt **cộng** → đạo hàm luôn ≥ 1             |
| **Degradation problem** | Mạng sâu hơn đôi khi kém hơn mạng nông — skip connection giải quyết điều này |

**Đạo hàm với skip connection:**

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial \text{Out}} \cdot \left(\frac{\partial F(x)}{\partial x} + 1\right)$$

Số hạng `+1` đảm bảo gradient không về 0.

### Điều kiện áp dụng skip

```
✅ stride = 1  VÀ  C_in = C_out   → dùng identity shortcut (cộng trực tiếp)
⚠️ stride ≠ 1  HOẶC  C_in ≠ C_out → dùng 1×1 Conv để điều chỉnh kích thước
```

---

## 2.4. Bottleneck: ResNet vs MobileNetV2

### ResNet Bottleneck — "Wide → Narrow → Wide"

```
Input:  [ · · · · · · · · ]  256 kênh (wide)
            ↓  1×1 Conv
         [ · · · ]           64 kênh  (narrow) ← thu hẹp
            ↓  3×3 Conv
         [ · · · ]           64 kênh  (narrow)
            ↓  1×1 Conv
        [ · · · · · · · · ]  256 kênh (wide) ← mở rộng lại
```

> Mục tiêu: giảm chi phí tính toán của Conv 3×3 bằng cách thu hẹp kênh trước.

---

### MobileNetV2 Inverted Bottleneck — "Narrow → Wide → Narrow"

```
Input:  [ · · ]              24 kênh  (narrow)
            ↓  1×1 Conv (Expansion, t=6)
        [ · · · · · · · ]   144 kênh  (wide)  ← MỞ RỘNG
            ↓  3×3 DW Conv
        [ · · · · · · · ]   144 kênh  (wide)
            ↓  1×1 Conv (Projection, NO ReLU)
         [ · · ]              24 kênh  (narrow) ← thu hẹp lại
```

### Tại sao đảo ngược?

Depthwise Convolution **không trộn kênh** — nó chỉ lọc không gian. Để DW Conv có đủ thông tin để lọc hiệu quả, **không gian biểu diễn (số kênh) phải đủ lớn** tại bước đó.

| Tiêu chí               | ResNet Bottleneck | MobileNetV2 Inverted Bottleneck |
|------------------------|-------------------|----------------------------------|
| Cấu trúc               | Wide → Narrow → Wide | Narrow → Wide → Narrow        |
| Conv chính             | Regular Conv 3×3  | **Depthwise Conv 3×3**          |
| Mục tiêu thu hẹp       | Tiết kiệm cho Conv nặng | Bottleneck đầu/cuối nhẹ   |
| Skip connection        | Ở ngoài (wide)    | Ở ngoài (narrow)                |

---

## 2.5. Linear Bottleneck — Lý Thuyết Manifold

### Giả thuyết Manifold

Paper gốc (Sandler et al., 2018) lập luận:

> *Tập hợp các activation có "ý nghĩa" trong một mạng thực ra nằm trên một **manifold chiều thấp** nhúng trong không gian chiều cao.*

Nói đơn giản: dù tensor có 144 kênh, thông tin thực sự chỉ chiếm một không gian con nhiều chiều nhỏ hơn nhiều.

### Vấn đề: ReLU phá hủy thông tin ở chiều thấp

```
Giả sử thông tin nằm trên manifold 2D trong không gian 3D:

Trước ReLU:  z = [ 1.2,  -0.5,  0.8]
Sau  ReLU:   z = [ 1.2,   0.0,  0.8]   ← thông tin âm bị xóa hoàn toàn
```

Khi **không gian đủ lớn** (nhiều kênh): ReLU chỉ triệt tiêu một phần nhỏ → thông tin vẫn được bảo toàn qua các kênh khác.

Khi **không gian quá nhỏ** (ít kênh, như tầng projection): ReLU có thể triệt tiêu quá nhiều → **mất thông tin không thể phục hồi**.

### Bằng chứng từ paper gốc

Sandler et al. thực nghiệm embed dữ liệu vào không gian n chiều rồi project về, so sánh có và không có ReLU:

```
n = 2–3:  Thông tin bị phá hủy nghiêm trọng khi có ReLU
n = 15–30: ReLU gần như không gây mất thông tin
```

### Giải pháp: Linear Bottleneck

```
Block cuối:
  ...
  ↓  1×1 Conv (Projection)  → 24 kênh
  ❌ KHÔNG có ReLU ở đây
  ↓
  (+) skip connection
```

**Quy tắc**: bỏ ReLU sau lớp Projection (1×1 Conv cuối cùng trong block) để bảo toàn thông tin trên manifold chiều thấp.

---

## 2.6. Hàm Kích Hoạt ReLU6

### Định nghĩa

$$\text{ReLU6}(x) = \min(\max(0,\ x),\ 6)$$

### Đồ thị so sánh (mô tả)

```
Giá trị y
  6 |          ┌──────────────────  ← ReLU6 (giới hạn tại 6)
    |         /
    |        /
    |       /
  0 |──────/                        ← ReLU (không giới hạn)
    └──────────────────────── x
        0   1   2   3   4   5   6
```

### Tại sao giới hạn tại 6?

Khi **quantize** mô hình sang **INT8** (số nguyên 8-bit), giá trị activation cần được ánh xạ vào khoảng [0, 255]:

```
Quantize formula:  x_int8 = round(x / scale)
                   scale  = max_value / 255

Với ReLU  (không giới hạn): max_value có thể = 30, 50, 100... → scale lớn → độ phân giải thấp
Với ReLU6 (giới hạn tại 6):  max_value = 6 (cố định)          → scale nhỏ → độ phân giải tốt hơn
```

> Giới hạn tại 6 cụ thể vì: thực nghiệm cho thấy hầu hết activation có ích nằm trong [0, 6].

### So sánh các hàm kích hoạt

| Hàm kích hoạt | Công thức                            | Ưu điểm                        | Nhược điểm                          |
|---------------|--------------------------------------|--------------------------------|-------------------------------------|
| **ReLU**      | max(0, x)                            | Đơn giản, nhanh                | Dying ReLU; khó quantize            |
| **ReLU6**     | min(max(0, x), 6)                    | Quantize tốt; ổn định          | Mất thông tin x > 6 (hiếm gặp)     |
| **Swish**     | x · σ(x)                             | Smooth; hiệu năng tốt hơn ReLU | Tốn kém hơn (sigmoid)              |
| **Hard-Swish**| x · ReLU6(x+3)/6                    | Xấp xỉ Swish; friendly với HW  | Phức tạp hơn ReLU6 chút            |

> MobileNetV2 dùng **ReLU6**. MobileNetV3 nâng cấp lên **Hard-Swish**.

---

## 2.7. Batch Normalization (BN)

### Định nghĩa

Với mini-batch B = {x₁, x₂, ..., xₘ}, BN chuẩn hóa từng feature dimension:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

Trong đó γ, β là các tham số học được (scale và shift).

### Vai trò trong MobileNetV2

BN được đặt **sau mỗi lớp Conv**, trước hàm kích hoạt:

```
Conv → BN → ReLU6     (tầng Expansion và DW)
Conv → BN             (tầng Projection — KHÔNG có ReLU6)
```

### Lợi ích

| Lợi ích                    | Giải thích                                                    |
|----------------------------|---------------------------------------------------------------|
| **Ổn định huấn luyện**     | Giữ phân phối activation ổn định qua các lớp                 |
| **Cho phép learning rate cao** | Gradient không explode/vanish dễ dàng                    |
| **Regularization nhẹ**     | Noise từ batch statistics giúp tránh overfitting             |
| **Giảm phụ thuộc vào init**| Khởi tạo trọng số ít quan trọng hơn                         |

### Tương tác với Linear Bottleneck

Câu hỏi thường gặp: **có BN trước skip add không?**

```
Cấu trúc đầy đủ của Inverted Residual Block:

Input ──────────────────────────────────┐
  ↓                                     │  (skip)
1×1 Conv → BN → ReLU6   (Expansion)   │
  ↓                                     │
3×3 DW Conv → BN → ReLU6              │
  ↓                                     │
1×1 Conv → BN           (Projection)  │  ← BN có, ReLU6 KHÔNG
  ↓                                     │
(+) ←───────────────────────────────────┘
  ↓
Output  (KHÔNG có thêm activation nào sau add)
```

**Tại sao không có ReLU sau skip add?**

- Nếu thêm ReLU sau phép cộng → triệt tiêu giá trị âm → phá hủy thông tin identity path.
- Linear Bottleneck yêu cầu **tuyến tính** tại điểm hợp nhất để bảo toàn manifold.

---

## Tóm Tắt Chương 2

```
Standard Conv (nặng)
      ↓ tách thành
DW Conv + PW Conv = Depthwise Separable Conv (~8–9× nhẹ hơn)
      ↓ kết hợp với
Skip Connection (từ ResNet) = chống vanishing gradient
      ↓ đảo ngược thành
Inverted Bottleneck (narrow→wide→narrow) = DW ở không gian rộng
      ↓ bỏ ReLU cuối vì
Linear Bottleneck = bảo toàn manifold chiều thấp
      ↓ dùng
ReLU6 = ổn định quantization INT8
      ↓ chuẩn hóa bằng
Batch Normalization = ổn định huấn luyện, không áp dụng trước skip add
```

> Toàn bộ những yếu tố trên kết hợp lại tạo nên kiến trúc **MobileNetV2** — hiệu quả, nhẹ, và thân thiện với thiết bị nhúng.

---

*Tài liệu tham khảo chính:*
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR 2018*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
- Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. *arXiv:1704.04861*.
