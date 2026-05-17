# CELL 5: VISUALIZE DỮ LIỆU

## 1. Nội dung

Cell này tạo 2 biểu đồ:
1. **Grid 4×8 ảnh mẫu** — hiển thị 32 ảnh biển báo từ test set
2. **Bar chart phân bố lớp** — số lượng ảnh mỗi class

## 2. Inverse Normalize

```python
inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(MEAN, STD)],
    std=[1/s for s in STD]
)
```

**Tại sao cần?** Ảnh đã normalize (mean=0, std=1) → pixel values âm, >1 → matplotlib không hiển thị đúng màu. Inverse normalize đưa về [0,1] để hiển thị đúng.

**Công thức:**
```
normalize:     x_norm = (x - mean) / std
inv_normalize: x_orig = x_norm × std + mean = (x_norm - (-mean/std)) / (1/std)
```

## 3. Phân bố lớp (Class Distribution)

```python
label_counts = Counter(all_train_labels)
ax.bar(range(43), [label_counts.get(i, 0) for i in range(43)])
```

**Tại sao cần biết phân bố?**
- **Class imbalance:** Nếu class A có 2000 ảnh, class B có 200 → model bias về A
- **Giải pháp tiềm năng:** Weighted sampling, class-weighted loss, oversampling
- **Trong project này:** Dùng label smoothing để giảm bias (không dùng weighted sampling)

**Kết quả quan sát:**
- Min: ~210 ảnh (Class 0: Speed limit 20km/h)
- Max: ~2250 ảnh (Class 2: Speed limit 50km/h)
- Tỷ lệ chênh ~10:1

## 4. Tác dụng tổng thể
Cell này đóng vai trò **Exploratory Data Analysis (EDA)** — bước quan trọng trước khi training để hiểu data. Trong báo cáo, các biểu đồ này thể hiện sự hiểu biết về dataset.
