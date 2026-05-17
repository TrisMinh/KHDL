# CELL 14: MODEL ANALYSIS & EXPORT

## 1. Thống kê Model

```python
total_params = sum(p.numel() for p in model.parameters())
# ~2.2M parameters
```

## 2. Phân bố Parameters

| Component | Params | Tỷ lệ |
|:---|:---|:---|
| First Conv (3→32) | ~864 | 0.04% |
| Inverted Residuals | ~1.7M | 77% |
| Last Conv (320→1280) | ~410K | 19% |
| Classifier (1280→43) | ~55K | 2.5% |

**Insight:** 77% params nằm ở backbone → phần lớn "trí tuệ" dùng cho feature extraction, chỉ 2.5% cho quyết định phân loại.

## 3. Model Size

```
FP32: 2.2M × 4 bytes = ~9 MB
FP16: 2.2M × 2 bytes = ~4.5 MB
INT8: 2.2M × 1 byte  = ~2.2 MB (quantized)
```

So sánh: VGG16 = 528 MB, ResNet50 = 98 MB → MobileNetV2 nhỏ hơn **58× so với VGG16**.

## 4. FLOPs Analysis
- MobileNetV2 (96×96): ~100M FLOPs
- VGG16 (224×224): ~15,000M FLOPs
- MobileNetV2 nhanh hơn **150×** trên cùng phần cứng

## 5. Tác dụng
Cell này cung cấp **bằng chứng định lượng** cho lợi thế của MobileNetV2: nhỏ, nhanh, phù hợp mobile.
