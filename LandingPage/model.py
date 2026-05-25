import io
from pathlib import Path

from PIL import Image


IMG_SIZE = 224
MEAN = [0.5734, 0.4815, 0.5165]
STD = [0.3181, 0.3130, 0.3367]
CLASS_NAMES = [
    "no_entry",
    "no_stopping",
    "no_vehicles",
    "parking",
    "priority_road",
    "roadworks",
    "roundabout",
    "speed_limit_30",
    "speed_limit_40",
    "speed_limit_50",
    "speed_limit_60",
    "stop_sign",
]


def _load_torch():
    try:
        import torch
        import torch.nn as nn
        import torchvision.transforms as transforms
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing model dependency. Install requirements first: "
            "pip install -r requirements.txt"
        ) from exc
    return torch, nn, transforms


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def build_model(num_classes=12, width_mult=1.0, dropout=0.2,
                use_attention=True, attention_reduction=4):
    torch, nn, _ = _load_torch()

    class ConvBNReLU6(nn.Sequential):
        def __init__(self, in_channels, out_channels, kernel_size=3,
                     stride=1, groups=1):
            padding = (kernel_size - 1) // 2
            super().__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                          padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True),
            )

    class SEBlock(nn.Module):
        def __init__(self, channels, reduction=4):
            super().__init__()
            reduced_channels = max(8, channels // reduction)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Conv2d(channels, reduced_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduced_channels, channels, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return x * self.fc(self.pool(x))

    class InvertedResidual(nn.Module):
        def __init__(self, in_channels, out_channels, stride, expand_ratio):
            super().__init__()
            hidden_dim = int(round(in_channels * expand_ratio))
            self.use_skip = stride == 1 and in_channels == out_channels

            layers = []
            if expand_ratio != 1:
                layers.append(ConvBNReLU6(in_channels, hidden_dim, kernel_size=1))
            layers.append(ConvBNReLU6(hidden_dim, hidden_dim, kernel_size=3,
                                      stride=stride, groups=hidden_dim))
            if use_attention:
                layers.append(SEBlock(hidden_dim, reduction=attention_reduction))
            layers.extend([
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            ])
            self.conv = nn.Sequential(*layers)

        def forward(self, x):
            if self.use_skip:
                return x + self.conv(x)
            return self.conv(x)

    class MobileNetV2(nn.Module):
        def __init__(self):
            super().__init__()
            inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

            input_channels = _make_divisible(32 * width_mult)
            last_channels = _make_divisible(1280 * max(1.0, width_mult))
            features = [ConvBNReLU6(3, input_channels, kernel_size=3, stride=2)]

            for t, c, n, s in inverted_residual_setting:
                output_channels = _make_divisible(c * width_mult)
                for i in range(n):
                    stride = s if i == 0 else 1
                    features.append(InvertedResidual(input_channels,
                                                     output_channels,
                                                     stride, t))
                    input_channels = output_channels

            features.append(ConvBNReLU6(input_channels, last_channels,
                                        kernel_size=1))
            self.features = nn.Sequential(*features)
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(last_channels, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            return self.classifier(x)

    return MobileNetV2()


class TrafficSignPredictor:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self._model = None
        self._device = None
        self._transform = None

    def load(self):
        if self._model is not None:
            return

        torch, _, transforms = _load_torch()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        checkpoint = torch.load(self.model_path, map_location=self._device)
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("model_state_dict")
                or checkpoint.get("state_dict")
                or checkpoint
            )

        model = build_model(num_classes=len(CLASS_NAMES))
        cleaned = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
            if hasattr(value, "shape")
        }
        model.load_state_dict(cleaned, strict=True)
        model.to(self._device)
        model.eval()
        self._model = model

    def predict(self, image, crop_box=None, topk=5):
        torch, _, _ = _load_torch()
        self.load()

        image = image.convert("RGB")
        if crop_box:
            image = image.crop(crop_box)

        tensor = self._transform(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            top_p, top_i = torch.topk(probs, k=topk)

        predictions = []
        for prob, idx in zip(top_p.cpu().tolist(), top_i.cpu().tolist()):
            predictions.append({
                "class_id": int(idx),
                "label": CLASS_NAMES[int(idx)],
                "confidence": round(float(prob) * 100, 2),
            })
        return predictions


def image_to_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = __import__("base64").b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
