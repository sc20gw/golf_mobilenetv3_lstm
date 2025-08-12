# src/model.py
import torch, torch.nn as nn
import torchvision.models as tv

class MobileNetV3LSTM(nn.Module):
    def __init__(self, num_events=9, lstm_hidden=512, lstm_layers=1, bidirectional=False, pretrained=True):
        super().__init__()
        # MobileNetV3-Large backbone
        m = tv.mobilenet_v3_large(weights=tv.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        self.backbone = nn.Sequential(*(list(m.features.children())))  # keep features only
        self.gap = nn.AdaptiveAvgPool2d(1)
        feat_dim = 960  # mobilenetv3-large last feature dim

        self.lstm = nn.LSTM(
            input_size=feat_dim, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=bidirectional, dropout=0.0
        )
        head_in = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(head_in, num_events)  # <-- 9 classes (background + 8 events)

    def forward(self, x):          # x: (B,T,C,H,W)
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.backbone(x)   # (B*T, C', h, w)
        feats = self.gap(feats).flatten(1)  # (B*T, C')
        feats = feats.view(B, T, -1)        # (B,T,C')
        y, _ = self.lstm(feats)             # (B,T,H)
        logits = self.head(y)               # (B,T,9)
        return logits
