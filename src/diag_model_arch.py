# src/diag_model_arch.py
"""
Print MobileNetV3LSTM architecture details:
- LSTM hidden size / layers / bidirectional
- output channels (should be 8 in BCE version; 9 if using background softmax)
- total params, trainable params
- dummy forward shapes
- check if backbone params are frozen (requires_grad=False)
"""
import torch
from model import MobileNetV3LSTM

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = MobileNetV3LSTM(
        num_events=8,          # 你当前版本应为8；若之后换softmax会改为9
        lstm_hidden=512,
        lstm_layers=1,
        bidirectional=False,
        pretrained=True
    ).to(device)
    print("[i] Device:", device)

    # backbone 冻结情况
    has_backbone = hasattr(m, "backbone")
    if has_backbone:
        bn_frozen = sum((not p.requires_grad) for p in m.backbone.parameters())
        bn_total  = sum(1 for _ in m.backbone.parameters())
        print(f"[i] Backbone params: frozen {bn_frozen}/{bn_total}")
    else:
        print("[i] No attribute 'backbone' found on model.")

    # LSTM 信息
    lstm = None
    for name, module in m.named_modules():
        if "lstm" in name.lower() and isinstance(module, torch.nn.LSTM):
            lstm = module
            break
    if lstm:
        print(f"[i] LSTM: input_size={lstm.input_size}, hidden_size={lstm.hidden_size}, "
              f"layers={lstm.num_layers}, bidirectional={lstm.bidirectional}")
    else:
        print("[i] LSTM module not found (name contains 'lstm').")

    # 分类头输出维度
    head_out = None
    for name, module in m.named_modules():
        if isinstance(module, torch.nn.Linear):
            head_out = module.out_features  # 最后一个Linear会覆盖
    print(f"[i] Last Linear out_features ≈ {head_out} (should be 8 in current BCE setup)")

    # 参量统计
    total, trainable = count_params(m)
    print(f"[i] Params: total={total/1e6:.2f}M  trainable={trainable/1e6:.2f}M")

    # dummy forward
    with torch.no_grad():
        x = torch.randn(1, 32, 3, 160, 160, device=device)  # B,T,C,H,W
        y = m(x)  # (B,T,E)
        print("[i] Dummy forward:", x.shape, "->", y.shape)

if __name__ == "__main__":
    main()
