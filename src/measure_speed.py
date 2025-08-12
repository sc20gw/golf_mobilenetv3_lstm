# src/measure_speed.py
"""
Measure approximate seconds per training batch on your machine.

Usage (Windows PowerShell / PyCharm Terminal):
python src\measure_speed.py --seq_len 32 --batch_size 8 --iters 20 --height 224 --width 224
"""

import time
import argparse
import torch
from model import MobileNetV3LSTM

def measure(seq_len=32, batch_size=8, iters=20, warmup=5, width=224, height=224):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    model = MobileNetV3LSTM(pretrained=False).to(device)
    model.train()
    C = 3
    x = torch.randn(batch_size, seq_len, C, height, width, device=device)
    y = torch.randn(batch_size, seq_len, 8, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # warm-up
    for _ in range(warmup):
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

    times = []
    for _ in range(iters):
        t0 = time.time()
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    avg = sum(times) / len(times)
    print(f"Average seconds per training batch (batch_size={batch_size}, seq_len={seq_len}): {avg:.4f}s")
    return avg

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--height", type=int, default=224)
    args = ap.parse_args()
    measure(args.seq_len, args.batch_size, args.iters, args.warmup, args.width, args.height)
