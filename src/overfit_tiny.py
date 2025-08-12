# src/overfit_tiny.py
"""
Quick sanity: can the model overfit a tiny subset?
- Uses rescaled CSV (after repair_events_scale.py)
- Forces event-centered sampling so positives are present
- Uses label dilation (radius=2) to learn a peak rather than a single spike
Expected: loss should drop quickly (e.g., from ~0.6 toward <0.1 within a few hundred steps)
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import GolfDBDataset            # <-- matches your dataset.py
from model import MobileNetV3LSTM           # <-- matches your train.py import

def collate_batch(batch):
    """Stack (frames, targets, vid) into tensors."""
    frames = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    vids = [b[2] for b in batch]
    return torch.stack(frames, 0), torch.stack(targets, 0), vids

def main():
    # ---- config (feel free to tweak) ----
    csv_path = "data/splits_rescaled/train.csv"  # use the RESCALED splits
    frames_root = "data/golfdb_frames"
    seq_len = 32
    img_wh = (160, 160)
    batch_size = 4
    tiny_num_items = 16   # take first 16 items as a tiny subset
    lr = 3e-4
    pos_weight_value = 128.0   # same spirit as training
    max_steps = 200
    print_every = 10
    # -------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Device: {device}")

    # Dataset (force positives inside the window)
    full_ds = GolfDBDataset(
        csv_file=csv_path,
        seq_len=seq_len,
        resize=img_wh,
        frames_root=frames_root,
        pos_sample_prob=1.0,   # force event-centered sampling
        center_jitter=0,
        label_radius=2
    )

    # Tiny subset
    indices = list(range(min(tiny_num_items, len(full_ds))))
    tiny_ds = Subset(full_ds, indices)

    loader = DataLoader(
        tiny_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_batch
    )

    # Model
    model = MobileNetV3LSTM(
        num_events=8,
        lstm_hidden=512,
        lstm_layers=1,
        bidirectional=False,
        pretrained=True
    ).to(device)

    # Loss: BCE with positive class weighting
    pos_weight = torch.full((8,), pos_weight_value, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Train on the tiny subset
    model.train()
    step = 0
    losses = []
    while step < max_steps:
        for frames, targets, vids in loader:
            frames = frames.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(frames)                 # (B, T, E)
            loss = criterion(logits, targets)      # same shape (B,T,E)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))
            if step % print_every == 0:
                print(f"Step {step:4d} | Loss {loss.item():.4f}")
            step += 1
            if step >= max_steps:
                break

    # Optional: save a tiny loss log to file (for later plotting if needed)
    try:
        with open("overfit_tiny_loss.txt", "w", encoding="utf-8") as f:
            for i, v in enumerate(losses):
                f.write(f"{i}\t{v}\n")
        print("[i] Saved loss log to overfit_tiny_loss.txt")
    except Exception as e:
        print(f"[!] Failed to save loss log: {e}")

if __name__ == "__main__":
    main()
