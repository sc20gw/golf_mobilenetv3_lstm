# src/eval.py
"""
Evaluate a checkpoint on test split using PCE@k.

Example:
python src\eval.py --ckpt models\checkpoint_latest.pth --test_csv data\splits\test.csv --frames_root data\golfdb_frames --seq_len 64 --batch_size 8 --k 5 --width 160 --height 160
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import GolfDBDataset
from model import MobileNetV3LSTM
from utils import load_checkpoint, pred_frame_indices_from_logits, compute_pce_for_video


def collate_batch(batch):
    import torch
    frames = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    vids = [b[2] for b in batch]
    return torch.stack(frames, 0), torch.stack(targets, 0), vids


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = MobileNetV3LSTM(
        num_events=8,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
        pretrained=False
    ).to(device)

    ckpt = load_checkpoint(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    ds = GolfDBDataset(args.test_csv, seq_len=args.seq_len, resize=(args.width, args.height), frames_root=args.frames_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_batch)

    total_correct = 0
    total_events = 0

    for frames, targets, vids in loader:
        frames = frames.to(device, non_blocking=True)
        logits = model(frames)            # (B, T, E)
        probs = logits.detach().cpu().numpy()

        for i in range(probs.shape[0]):
            p = probs[i]                  # (T, E)
            pred_indices = pred_frame_indices_from_logits(p)
            gt = targets[i].numpy()       # (T, E)
            # reconstruct gt event indices from one-hot per time step
            gt_indices = []
            for e in range(gt.shape[1]):
                pos = np.where(gt[:, e] == 1)[0]
                gt_indices.append(int(pos[0]) if pos.size > 0 else -1)

            c, tot = compute_pce_for_video(pred_indices, gt_indices, k=args.k)
            total_correct += c
            total_events += tot

    pce = total_correct / total_events if total_events > 0 else 0.0
    print(f"PCE @ k={args.k}: {pce*100:.2f}%  ({int(total_correct)}/{int(total_events)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--frames_root", default="data/golfdb_frames")
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lstm_hidden", type=int, default=512)
    ap.add_argument("--lstm_layers", type=int, default=1)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=160)
    args = ap.parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()
