# src/diag_labels_distribution.py
"""
Check label distribution AFTER dataloader transforms (pos sampling + dilation)
and sanity-check GT event ordering in CSV.

It prints:
- avg / min / max positives per sample (after dilation)
- per-event positive rate (fraction of frames labeled 1) aggregated over sampled batches
- count of GT ordering violations (non-monotonic events)
- % of samples that still end up with zero positives (should be near 0 when pos_sample_prob=1)

Usage (PowerShell):
python src\diag_labels_distribution.py --csv data\splits_rescaled_bbox\train.csv --frames_root data\golfdb_frames --seq_len 32 --batch 8 --batches 50
"""

import argparse, json, ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import GolfDBDataset

def parse_list(s):
    try: return json.loads(s)
    except Exception: return ast.literal_eval(s)

def check_gt_order(csv_path):
    df = pd.read_csv(csv_path)
    bad = 0
    total = 0
    for _, r in df.iterrows():
        ev = parse_list(r["events"])[:8]
        # keep only non-negative
        ev = [int(x) for x in ev if x is not None and int(x) >= 0]
        if len(ev) <= 1:
            continue
        total += 1
        ok = all(ev[i] < ev[i+1] for i in range(len(ev)-1))
        if not ok:
            bad += 1
    return bad, total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--frames_root", default="data/golfdb_frames")
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--batches", type=int, default=50, help="how many batches to sample")
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=160)
    ap.add_argument("--label_radius", type=int, default=2)
    args = ap.parse_args()

    ds = GolfDBDataset(
        csv_file=args.csv,
        seq_len=args.seq_len,
        resize=(args.width, args.height),
        frames_root=args.frames_root,
        pos_sample_prob=1.0,    # force event-centered to ensure positives
        center_jitter=0,
        label_radius=args.label_radius,
        use_bbox=True,
        bbox_margin=0.10
    )
    def collate(b):
        import torch
        return torch.stack([x[0] for x in b],0), torch.stack([x[1] for x in b],0), [x[2] for x in b]
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=collate)

    pos_counts = []
    per_event_pos = np.zeros(8, dtype=np.float64)
    total_frames = 0
    zero_pos_samples = 0

    iters = 0
    for frames, targets, vids in dl:
        # targets: (B, T, 8)
        b, t, e = targets.shape
        pc = targets.sum(dim=(1,2)).cpu().numpy().tolist()
        pos_counts.extend(pc)
        zero_pos_samples += int((targets.sum(dim=(1,2)) == 0).sum().item())

        # per-event positive frames
        per_event_pos += targets.sum(dim=(0,1)).cpu().numpy()  # sum over B and T
        total_frames += b * t
        iters += 1
        if iters >= args.batches:
            break

    pos_counts = np.array(pos_counts, dtype=np.float64)
    print(f"[AFTER DATALOADER] positives per sample (with dilation): "
          f"avg={pos_counts.mean():.2f}, min={pos_counts.min():.0f}, max={pos_counts.max():.0f}")
    print(f"[AFTER DATALOADER] % of samples with ZERO positives: {100.0*zero_pos_samples/len(pos_counts):.2f}% (should be ~0%)")

    # per-event positive rate relative to all frames across sampled batches
    rates = per_event_pos / total_frames
    for i, r in enumerate(rates):
        print(f"event {i} positive-frame ratio: {100.0*r:.3f}%")
    print(f"(Sum over events) avg positive-frame ratio per frame: {100.0*rates.sum():.3f}% "
          f"(note: multiple events rarely overlap; with dilation radius={args.label_radius})")

    # GT ordering sanity
    bad, total = check_gt_order(args.csv)
    if total > 0:
        print(f"[GT ORDER] non-monotonic samples: {bad}/{total} = {100.0*bad/total:.2f}%")
    else:
        print("[GT ORDER] not enough samples to evaluate.")

if __name__ == "__main__":
    main()
