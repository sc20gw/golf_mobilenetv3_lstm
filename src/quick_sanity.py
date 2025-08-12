# src/quick_sanity.py
import torch
from torch.utils.data import DataLoader
from dataset import GolfDBDataset
import numpy as np

def main():
    ds = GolfDBDataset(
        csv_file="data/splits_rescaled/train.csv",
        seq_len=32,
        resize=(160,160),
        frames_root="data/golfdb_frames",
        pos_sample_prob=1.0,
        center_jitter=0,
        label_radius=2
    )
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0,
                    collate_fn=lambda b: (torch.stack([x[0] for x in b],0),
                                          torch.stack([x[1] for x in b],0),
                                          [x[2] for x in b]))
    frames, targets, vids = next(iter(dl))
    print("frames:", frames.shape)     # (B, T, C, H, W)
    print("targets:", targets.shape)   # (B, T, 8)
    pos_per_sample = targets.sum(dim=(1,2)).tolist()
    print("positives per sample (should be >0, often ~3-9 due to dilation):", pos_per_sample)

    t0 = targets[0].numpy()
    for e in range(8):
        pos = np.where(t0[:,e] == 1)[0]
        if pos.size > 0:
            print(f"event {e}: local idx range {int(pos[0])}..{int(pos[-1])}")
        else:
            print(f"event {e}: NOT in this 32-frame window (unexpected for this sanity run)")

if __name__ == "__main__":
    main()
