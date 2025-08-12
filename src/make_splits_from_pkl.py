# src/make_splits_from_pkl.py
"""
Create train/val/test CSVs for our loader from golfDB.pkl + extracted frames.

Input:
  - golfDB.pkl (a pandas DataFrame with columns: id, events, split, ...)
  - frames_root (your extracted frames folders: e.g., data/golfdb_frames/<id>/frame_XXXX.jpg)

Output CSVs (saved under data/splits/):
  - train.csv, val.csv, test.csv
Each CSV has columns:
  video,frames_dir,n_frames,events
Where:
  - video: string id (we use the integer id as given in the pkl)
  - frames_dir: path to the folder containing frames for that id
  - n_frames: number of frame images found in that folder
  - events: JSON list of length 8 (0-based frame indices; -1 if missing)

Notes:
  - We will coerce events to length 8:
      * if len(events) > 8: take the first 8
      * if len(events) < 8: pad with -1
  - We DO NOT shift indices here. If your dataset uses 1-based indices, set --one_based to subtract 1.
  - 'split' values mapping: 1 -> train, 2 -> val, 3 -> test (as seen in your .pkl/.mat)
"""

import os
import json
import argparse
from pathlib import Path
from typing import List

import pandas as pd

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def count_frames(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)


def coerce_events_to_len8(ev_list: List[int], one_based: bool) -> List[int]:
    # Convert to list of ints; coerce to length 8; optionally convert from 1-based to 0-based.
    ev = []
    for x in ev_list:
        try:
            xi = int(x)
        except Exception:
            xi = -1
        ev.append(xi)

    # 1-based -> 0-based if requested
    if one_based:
        ev = [ (e - 1) if (isinstance(e, int) and e > 0) else (-1 if e == 0 else e) for e in ev ]

    # Coerce to length 8
    if len(ev) >= 8:
        ev = ev[:8]
    else:
        ev = ev + [-1] * (8 - len(ev))

    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=str, required=True, help="Path to golfDB.pkl")
    ap.add_argument("--frames_root", type=str, default="data/golfdb_frames", help="Root folder containing frame folders named by id (e.g., 0,1,2,...)")
    ap.add_argument("--splits_dir", type=str, default="data/splits", help="Where to save CSVs")
    ap.add_argument("--one_based", action="store_true", help="If set, subtract 1 from event indices (use if events are 1-based).")
    ap.add_argument("--min_frames", type=int, default=1, help="Skip samples with fewer than this many frames")
    args = ap.parse_args()

    frames_root = Path(args.frames_root)
    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"[i] Loading PKL: {args.pkl}")
    df = pd.read_pickle(args.pkl)
    print(f"[i] DataFrame shape: {df.shape}")
    req_cols = {"id", "events", "split"}
    if not req_cols.issubset(set(df.columns)):
        raise ValueError(f"PKL must contain columns {req_cols}, but got {df.columns.tolist()}")

    rows = {"train": [], "val": [], "test": []}
    miss_frames = 0
    short_frames = 0

    for _, r in df.iterrows():
        vid = r["id"]                      # integer id (0..1399)
        split = int(r["split"])            # 1=train, 2=val, 3=test
        events = r["events"]

        # normalize events to python list
        if isinstance(events, (list, tuple)):
            ev_list = list(events)
        else:
            # if it's a numpy array or something else:
            try:
                ev_list = list(events)
            except Exception:
                ev_list = []

        ev8 = coerce_events_to_len8(ev_list, one_based=args.one_based)

        frames_dir = frames_root / str(int(vid))  # your folders are non-padded integers: "0","1","10",...
        n_frames = count_frames(frames_dir)
        if n_frames < args.min_frames:
            short_frames += 1
            continue

        if n_frames == 0:
            miss_frames += 1
            continue

        row = {
            "video": str(int(vid)),
            "frames_dir": str(frames_dir).replace("\\", "/"),
            "n_frames": int(n_frames),
            "events": json.dumps(ev8),
        }

        if split == 1:
            rows["train"].append(row)
        elif split == 2:
            rows["val"].append(row)
        elif split == 3:
            rows["test"].append(row)
        else:
            # unexpected split value: skip or assign to train
            rows["train"].append(row)

    # Save CSVs
    for name in ["train", "val", "test"]:
        out_csv = splits_dir / f"{name}.csv"
        pd.DataFrame(rows[name]).to_csv(out_csv, index=False)
        print(f"[+] Saved {name}.csv with {len(rows[name])} rows -> {out_csv}")

    print(f"[!] Missing-frame folders: {miss_frames}  |  Too few frames (<{args.min_frames}): {short_frames}")
    print("[i] Done.")

if __name__ == "__main__":
    main()
