# src/repair_events_scale.py
"""
Rescale event indices in CSV splits so they fit the actual frame count per video.

Why: if your extracted frames are fewer than the original annotation timeline,
many events are > n_frames-1 (over-high). This script linearly maps events to
[0, n_frames-1] per video, preserving relative timing.

Input:
  data/splits/train.csv, val.csv, test.csv  (columns: video,frames_dir,n_frames,events)
Output:
  data/splits_rescaled/train.csv, val.csv, test.csv
"""

import os, json, ast
import pandas as pd
from pathlib import Path

def parse_events(x):
    try:
        return json.loads(x)
    except Exception:
        return ast.literal_eval(x)

def rescale_events_for_row(n_frames, events):
    # events are 0-based with -1 for missing
    pos = [int(e) for e in events if isinstance(e, (int,)) and int(e) >= 0]
    if len(pos) == 0:
        return events  # nothing to do
    max_ev = max(pos)
    # approximate original timeline length by (max_event + 1)
    orig_len = max_ev + 1
    if orig_len <= n_frames:
        # already fits; just clamp
        return [min(max(0, int(e)), n_frames-1) if int(e) >= 0 else -1 for e in events]
    # scale linearly to [0, n_frames-1]
    denom = max(1, orig_len - 1)
    scaled = []
    for e in events:
        if e is None or int(e) < 0:
            scaled.append(-1)
        else:
            r = int(round(int(e) * (n_frames - 1) / denom))
            r = max(0, min(n_frames - 1, r))
            scaled.append(r)
    return scaled

def process_split(in_csv, out_csv):
    df = pd.read_csv(in_csv)
    out = []
    overhigh_before = 0
    overhigh_after = 0
    total_pos = 0

    for _, r in df.iterrows():
        n = int(r["n_frames"])
        ev = parse_events(r["events"])
        # stats before
        for e in ev:
            if isinstance(e, (int,)) and e >= 0:
                total_pos += 1
                if e >= n:
                    overhigh_before += 1

        ev2 = rescale_events_for_row(n, ev)

        # stats after
        for e in ev2:
            if isinstance(e, (int,)) and e >= n:
                overhigh_after += 1

        out.append({
            "video": r["video"],
            "frames_dir": r["frames_dir"],
            "n_frames": n,
            "events": json.dumps(ev2)
        })

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(out_csv, index=False)
    print(f"[{os.path.basename(in_csv)}] total_pos={total_pos}, overhigh_before={overhigh_before}, overhigh_after={overhigh_after}")
    print(f"  -> saved to {out_csv}")

def main():
    pairs = [
        ("data/splits/train.csv", "data/splits_rescaled/train.csv"),
        ("data/splits/val.csv",   "data/splits_rescaled/val.csv"),
        ("data/splits/test.csv",  "data/splits_rescaled/test.csv"),
    ]
    for src, dst in pairs:
        if os.path.exists(src):
            process_split(src, dst)
        else:
            print(f"Missing: {src}")

if __name__ == "__main__":
    main()
