# src/fix_events_monotonic.py
"""
Enforce strictly monotonic event indices per video:
- Input splits: data/splits_rescaled_bbox/*.csv  (events already rescaled, with bbox)
- Output splits: data/splits_monotonic/*.csv     (same columns, events fixed)

Strategy:
1) Parse events (length 8, -1 means missing).
2) Keep non-negative events and sort by their original order (0..7 is the semantic order).
3) Enforce strict increasing with a minimal gap (gap=1) while staying in [0, n_frames-1].
   We try to preserve the original relative positions by:
   - First clamping to [0, n-1]
   - Then applying cumulative max with +1 step to break ties
   - If we hit the end, back-propagate from the tail to maintain order.
4) Leave -1 as -1 (missing) but still ensure remaining events preserve order.

Usage (PowerShell):
python src\fix_events_monotonic.py --in_dir data\splits_rescaled_bbox --out_dir data\splits_monotonic
"""

import os, json, ast, argparse
import pandas as pd
from pathlib import Path

def parse_list(x):
    try: return json.loads(x)
    except Exception: return ast.literal_eval(x)

def enforce_monotonic(ev, n_frames):
    """
    ev: list of 8 ints (>=0 or -1), 0-based indices
    returns a new list with strictly increasing non-negative events.
    """
    ev = list(ev[:8]) + [-1] * max(0, 8 - len(ev))
    # clamp first (just in case)
    ev = [min(max(int(e), -1), n_frames-1) if e is not None else -1 for e in ev]

    # collect valid pairs (idx, val)
    idxs = [i for i, e in enumerate(ev) if e >= 0]
    if len(idxs) <= 1:
        return ev  # nothing to fix

    vals = [ev[i] for i in idxs]

    # forward pass: enforce non-decreasing first
    for k in range(1, len(vals)):
        if vals[k] < vals[k-1]:
            vals[k] = vals[k-1]

    # then make it strictly increasing by adding minimal gaps
    for k in range(1, len(vals)):
        if vals[k] <= vals[k-1]:
            vals[k] = vals[k-1] + 1

    # clamp tail if overflow, then backward adjust to keep strict order
    overflow = max(0, vals[-1] - (n_frames - 1))
    if overflow > 0:
        vals[-1] = n_frames - 1
        # back-propagate
        for k in range(len(vals)-2, -1, -1):
            vals[k] = min(vals[k], vals[k+1] - 1)
        # if head becomes negative after back-prop, shift whole chain up
        shift = 0
        if vals[0] < 0:
            shift = -vals[0]
            vals = [v + shift for v in vals]

    # write back
    out = ev[:]
    for j, v in zip(idxs, vals):
        out[j] = int(v)
    # sanity: ensure strict increasing among non-negatives
    last = -1
    for j in idxs:
        if out[j] <= last:
            out[j] = last + 1
        last = out[j]
    # clamp again
    out = [min(max(e, -1), n_frames-1) if e >= 0 else -1 for e in out]
    return out

def fix_one_csv(in_csv, out_csv):
    df = pd.read_csv(in_csv)
    fixed_rows = []
    bad_before = 0
    bad_after  = 0

    def non_monotonic_count(ev):
        xs = [int(x) for x in ev if x is not None and int(x) >= 0]
        if len(xs) <= 1: return 0
        return 0 if all(xs[i] < xs[i+1] for i in range(len(xs)-1)) else 1

    for _, r in df.iterrows():
        n = int(r["n_frames"]) if "n_frames" in r and not pd.isna(r["n_frames"]) else 0
        ev = parse_list(r["events"])
        ev = list(ev[:8]) + [-1]*max(0, 8-len(ev))

        bad_before += non_monotonic_count(ev)
        ev_fixed = enforce_monotonic(ev, max(1, n))
        bad_after  += non_monotonic_count(ev_fixed)

        row = r.to_dict()
        row["events"] = json.dumps(ev_fixed)
        fixed_rows.append(row)

    out_df = pd.DataFrame(fixed_rows)
    Path(os.path.dirname(out_csv)).mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[{os.path.basename(in_csv)}] non-monotonic before={bad_before}, after={bad_after} -> {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for fn in ["train.csv", "val.csv", "test.csv"]:
        src = os.path.join(args.in_dir, fn)
        dst = os.path.join(args.out_dir, fn)
        if os.path.exists(src):
            fix_one_csv(src, dst)
        else:
            print(f"Missing: {src}")

if __name__ == "__main__":
    main()
