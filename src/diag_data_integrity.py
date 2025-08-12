# src/diag_data_integrity.py
"""
Data integrity check for GolfDB CSV splits.

It verifies:
- frames_dir exists and contains image files
- actual n_images vs CSV n_frames
- event indices within [0, n_images-1] or -1
- bbox validity (normalized, w/h>0)
- split leakage (same video id appearing across splits)
- too-short videos (< seq_len)

Writes per-row reports to data/diagnostics/*.csv and prints a summary.
Optionally rewrites CSVs with corrected n_frames.

Usage (PowerShell):
python src\diag_data_integrity.py --splits_dir data\splits_rescaled_bbox --frames_root data\golfdb_frames --seq_len 64
# (optional) also create corrected CSVs with fixed n_frames:
python src\diag_data_integrity.py --splits_dir data\splits_rescaled_bbox --frames_root data\golfdb_frames --seq_len 64 --rewrite_n_frames
"""

import os, re, json, ast, argparse
from pathlib import Path
import pandas as pd

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def parse_list(x):
    try: return json.loads(x)
    except Exception:
        try: return ast.literal_eval(x)
        except Exception: return None

def list_images_sorted(frames_dir):
    if not os.path.isdir(frames_dir):
        return []
    files = [f for f in os.listdir(frames_dir) if f.lower().endswith(IMG_EXTS)]
    def key(fn):
        nums = re.findall(r"\d+", fn)
        return (0, int(nums[-1])) if nums else (1, fn.lower())
    return sorted(files, key=key)

def resolve_frames_dir(frames_dir, frames_root):
    if os.path.isabs(frames_dir) and os.path.isdir(frames_dir):
        return frames_dir
    if os.path.isdir(frames_dir):
        return frames_dir
    cand = os.path.join(frames_root, frames_dir)
    if os.path.isdir(cand):
        return cand
    base = os.path.join(frames_root, os.path.basename(frames_dir))
    if os.path.isdir(base):
        return base
    return frames_dir  # may be invalid

def check_split(csv_path, frames_root, seq_len):
    df = pd.read_csv(csv_path)
    rows = []
    over_high = 0
    below_neg1 = 0
    total_nonneg = 0
    missing_frames_dir = 0
    n_mismatch = 0
    invalid_bbox = 0
    too_short = 0

    for i, r in df.iterrows():
        video = r.get("video")
        frames_dir_raw = str(r.get("frames_dir"))
        frames_dir = resolve_frames_dir(frames_dir_raw, frames_root)
        imgs = list_images_sorted(frames_dir)
        n_img = len(imgs)
        n_csv = int(r.get("n_frames")) if "n_frames" in r and pd.notna(r.get("n_frames")) else -1

        # events
        ev = parse_list(r.get("events"))
        if not isinstance(ev, (list,tuple)):
            ev = []
        ev = ev[:8] + [-1] * max(0, 8 - len(ev))

        # bbox
        bb_raw = r.get("bbox") if "bbox" in df.columns else None
        bbox_ok = True
        bbox_val = None
        if bb_raw is not None and pd.notna(bb_raw):
            bb = parse_list(bb_raw)
            if isinstance(bb, (list,tuple)) and len(bb) == 4:
                x,y,w,h = bb
                bbox_val = [x,y,w,h]
                # normalized 0..1 check (allow a bit slack)
                if not (0 <= float(w) and 0 <= float(h) and float(w) <= 1.2 and float(h) <= 1.2 and float(w) > 0 and float(h) > 0):
                    bbox_ok = False
                if not (-0.2 <= float(x) <= 1.2 and -0.2 <= float(y) <= 1.2):
                    bbox_ok = False
            else:
                bbox_ok = False

        # event bounds
        ev_over = 0; ev_below = 0; ev_nonneg = 0
        for e in ev:
            if e is None: continue
            e = int(e)
            if e >= 0: ev_nonneg += 1
            if n_img > 0 and e >= n_img: ev_over += 1
            if e < -1: ev_below += 1
        over_high += ev_over
        below_neg1 += ev_below
        total_nonneg += ev_nonneg

        # frames_dir exists?
        exists = os.path.isdir(frames_dir)
        if not exists: missing_frames_dir += 1

        # n_frames mismatch?
        mismatch = (n_img != n_csv and n_csv >= 0)
        if mismatch: n_mismatch += 1

        # too short?
        too_short_flag = (n_img < seq_len)
        if too_short_flag: too_short += 1

        if not bbox_ok: invalid_bbox += 1

        rows.append({
            "video": video,
            "frames_dir_csv": frames_dir_raw,
            "frames_dir_resolved": frames_dir,
            "exists": int(exists),
            "n_images_actual": n_img,
            "n_frames_csv": n_csv,
            "n_mismatch": int(mismatch),
            "events_nonneg": ev_nonneg,
            "events_overhigh": ev_over,
            "events_below_minus1": ev_below,
            "bbox_ok": int(bbox_ok),
            "too_short(<seq_len)": int(too_short_flag)
        })

    rep = pd.DataFrame(rows)
    summary = {
        "rows": len(df),
        "missing_frames_dir": missing_frames_dir,
        "n_mismatch_rows": n_mismatch,
        "events_total_nonneg": total_nonneg,
        "events_over_high": over_high,
        "events_below_-1": below_neg1,
        "invalid_bbox_rows": invalid_bbox,
        "too_short_rows(<seq_len)": too_short
    }
    return rep, summary

def check_leakage(train_csv, val_csv, test_csv):
    vids = {}
    for name, p in [("train", train_csv), ("val", val_csv), ("test", test_csv)]:
        if not os.path.exists(p): continue
        df = pd.read_csv(p)
        vids[name] = set(map(int, df["video"].tolist()))
    leaks = {}
    if "train" in vids and "val" in vids:
        leaks["train_val_overlap"] = sorted(list(vids["train"] & vids["val"]))
    if "train" in vids and "test" in vids:
        leaks["train_test_overlap"] = sorted(list(vids["train"] & vids["test"]))
    if "val" in vids and "test" in vids:
        leaks["val_test_overlap"] = sorted(list(vids["val"] & vids["test"]))
    return leaks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True, help="dir containing train.csv/val.csv/test.csv")
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--out_dir", default="data/diagnostics")
    ap.add_argument("--rewrite_n_frames", action="store_true", help="write corrected CSVs with actual n_images")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    split_paths = {fn: os.path.join(args.splits_dir, fn) for fn in ["train.csv","val.csv","test.csv"]}
    all_summ = {}
    for name, p in split_paths.items():
        if not os.path.exists(p):
            print(f"[WARN] Missing: {p}")
            continue
        rep, summ = check_split(p, args.frames_root, args.seq_len)
        out_csv = os.path.join(args.out_dir, f"report_{name}")
        rep.to_csv(out_csv, index=False)
        print(f"[{name}] rows={summ['rows']}  missing_frames_dir={summ['missing_frames_dir']}  n_mismatch_rows={summ['n_mismatch_rows']}  "
              f"events_over_high={summ['events_over_high']}  invalid_bbox_rows={summ['invalid_bbox_rows']}  too_short(<{args.seq_len})={summ['too_short_rows(<seq_len)']}")
        all_summ[name] = summ

        # optionally write corrected CSVs
        if args.rewrite_n_frames:
            df = pd.read_csv(p)
            # load rep again to align ordering
            rep_df = pd.read_csv(out_csv)
            fixed = df.copy()
            fixed["n_frames"] = rep_df["n_images_actual"].values
            out_fixed_dir = os.path.join(args.out_dir, "splits_verified")
            Path(out_fixed_dir).mkdir(parents=True, exist_ok=True)
            fixed_out = os.path.join(out_fixed_dir, name)
            fixed.to_csv(fixed_out, index=False)
            print(f"[{name}] wrote corrected CSV with actual n_frames -> {fixed_out}")

    # leakage
    leaks = check_leakage(split_paths["train.csv"], split_paths["val.csv"], split_paths["test.csv"])
    for k, v in leaks.items():
        print(f"[leakage] {k}: {len(v)} overlaps")

if __name__ == "__main__":
    main()
