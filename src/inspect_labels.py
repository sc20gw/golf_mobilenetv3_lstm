# src/inspect_labels.py
"""
Inspect GolfDB .pkl / .mat annotation files and probe how to map them to your frames folders.

Usage (PowerShell):
python src\inspect_labels.py --pkl path\to\labels.pkl --frames_root data\golfdb_frames
# 或
python src\inspect_labels.py --mat path\to\labels.mat --frames_root data\golfdb_frames
"""

import os
import re
import json
import argparse
from pathlib import Path

def _norm_vid_to_folder(vid_str: str):
    # Try to turn things like "0001", "0010", "1", "10", "vid_0001", etc. into a folder name you actually have.
    # Your folders are like data/golfdb_frames/1/, /2/, /10/ ... (no leading zeros)
    # Extract last number in the string; if none, return as-is.
    m = re.findall(r"\d+", str(vid_str))
    if not m:
        return str(vid_str)
    n = int(m[-1])  # strip leading zeros
    return str(n)

def _peek_frames(frames_root):
    root = Path(frames_root)
    if not root.exists():
        print(f"[!] frames_root not found: {root}")
        return
    sub = [p for p in root.iterdir() if p.is_dir()]
    print(f"[i] frames_root: {root} | found {len(sub)} subfolders. Examples:", [p.name for p in sub[:10]])

def _check_folder_exists(frames_root, vid_any):
    # try non-padded (preferred for your case)
    p1 = Path(frames_root) / _norm_vid_to_folder(vid_any)
    p2 = None
    # also try zero-padded to 4 digits (0001) just in case
    m = re.findall(r"\d+", str(vid_any))
    if m:
        p2 = Path(frames_root) / str(int(m[-1])).zfill(4)
    ok1 = p1.exists()
    ok2 = p2.exists() if p2 else False
    return p1, ok1, p2, ok2

def _preview_item(idx, item, frames_root):
    print(f"\n=== Sample item #{idx} ===")
    if isinstance(item, dict):
        print("keys:", list(item.keys()))
        # try common keys for video id
        for k in ["video", "video_id", "vid", "name", "id", "recording"]:
            if k in item:
                vid = item[k]
                p1, ok1, p2, ok2 = _check_folder_exists(frames_root, str(vid))
                print(f"video id via '{k}':", vid, "| map ->", p1.name, "| exists:", ok1, "| alt:", (p2.name if p2 else None), "| exists:", ok2)
                break
        # try events
        for k in ["events", "labels", "annotations"]:
            if k in item:
                ev = item[k]
                try:
                    L = len(ev)
                except Exception:
                    L = "N/A"
                print(f"events under key '{k}': length={L}, head=", str(ev)[:120], "...")
                break
    else:
        print("type:", type(item))
        print("repr:", repr(item)[:200], "...")

def load_pkl(pkl_path):
    import pickle
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    return obj

def load_mat(mat_path):
    import scipy.io as sio
    mat = sio.loadmat(mat_path)
    # remove meta keys
    data = {k: v for k, v in mat.items() if not k.startswith("__")}
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=str, default=None)
    ap.add_argument("--mat", type=str, default=None)
    ap.add_argument("--frames_root", type=str, default="data/golfdb_frames")
    args = ap.parse_args()

    _peek_frames(args.frames_root)

    if args.pkl:
        obj = load_pkl(args.pkl)
        print("\n[PKL] type:", type(obj))
        if isinstance(obj, list):
            print("[PKL] list length:", len(obj))
            for i in range(min(2, len(obj))):
                _preview_item(i, obj[i], args.frames_root)
        elif isinstance(obj, dict):
            print("[PKL] dict keys:", list(obj.keys())[:30])
            # try some common containers
            for k in ["data", "videos", "items", "samples", "annotations"]:
                if k in obj and isinstance(obj[k], list):
                    print(f"[PKL] found list under '{k}', length:", len(obj[k]))
                    for i in range(min(2, len(obj[k]))):
                        _preview_item(i, obj[k][i], args.frames_root)
                    break
        else:
            print("[PKL] preview:", repr(obj)[:400])

    if args.mat:
        data = load_mat(args.mat)
        print("\n[MAT] top-level keys:", list(data.keys()))
        # try to guess main array
        for k, v in data.items():
            try:
                import numpy as np
                if hasattr(v, "dtype") and v.size > 0:
                    print(f"[MAT] key '{k}' shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', None)}")
            except Exception:
                pass

        # Often annotations are arrays-of-structs; we try to pretty-print first 1–2 items if possible
        # This part is heuristic; we’ll adapt once you send back the printout.
        print("\n[MAT] (If you see an array of objects/structs above, tell me which key looks like annotations.)")

if __name__ == "__main__":
    main()
