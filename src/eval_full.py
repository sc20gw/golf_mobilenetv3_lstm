# src/eval_full.py
"""
Full-video sliding-window evaluation for the softmax (9-class) version:
- Model outputs per-frame logits over 9 classes: 0=background, 1..8=events
- We slide a window, accumulate per-frame logits, optionally smooth,
  then for each event id in 1..8 we take the frame index with the max logit.

Now robust to checkpoint hyperparams:
- If the checkpoint contains 'args' (from train.py), we read lstm_hidden / lstm_layers / bidirectional
  so you don't need to pass them on CLI.
"""

import argparse, os, json, ast
import numpy as np
import torch
import cv2
import pandas as pd
from tqdm import tqdm

from model import MobileNetV3LSTM
from utils import load_checkpoint, compute_pce_for_video

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def list_images_sorted(frames_dir):
    files = [f for f in os.listdir(frames_dir) if f.lower().endswith(IMG_EXTS)]
    import re
    files = sorted(files, key=lambda fn: (0, int(re.findall(r"\d+", fn)[-1])) if re.findall(r"\d+", fn) else (1, fn.lower()))
    return [os.path.join(frames_dir, f) for f in files]

def _clamp(v, lo, hi): return max(lo, min(hi, v))

def crop_by_bbox(img, bbox, margin=0.1):
    if bbox is None: return img
    x,y,w,h = bbox
    if w<=0 or h<=0 or x<0 or y<0: return img
    H,W = img.shape[:2]
    x1 = _clamp(int((x - margin) * W), 0, W-1)
    y1 = _clamp(int((y - margin) * H), 0, H-1)
    x2 = _clamp(int((x + w + margin) * W), 1, W)
    y2 = _clamp(int((y + h + margin) * H), 1, H)
    if x2<=x1 or y2<=y1: return img
    return img[y1:y2, x1:x2, :]

def read_resize_norm(path, wh, bbox=None, bbox_margin=0.1):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if img is None: raise FileNotFoundError(path)
    if bbox is not None:
        img = crop_by_bbox(img, bbox, margin=bbox_margin)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if wh is not None:
        img = cv2.resize(img, wh, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2,0,1)   # C,H,W
    return img

def parse_json_or_literal(x):
    try: return json.loads(x)
    except Exception: return ast.literal_eval(x)

@torch.no_grad()
def infer_full_video(model, device, frames_dir, seq_len=64, stride=16, wh=(160,160), bbox=None, bbox_margin=0.1):
    files = list_images_sorted(frames_dir)
    n = len(files)
    if n == 0:
        raise RuntimeError(f"No images in {frames_dir}")

    C = 9  # softmax classes: 0=background, 1..8=events
    acc = np.zeros((n, C), dtype=np.float32)   # accumulate logits per frame
    cnt = np.zeros((n, 1), dtype=np.float32)   # how many windows cover each frame

    starts = list(range(0, max(1, n - seq_len + 1), stride)) or [0]
    for s in starts:
        clip_files = files[s : min(n, s + seq_len)]
        if len(clip_files) < seq_len:
            clip_files += [files[-1]] * (seq_len - len(clip_files))
        clip = np.stack([read_resize_norm(p, wh, bbox=bbox, bbox_margin=bbox_margin) for p in clip_files], axis=0)  # (T,C,H,W)
        x = torch.from_numpy(clip).unsqueeze(0).to(device)  # (1,T,C,H,W)
        logits = model(x).squeeze(0).detach().cpu().numpy() # (T,9) raw logits

        T = logits.shape[0]
        for i in range(T):
            g = s + i
            if g >= n: break
            acc[g] += logits[i]
            cnt[g] += 1.0

    acc = acc / np.maximum(cnt, 1.0)

    # Optional temporal smoothing
    try:
        from scipy.ndimage import uniform_filter1d
        acc = uniform_filter1d(acc, size=5, axis=0, mode="nearest")
    except Exception:
        pass

    # For events 1..8, take frame index of argmax on that class column
    pred_indices = [int(np.argmax(acc[:, e])) for e in range(1, 9)]
    return pred_indices, acc  # len=8

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--frames_root", default="data/golfdb_frames")
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=160)
    ap.add_argument("--bbox_margin", type=float, default=0.10)
    # (CLI fallbacks; usually ignored because we read from ckpt['args'])
    ap.add_argument("--lstm_hidden", type=int, default=512)
    ap.add_argument("--lstm_layers", type=int, default=1)
    ap.add_argument("--bidirectional", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt = load_checkpoint(args.ckpt, map_location=device)
    ck_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    lstm_hidden = int(ck_args.get("lstm_hidden", args.lstm_hidden))
    lstm_layers = int(ck_args.get("lstm_layers", args.lstm_layers))
    bidirectional = bool(ck_args.get("bidirectional", args.bidirectional))
    print(f"[i] Using model hyperparams -> hidden={lstm_hidden}, layers={lstm_layers}, bidirectional={bidirectional}")

    model = MobileNetV3LSTM(
        num_events=9,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        bidirectional=bidirectional,
        pretrained=False
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    df = pd.read_csv(args.test_csv)

    total_c = 0
    total_t = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        frames_dir = row["frames_dir"]
        if not os.path.isabs(frames_dir) and not os.path.exists(frames_dir):
            frames_dir = os.path.join(args.frames_root, os.path.basename(frames_dir))

        gt_events = parse_json_or_literal(row["events"])
        gt_events = [int(e) if e is not None else -1 for e in gt_events]
        gt_events = gt_events[:8] + [-1] * max(0, 8 - len(gt_events))

        bbox = None
        if "bbox" in df.columns and not pd.isna(row["bbox"]):
            bb = parse_json_or_literal(row["bbox"])
            if isinstance(bb, (list,tuple)) and len(bb)==4:
                bbox = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]

        pred_indices, _ = infer_full_video(
            model, device, frames_dir,
            seq_len=args.seq_len, stride=args.stride, wh=(args.width, args.height),
            bbox=bbox, bbox_margin=args.bbox_margin
        )

        c, t = compute_pce_for_video(pred_indices, gt_events, k=args.k)
        total_c += c
        total_t += t

    pce = total_c / total_t if total_t > 0 else 0.0
    print(f"PCE @ k={args.k}: {pce*100:.2f}%  ({int(total_c)}/{int(total_t)})")

if __name__ == "__main__":
    main()
