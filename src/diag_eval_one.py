# src/diag_eval_one.py
# Inspect one test sample: prints GT vs Pred and per-event hit@k.
# - Auto-restores LSTM hyper-params from checkpoint["args"]
# - Uses 9-class softmax (0=bg, 1..8=events) to match training
# - Sliding-window inference with stride, then per-frame fusion + light smoothing

import argparse, os, json, ast, re
import numpy as np
import pandas as pd
import torch, cv2

from model import MobileNetV3LSTM
from utils import load_checkpoint

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def list_images_sorted(d):
    files = [f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)]
    files = sorted(files, key=lambda fn: (0, int(re.findall(r"\d+", fn)[-1])) if re.findall(r"\d+", fn) else (1, fn.lower()))
    return [os.path.join(d, f) for f in files]

def parse_list(x):
    if isinstance(x, (list, tuple)): return list(x)
    try: return json.loads(x)
    except Exception: return ast.literal_eval(x)

def clamp(v, lo, hi): return max(lo, min(hi, v))

def crop(im, bbox, margin=0.10):
    if bbox is None: return im
    x,y,w,h = bbox
    if w<=0 or h<=0 or x<0 or y<0: return im
    H,W = im.shape[:2]
    x1 = clamp(int((x-margin)*W), 0, W-1)
    y1 = clamp(int((y-margin)*H), 0, H-1)
    x2 = clamp(int((x+w+margin)*W), 1, W)
    y2 = clamp(int((y+h+margin)*H), 1, H)
    if x2<=x1 or y2<=y1: return im
    return im[y1:y2, x1:x2, :]

def rrn(path, wh, bbox=None, margin=0.10):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(path)
    im = crop(im, bbox, margin)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if wh is not None:
        im = cv2.resize(im, wh, interpolation=cv2.INTER_LINEAR)
    im = im.astype(np.float32) / 255.0
    im = (im - IMAGENET_MEAN) / IMAGENET_STD
    return im.transpose(2,0,1)  # (3,H,W)

@torch.no_grad()
def fuse_logits_over_windows(model, device, frames_dir, seq_len, stride, wh, bbox, margin):
    files = list_images_sorted(frames_dir); n = len(files)
    if n == 0: raise RuntimeError(f"No frames in {frames_dir}")
    acc = np.zeros((n, 9), dtype=np.float32)  # 9 classes (0=bg)
    cnt = np.zeros((n, 1), dtype=np.float32)
    starts = list(range(0, max(1, n-seq_len+1), stride)) or [0]
    for s in starts:
        clip = files[s:min(n, s+seq_len)]
        if len(clip) < seq_len:
            clip += [files[-1]] * (seq_len - len(clip))
        x = np.stack([rrn(p, wh, bbox, margin) for p in clip], axis=0)  # (T,3,H,W)
        x = torch.from_numpy(x).unsqueeze(0).to(device)                  # (1,T,3,H,W)
        logits = model(x).squeeze(0).cpu().numpy()                       # (T,9)
        T = logits.shape[0]
        for i in range(T):
            g = s + i
            if g >= n: break
            acc[g] += logits[i]
            cnt[g] += 1.0
    acc = acc / np.maximum(cnt, 1.0)
    # optional smoothing (no hard dependency)
    try:
        from scipy.ndimage import uniform_filter1d
        acc = uniform_filter1d(acc, size=5, axis=0, mode="nearest")
    except Exception:
        pass
    return acc  # (N,9)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--frames_root", default="data/golfdb_frames")
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=160)
    ap.add_argument("--bbox_margin", type=float, default=0.10)
    ap.add_argument("--k", type=int, default=5)
    # optional manual overrides
    ap.add_argument("--lstm_hidden", type=int, default=None)
    ap.add_argument("--lstm_layers", type=int, default=None)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--num_events", type=int, default=9)  # must be 9 to match training (0=bg + 8 events)
    ap.add_argument("--one_based", action="store_true")   # set True if CSV events are 1-based (your splits_monotonic are)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[i] device:", device, flush=True)

    # load ckpt & infer hyper-params
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    ck_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    lstm_hidden = args.lstm_hidden if args.lstm_hidden is not None else int(ck_args.get("lstm_hidden", 512))
    lstm_layers = args.lstm_layers if args.lstm_layers is not None else int(ck_args.get("lstm_layers", 1))
    bidir = bool(args.bidirectional or ck_args.get("bidirectional", False))

    # build model with the SAME head size as training
    model = MobileNetV3LSTM(
        num_events=args.num_events,   # 9 classes
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        bidirectional=bidir,
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # read row
    df = pd.read_csv(args.test_csv)
    if not (0 <= args.row < len(df)):
        raise IndexError(f"--row {args.row} out of range (0..{len(df)-1})")
    r = df.iloc[args.row]
    frames_dir = r["frames_dir"]
    if not os.path.isabs(frames_dir) and not os.path.exists(frames_dir):
        frames_dir = os.path.join(args.frames_root, os.path.basename(frames_dir))

    # parse GT events (convert 1-based -> 0-based if requested)
    gt = parse_list(r["events"])
    gt = list(gt)[:8] + [-1]*max(0, 8-len(gt))
    gt2 = []
    for e in gt:
        try:
            e = int(e)
        except Exception:
            e = -1
        if e >= 0 and args.one_based:
            e = e - 1
        gt2.append(e)
    gt = gt2

    # parse bbox (optional)
    bbox = None
    if "bbox" in df.columns and isinstance(r["bbox"], str) and len(r["bbox"])>0:
        bb = parse_list(r["bbox"])
        if isinstance(bb, (list, tuple)) and len(bb)==4:
            bbox = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]

    # inference
    logits = fuse_logits_over_windows(
        model, device, frames_dir,
        seq_len=args.seq_len, stride=args.stride,
        wh=(args.width, args.height),
        bbox=bbox, margin=args.bbox_margin
    )  # (N,9)

    # pick per-event indices (argmax over classes 1..8)
    scores = logits[:, 1:9]  # drop bg
    pred = [int(np.argmax(scores[:, e])) for e in range(8)]

    # print GT & Pred & per-event errors
    print("GT  :", [int(x) if x is not None else -1 for x in gt])
    print("Pred:", pred)
    hits = 0; tot = 0
    for i, (p, g) in enumerate(zip(pred, gt)):
        if g < 0: continue
        tot += 1
        hit = abs(p - g) <= args.k
        hits += int(hit)
        print(f"event {i}: pred={p:>3}, gt={g:>3}, err={abs(p-g):>2}, hit@{args.k}={str(hit)}")
    pce = 100.0 * hits / max(1, tot)
    print(f"[one sample] PCE@{args.k} = {pce:.2f}%  ({hits}/{tot})")

if __name__ == "__main__":
    main()
