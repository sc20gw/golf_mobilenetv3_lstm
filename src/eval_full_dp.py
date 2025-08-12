# src/eval_full_dp.py
import argparse, os, json, ast
import numpy as np
import torch, cv2, pandas as pd
from tqdm import tqdm
from model import MobileNetV3LSTM
from utils import load_checkpoint, compute_pce_for_video

IMAGENET_MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229,0.224,0.225], dtype=np.float32)
IMG_EXTS = (".jpg",".jpeg",".png",".bmp")

def list_images_sorted(d):
    fs = [f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)]
    import re
    fs = sorted(fs, key=lambda fn: (0,int(re.findall(r"\d+",fn)[-1])) if re.findall(r"\d+",fn) else (1,fn.lower()))
    return [os.path.join(d,f) for f in fs]

def _clamp(v,lo,hi): return max(lo,min(hi,v))
def crop(img,bbox,margin=0.1):
    if bbox is None: return img
    x,y,w,h = bbox
    if w<=0 or h<=0 or x<0 or y<0: return img
    H,W = img.shape[:2]
    x1=_clamp(int((x-margin)*W),0,W-1); y1=_clamp(int((y-margin)*H),0,H-1)
    x2=_clamp(int((x+w+margin)*W),1,W); y2=_clamp(int((y+h+margin)*H),1,H)
    if x2<=x1 or y2<=y1: return img
    return img[y1:y2,x1:x2,:]

def rrn(path,wh,bbox=None,margin=0.1):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(path)
    im = crop(im,bbox,margin)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if wh is not None: im = cv2.resize(im,wh,interpolation=cv2.INTER_LINEAR)
    im = im.astype(np.float32)/255.0
    im = (im-IMAGENET_MEAN)/IMAGENET_STD
    return im.transpose(2,0,1)

@torch.no_grad()
def slide_logits(model, device, frames_dir, seq_len, stride, wh, bbox, margin):
    files = list_images_sorted(frames_dir)
    n = len(files)
    acc = np.zeros((n,9),dtype=np.float32)
    cnt = np.zeros((n,1),dtype=np.float32)
    starts = list(range(0,max(1,n-seq_len+1),stride)) or [0]
    for s in starts:
        clip = files[s:min(n,s+seq_len)]
        if len(clip) < seq_len: clip += [files[-1]]*(seq_len-len(clip))
        x = np.stack([rrn(p,wh,bbox,margin) for p in clip],axis=0)
        x = torch.from_numpy(x).unsqueeze(0).to(device)
        logits = model(x).squeeze(0).cpu().numpy()  # (T,9)
        T = logits.shape[0]
        for i in range(T):
            g = s+i
            if g>=n: break
            acc[g]+=logits[i]; cnt[g]+=1.0
    acc = acc/np.maximum(cnt,1.0)
    try:
        from scipy.ndimage import uniform_filter1d
        acc = uniform_filter1d(acc,size=5,axis=0,mode="nearest")
    except Exception:
        pass
    return acc

def dp_monotonic(scores):
    """scores: (n,8). Return increasing indices for 8 events via DP."""
    n,E = scores.shape
    dp = np.full((n,E), -1e30, dtype=np.float32)
    prev = np.full((n,E), -1, dtype=np.int32)
    dp[:,0] = scores[:,0]
    for e in range(1,E):
        best = -1e30; best_t = -1
        prefix = np.empty(n, dtype=np.float32)
        argmax = np.empty(n, dtype=np.int32)
        for t in range(n):
            if dp[t,e-1] > best:
                best = dp[t,e-1]; best_t = t
            prefix[t] = best; argmax[t] = best_t
        for t in range(n):
            if argmax[t] >= 0 and argmax[t] < t:
                dp[t,e] = prefix[t] + scores[t,e]
                prev[t,e] = argmax[t]
    end_t = int(np.argmax(dp[:,E-1]))
    path = [0]*E; t = end_t
    for e in range(E-1,-1,-1):
        path[e] = t
        t = prev[t,e] if e>0 else -1
    return path

def parse_list(x):
    try: return json.loads(x)
    except Exception: return ast.literal_eval(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--frames_root", default="data/golfdb_frames")
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=160)
    ap.add_argument("--bbox_margin", type=float, default=0.10)
    # 下面这些只作为“兜底”，优先从 ckpt['args'] 读取
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
    total_c = 0; total_t = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        frames_dir = row["frames_dir"]
        if not os.path.isabs(frames_dir) and not os.path.exists(frames_dir):
            frames_dir = os.path.join(args.frames_root, os.path.basename(frames_dir))

        gt = parse_list(row["events"])
        gt = [int(e) if e is not None else -1 for e in gt]
        gt = gt[:8] + [-1]*max(0,8-len(gt))

        bbox = None
        if "bbox" in df.columns and not pd.isna(row["bbox"]):
            bb = parse_list(row["bbox"])
            if isinstance(bb,(list,tuple)) and len(bb)==4:
                bbox = [float(bb[0]),float(bb[1]),float(bb[2]),float(bb[3])]

        logits = slide_logits(model, device, frames_dir, args.seq_len, args.stride, (args.width,args.height), bbox, args.bbox_margin)
        scores = logits[:,1:9]  # drop background
        pred = dp_monotonic(scores)

        c,t = compute_pce_for_video(pred, gt, k=args.k)
        total_c += c; total_t += t

    pce = total_c/total_t if total_t>0 else 0.0
    print(f"PCE @ k={args.k}: {pce*100:.2f}%  ({int(total_c)}/{int(total_t)})")

if __name__ == "__main__":
    main()
