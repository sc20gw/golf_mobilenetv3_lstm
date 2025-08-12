# src/per_event_metrics.py  (with tqdm progress bar + ckpt hyperparams)
import argparse, os, json, ast, numpy as np, torch, cv2, pandas as pd
from model import MobileNetV3LSTM
from utils import load_checkpoint
from tqdm import tqdm

IMAGENET_MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229,0.224,0.225], dtype=np.float32)
IMG_EXTS = (".jpg",".jpeg",".png",".bmp")

def list_images_sorted(d):
    files = [f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)]
    import re
    files = sorted(files, key=lambda fn: (0,int(re.findall(r"\d+",fn)[-1])) if re.findall(r"\d+",fn) else (1,fn.lower()))
    return [os.path.join(d,f) for f in files]

def crop(im,bbox,margin=0.1):
    if bbox is None: return im
    x,y,w,h=bbox
    if w<=0 or h<=0 or x<0 or y<0: return im
    H,W=im.shape[:2]
    x1=max(0,int((x-margin)*W)); y1=max(0,int((y-margin)*H))
    x2=min(W,int((x+w+margin)*W)); y2=min(H,int((y+h+margin)*H))
    if x2<=x1 or y2<=y1: return im
    return im[y1:y2,x1:x2,:]

def rrn(path,wh,bbox=None,margin=0.1):
    im=cv2.imread(path,cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(path)
    im=crop(im,bbox,margin)
    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    if wh is not None: im=cv2.resize(im,wh,interpolation=cv2.INTER_LINEAR)
    im=im.astype(np.float32)/255.0
    im=(im-IMAGENET_MEAN)/IMAGENET_STD
    return im.transpose(2,0,1)

@torch.no_grad()
def infer_logits(model, device, frames_dir, seq_len, stride, wh, bbox, margin):
    files=list_images_sorted(frames_dir); n=len(files)
    acc=np.zeros((n,9),dtype=np.float32); cnt=np.zeros((n,1),dtype=np.float32)
    starts=list(range(0,max(1,n-seq_len+1),stride)) or [0]
    for s in starts:
        clip=files[s:min(n,s+seq_len)]
        if len(clip)<seq_len: clip+= [files[-1]]*(seq_len-len(clip))
        x=np.stack([rrn(p,wh,bbox,margin) for p in clip],axis=0)
        x=torch.from_numpy(x).unsqueeze(0).to(device)
        logits=model(x).squeeze(0).cpu().numpy()
        T=logits.shape[0]
        for i in range(T):
            g=s+i
            if g>=n: break
            acc[g]+=logits[i]; cnt[g]+=1.0
    acc=acc/np.maximum(cnt,1.0)
    try:
        from scipy.ndimage import uniform_filter1d
        acc=uniform_filter1d(acc,size=5,axis=0,mode="nearest")
    except Exception:
        pass
    return acc

def parse_list(x):
    try: return json.loads(x)
    except Exception: return ast.literal_eval(x)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True)
    ap.add_argument("--test_csv",required=True)
    ap.add_argument("--frames_root",default="data/golfdb_frames")
    ap.add_argument("--seq_len",type=int,default=64)
    ap.add_argument("--stride",type=int,default=8)
    ap.add_argument("--k",type=int,default=5)
    ap.add_argument("--width",type=int,default=160)
    ap.add_argument("--height",type=int,default=160)
    ap.add_argument("--bbox_margin",type=float,default=0.10)
    args=ap.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt=load_checkpoint(args.ckpt, map_location=device)
    ck_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    lstm_hidden = int(ck_args.get("lstm_hidden", 512))
    lstm_layers = int(ck_args.get("lstm_layers", 1))
    bidirectional = bool(ck_args.get("bidirectional", False))
    print(f"[i] Using model hyperparams -> hidden={lstm_hidden}, layers={lstm_layers}, bidirectional={bidirectional}", flush=True)

    model=MobileNetV3LSTM(
        num_events=9,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        bidirectional=bidirectional,
        pretrained=False
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    df=pd.read_csv(args.test_csv)
    ev_hit=np.zeros(8,dtype=np.int64); ev_tot=np.zeros(8,dtype=np.int64)
    names=["Address","Toe-up","Mid-backswing","Top","Mid-downswing","Impact","Mid-follow-through","Finish"]

    # 带进度条的循环
    for _,r in tqdm(df.iterrows(), total=len(df), desc="Per-event eval", ncols=80):
        frames_dir=r["frames_dir"]
        if not os.path.isabs(frames_dir) and not os.path.exists(frames_dir):
            frames_dir=os.path.join(args.frames_root, os.path.basename(frames_dir))
        gt=parse_list(r["events"])
        gt=[int(e) if e is not None else -1 for e in gt]
        gt=gt[:8]+[-1]*max(0,8-len(gt))
        bbox=None
        if "bbox" in df.columns and not pd.isna(r["bbox"]):
            bb=parse_list(r["bbox"])
            if isinstance(bb,(list,tuple)) and len(bb)==4:
                bbox=[float(bb[0]),float(bb[1]),float(bb[2]),float(bb[3])]

        logits=infer_logits(model, device, frames_dir, args.seq_len, args.stride, (args.width,args.height), bbox, args.bbox_margin)
        scores=logits[:,1:9]
        pred=[int(np.argmax(scores[:,e])) for e in range(8)]
        for e,(p,g) in enumerate(zip(pred,gt)):
            if g<0: continue
            ev_tot[e]+=1
            if abs(p-g) <= args.k: ev_hit[e]+=1

    print("Per-event PCE@{}:".format(args.k))
    for i,n in enumerate(names):
        rate = 100.0*ev_hit[i]/ev_tot[i] if ev_tot[i]>0 else 0.0
        print(f"{i}-{n}: {rate:.2f}%  ({ev_hit[i]}/{ev_tot[i]})")

    os.makedirs("logs",exist_ok=True)
    with open("logs/per_event.txt","w",encoding="utf-8") as f:
        for i,n in enumerate(names):
            rate = 100.0*ev_hit[i]/ev_tot[i] if ev_tot[i]>0 else 0.0
            f.write(f"{i}-{n}\t{rate:.2f}\t{ev_hit[i]}\t{ev_tot[i]}\n")
    print("[i] saved logs/per_event.txt")

if __name__ == "__main__":
    main()
