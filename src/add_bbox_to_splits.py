# src/add_bbox_to_splits.py
"""
Read GolfDB.pkl to fetch normalized bbox per video id, then
add a 'bbox' (json [x,y,w,h] in 0..1) and 'slow' column into your rescaled splits.

Input:
  data/golfDB.pkl
  data/splits_rescaled/train.csv, val.csv, test.csv  (must exist)
Output:
  data/splits_rescaled_bbox/train.csv, val.csv, test.csv
"""
import os, json, ast
import pandas as pd

PKL_PATH = "data/golfDB.pkl"
IN_DIR   = "data/splits_rescaled"
OUT_DIR  = "data/splits_rescaled_bbox"

def parse_events(s):
    try: return json.loads(s)
    except: return ast.literal_eval(s)

def main():
    # load pkl to map id -> bbox, slow
    dfp = pd.read_pickle(PKL_PATH)
    # normalize column names
    if "id" not in dfp.columns:
        raise RuntimeError("PKL must contain 'id' column.")
    if "bbox" not in dfp.columns:
        raise RuntimeError("PKL must contain 'bbox' column.")
    if "slow" not in dfp.columns:
        raise RuntimeError("PKL must contain 'slow' column.")
    id2bbox = {}
    id2slow = {}
    for _, r in dfp.iterrows():
        vid = int(r["id"])
        bb = r["bbox"]
        # bbox is expected as [x, y, w, h] normalized by width/height
        if isinstance(bb, str):
            try: bb = json.loads(bb)
            except: bb = ast.literal_eval(bb)
        id2bbox[vid] = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
        id2slow[vid] = int(r["slow"])

    os.makedirs(OUT_DIR, exist_ok=True)
    for fn in ["train.csv", "val.csv", "test.csv"]:
        in_csv  = os.path.join(IN_DIR, fn)
        out_csv = os.path.join(OUT_DIR, fn)
        if not os.path.exists(in_csv):
            print(f"Missing: {in_csv}")
            continue
        df = pd.read_csv(in_csv)
        bboxes = []
        slows = []
        miss = 0
        for _, r in df.iterrows():
            vid = int(r["video"])
            if vid in id2bbox:
                bboxes.append(json.dumps(id2bbox[vid]))
                slows.append(int(id2slow[vid]))
            else:
                miss += 1
                bboxes.append(json.dumps([-1,-1,-1,-1]))
                slows.append(0)
        df["bbox"] = bboxes
        df["slow"] = slows
        df.to_csv(out_csv, index=False)
        print(f"[{fn}] wrote bbox/slow | missing={miss} -> {out_csv}")

if __name__ == "__main__":
    main()
