# src/debug_csv_events.py
import json, ast, pandas as pd, os

def parse_events(s):
    try: return json.loads(s)
    except: return ast.literal_eval(s)

def check(csv_path):
    df = pd.read_csv(csv_path)
    bad_hi = bad_lo = 0
    total_ev = 0
    for i, r in df.iterrows():
        n = int(r["n_frames"])
        ev = parse_events(r["events"])[:8]
        for e in ev:
            if e is None: continue
            e = int(e)
            if e < -1: bad_lo += 1
            if e >= n: bad_hi += 1
            if e >= 0: total_ev += 1
    print(f"[{os.path.basename(csv_path)}] total non-negative events: {total_ev}, over-high: {bad_hi}, below -1: {bad_lo}")

def main():
    for p in ["data/splits_rescaled/train.csv", "data/splits_rescaled/val.csv", "data/splits_rescaled/test.csv"]:
        if os.path.exists(p): check(p)
        else: print(f"Missing: {p}")


if __name__ == "__main__":
    main()
