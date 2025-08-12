# src/compile_report.py
"""
Merge evaluation results (logs/results_summary.csv, produced by eval_matrix.py)
and validation summaries (logs/val_summaries.csv, produced by parse_train_log.py)
into a wide "logs/experiments_overview.csv" for your report.
"""
import argparse, os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="logs/results_summary.csv")
    ap.add_argument("--val", default="logs/val_summaries.csv")
    ap.add_argument("--out", default="logs/experiments_overview.csv")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        raise FileNotFoundError(args.results)
    res = pd.read_csv(args.results)  # time, tag, decoder, stride, pce, hit, total, ckpt
    res["col"] = res["decoder"].astype(str) + "_s" + res["stride"].astype(str)
    wide = res.pivot_table(index="tag", columns="col", values="pce", aggfunc="max")

    expected = ["argmax_s16","argmax_s8","argmax_s4","dp_s8","dp_s4"]
    for c in expected:
        if c not in wide.columns: wide[c] = None
    wide = wide[expected].reset_index()

    if os.path.exists(args.val):
        val = pd.read_csv(args.val)
        val_last = val.sort_values(["tag","time"]).drop_duplicates("tag", keep="last")[["tag","val_best","val_last"]]
        merged = pd.merge(val_last, wide, how="outer", on="tag")
    else:
        merged = wide
        merged["val_best"] = None; merged["val_last"] = None
        merged = merged[["tag","val_best","val_last"] + expected]

    merged = merged[["tag","val_best","val_last","argmax_s16","argmax_s8","argmax_s4","dp_s8","dp_s4"]]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[i] Wrote overview -> {args.out}")
    print(merged)

if __name__ == "__main__":
    main()
