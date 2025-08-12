# src/plot_curves.py
# Plot validation loss curves from logs/val_curve_*.csv into PNGs.

import argparse, os, glob, csv
import matplotlib.pyplot as plt

def load_curve(path):
    xs, ys = [], []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(int(row['epoch']))
            ys.append(float(row['val_loss']))
    return xs, ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="logs/val_curve_*.csv", help="glob pattern for curve csvs")
    ap.add_argument("--outdir", default="logs/figs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    files = sorted(glob.glob(args.glob))
    if not files:
        print("[WARN] no curve files found:", args.glob)
        return

    for path in files:
        xs, ys = load_curve(path)
        tag = os.path.splitext(os.path.basename(path))[0].replace("val_curve_", "")
        plt.figure(figsize=(6,4))
        plt.plot(xs, ys, marker="o")
        plt.title(f"Validation Loss - {tag}")
        plt.xlabel("Epoch")
        plt.ylabel("Val Loss")
        plt.grid(True, alpha=0.3)
        out = os.path.join(args.outdir, f"val_curve_{tag}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print("[i] saved", out)

if __name__ == "__main__":
    main()
