# src/make_report_tables.py
# Produce Markdown tables from experiments_overview.csv and per_event.txt

import argparse, csv, os

def read_overview(path):
    rows=[]
    with open(path, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def read_per_event_txt(path):
    # expects lines like: "0-Address\t1.14\t4\t350"
    out=[]
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts=line.strip().split("\t")
            if len(parts)>=4:
                name=parts[0]
                try:
                    p=float(parts[1]); hit=int(parts[2]); tot=int(parts[3])
                except Exception:
                    continue
                out.append((name,p,hit,tot))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--overview", default="logs/experiments_overview.csv")
    ap.add_argument("--per_event", default="logs/per_event.txt")
    ap.add_argument("--out", default="logs/tables.md")
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ov=read_overview(args.overview)
    pe=read_per_event_txt(args.per_event)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("## Overall comparison (PCE@5, higher is better)\n\n")
        f.write("| tag | val_best | argmax_s16 | argmax_s8 | argmax_s4 | dp_s8 | dp_s4 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in ov:
            f.write(f"| {r['tag']} | {r['val_best'] or ''} | {r['argmax_s16']} | {r['argmax_s8']} | {r['argmax_s4']} | {r['dp_s8']} | {r['dp_s4']} |\n")

        f.write("\n## Per-event PCE@5 on test set\n\n")
        if pe:
            f.write("| event | PCE% | hits | total |\n")
            f.write("|---|---:|---:|---:|\n")
            for name,p,hit,tot in pe:
                f.write(f"| {name} | {p:.2f} | {hit} | {tot} |\n")
        else:
            f.write("_No per-event file found; run per_event_metrics.py first._\n")

    print("[i] wrote", args.out)

if __name__ == "__main__":
    main()
