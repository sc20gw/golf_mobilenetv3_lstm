# src/parse_train_log.py  — auto-detect UTF-8 / UTF-16LE
import re, argparse, os, csv, datetime, codecs

def read_text_auto(path):
    with open(path, 'rb') as f:
        b = f.read()
    # BOM / NULL 字节检测
    if b.startswith(codecs.BOM_UTF16_LE) or b[1:2] == b'\x00':
        return b.decode('utf-16-le', errors='ignore')
    if b.startswith(codecs.BOM_UTF8):
        return b.decode('utf-8', errors='ignore')
    # 兜底按 UTF-8
    try:
        return b.decode('utf-8')
    except UnicodeDecodeError:
        return b.decode('utf-16-le', errors='ignore')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--curve", required=True)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.curve), exist_ok=True)

    text = read_text_auto(args.log)
    pat = re.compile(r"Epoch\s+(\d+)\s+done\. ValLoss\s+([0-9]*\.?[0-9]+)")
    epochs, vals = [], []
    for line in text.splitlines():
        m = pat.search(line)
        if m:
            epochs.append(int(m.group(1)))
            vals.append(float(m.group(2)))

    if epochs:
        with open(args.curve, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["epoch","val_loss"])
            for e,v in zip(epochs, vals): w.writerow([e,v])
        print(f"[i] Saved curve -> {args.curve} ({len(epochs)} points)")
    else:
        print("[WARN] No 'Epoch ... done. ValLoss ...' lines found in", args.log)

    if args.append:
        best = min(vals) if vals else ""
        last = vals[-1] if vals else ""
        out_csv = "logs/val_summaries.csv"
        newfile = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow(["time","tag","val_best","val_last","curve_file","log_file"])
            w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), args.tag, best, last, args.curve, args.log])
        print(f"[i] Appended summary -> {out_csv} (best={best}, last={last})")

if __name__ == "__main__":
    main()
