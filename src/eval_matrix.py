# src/eval_matrix.py
import argparse, csv, os, subprocess, sys, datetime

def run(cmd):
    print(">>", " ".join(cmd))
    out = subprocess.check_output(cmd, text=True)
    print(out)
    return out

def parse_pce(text):
    import re
    m = re.search(r"PCE @ k=\d+:\s*([\d\.]+)%\s*\((\d+)\/(\d+)\)", text)
    if not m: return None
    return float(m.group(1)), int(m.group(2)), int(m.group(3))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--frames_root", default="data/golfdb_frames")
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=160)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tag", default="exp")
    # >>> 新增：把模型结构也传给评估脚本
    ap.add_argument("--lstm_hidden", type=int, default=512)
    ap.add_argument("--lstm_layers", type=int, default=1)
    ap.add_argument("--bidirectional", action="store_true")
    args = ap.parse_args()

    os.makedirs("logs", exist_ok=True)
    rows = []

    def add_model_args(cmd:list):
        cmd += ["--lstm_hidden", str(args.lstm_hidden),
                "--lstm_layers", str(args.lstm_layers)]
        if args.bidirectional:
            cmd += ["--bidirectional"]
        return cmd

    # argmax 解码：stride 16/8/4
    for stride in [16,8,4]:
        cmd = [sys.executable,"src/eval_full.py",
               "--ckpt", args.ckpt,
               "--test_csv", args.test_csv,
               "--frames_root", args.frames_root,
               "--seq_len", str(args.seq_len),
               "--stride", str(stride),
               "--k", str(args.k),
               "--width", str(args.width),
               "--height", str(args.height),
               "--bbox_margin","0.10"]
        cmd = add_model_args(cmd)
        out = run(cmd)
        p = parse_pce(out)
        if p: rows.append({"tag":args.tag,"decoder":"argmax","stride":stride,"pce":p[0],"hit":p[1],"total":p[2]})

    # DP 解码：stride 8/4
    for stride in [8,4]:
        cmd = [sys.executable,"src/eval_full_dp.py",
               "--ckpt", args.ckpt,
               "--test_csv", args.test_csv,
               "--frames_root", args.frames_root,
               "--seq_len", str(args.seq_len),
               "--stride", str(stride),
               "--k", str(args.k),
               "--width", str(args.width),
               "--height", str(args.height),
               "--bbox_margin","0.10"]
        cmd = add_model_args(cmd)
        out = run(cmd)
        p = parse_pce(out)
        if p: rows.append({"tag":args.tag,"decoder":"dp","stride":stride,"pce":p[0],"hit":p[1],"total":p[2]})

    # 追加写入总表
    out_csv = "logs/results_summary.csv"
    newfile = not os.path.exists(out_csv)
    with open(out_csv,"a",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time","tag","decoder","stride","pce","hit","total","ckpt","model"])
        if newfile: w.writeheader()
        for r in rows:
            r["time"] = datetime.datetime.now().isoformat(timespec="seconds")
            r["ckpt"] = args.ckpt
            r["model"] = f"hidden={args.lstm_hidden},layers={args.lstm_layers},bi={args.bidirectional}"
            w.writerow(r)
    print(f"[i] Appended {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    main()
