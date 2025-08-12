# Golf Swing Sequencing on GolfDB with MobileNetV3 + BiLSTM (PyTorch)

This repository implements golf swing event sequencing on GolfDB using a MobileNetV3 backbone (ImageNet-pretrained) and a BiLSTM head (hidden=256, layers=1, bidirectional=True) with a 9-class softmax output (0=background, 1..8=Address→Finish).  
Evaluation uses **PCE@k (k=5)**: a prediction is correct if the absolute frame error is ≤ 5.

---

## Key Facts

- **Splits:**  
  `data/splits_monotonic/{train,val,test}.csv` with 1-based event indices.

- **Inference:**  
  Argmax and DP decoding; test strides: 16/8/4 (denser → better).

- **Input:**  
  Bbox-cropped RGB (`--bbox_margin 0.10`), resize 160×160, ImageNet normalization.

- **Environment:**  
  - Windows 11  
  - Python 3.11  
  - PyTorch 2.5.1+cu121  
  - RTX 4060 Laptop 8GB  
  - (see `logs/env_info.json`)

---

## 1. Quick Start

### 1.1 Setup

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 1.2 Data Layout

- **Frames:**  
  `data/golfdb_frames/<video_id>/frame_XXXX.jpg` (starts at `frame_0001.jpg`)

- **Splits:**  
  `data/splits_monotonic/train.csv`, `val.csv`, `test.csv` (1-based events)

---

## 2. Evaluate (No Retraining)

Replace `--ckpt` with your checkpoint. Two checkpoints used in our analysis:
- `models_softmax_bi_aug_early_e20/checkpoint_latest.pth` (early-event-biased sampling)
- `models_softmax_bi_aug_clswt_e20/checkpoint_latest.pth` (class-weighted loss)

### 2.1 Argmax Decoding (single line)

```bash
python src/eval_full.py \
  --ckpt models_softmax_bi_aug_early_e20/checkpoint_latest.pth \
  --test_csv data/splits_monotonic/test.csv \
  --frames_root data/golfdb_frames \
  --seq_len 64 --stride 4 --k 5 \
  --width 160 --height 160 \
  --bbox_margin 0.10 \
  --lstm_hidden 256 --lstm_layers 1 --bidirectional
```

### 2.2 DP Decoding (single line)

```bash
python src/eval_full_dp.py \
  --ckpt models_softmax_bi_aug_early_e20/checkpoint_latest.pth \
  --test_csv data/splits_monotonic/test.csv \
  --frames_root data/golfdb_frames \
  --seq_len 64 --stride 4 --k 5 \
  --width 160 --height 160 \
  --bbox_margin 0.10 \
  --lstm_hidden 256 --lstm_layers 1 --bidirectional
```

### 2.3 Per-Event PCE@5 (test, stride=4)

```bash
python src/per_event_metrics.py \
  --ckpt models_softmax_bi_aug_early_e20/checkpoint_latest.pth \
  --test_csv data/splits_monotonic/test.csv \
  --frames_root data/golfdb_frames \
  --seq_len 64 --stride 4 --k 5 \
  --width 160 --height 160 \
  --bbox_margin 0.10 \
  --lstm_hidden 256 --lstm_layers 1 --bidirectional
```

---

## 3. Key Results

### 3.1 Overall PCE@5 (%) on Test (higher is better)

*(From `logs/experiments_overview.csv`)*

| Tag                        | Argmax s=16 | Argmax s=8 | Argmax s=4 | DP s=8 | DP s=4 | Val best |
|----------------------------|-------------|------------|------------|--------|--------|----------|
| bi256_aug_early_e20        | **29.61**   | **42.21**  | **51.39**  | **42.25** | **51.75** | **0.5808** |
| bi256_aug_clswt_e20        | **28.25**   | **41.43**  | **51.00**  | **41.46** | **51.00** | **0.7078** |
| bi256_aug_lr3e-4_e30       | 25.04       | 38.04      | 46.29      | 38.00 | 45.89 | (see CSV) |
| bi_256_lr3e-4_e30          | 28.93       | 41.93      | 48.75      | 41.89 | 48.25 | (see CSV) |

### 3.2 Per-Event PCE@5 (%) on Test, stride=4 (DP)

*(From `logs/per_event.txt`, 350 swings × 8 events)*

| Event              | PCE@5  | Correct/350 |
|--------------------|-------:|------------:|
| Address            |   0.57 |     2 / 350 |
| Toe-up             |  10.00 |    35 / 350 |
| Mid-backswing      |  24.86 |    87 / 350 |
| Top                |  34.57 |   121 / 350 |
| Mid-downswing      |  60.00 |   210 / 350 |
| Impact             |  87.14 |   305 / 350 |
| Mid-follow-through |  96.86 |   339 / 350 |
| Finish             |  97.14 |   340 / 350 |

**Takeaways:** Early/static events (Address, Toe-up, Mid-backswing) are hard; dynamic/late phases (Impact→Finish) are strong.

---

## 4. Diagnostics

- **Curves:**  
  `logs/figs/val_curve_*.png` (or replot from `logs/val_curve_*.csv`)

- **Case studies (GT vs Pred):**  
  - `logs/diag_row_0.txt` (medium)  
  - `logs/diag_row_147.txt` (good)  
  - `logs/diag_row_150.txt` (poor)

---

## 5. Directory Hints

```text
golf_mobilenetv3_lstm/
  data/
    golfdb_frames/...
    splits_monotonic/{train,val,test}.csv
  logs/
    experiments_overview.csv
    per_event.txt
    figs/val_curve_*.png
  models_softmax_bi_aug_early_e20/checkpoint_latest.pth
  src/*.py
```

---

## 6. References

- McNally et al., GolfDB: A Video Database for Golf Swing Sequencing, CVPRW 2019.
- Deep learning-based golf swing sequencing from videos.

---

## 7. License

Released under the MIT License (see [LICENSE](LICENSE)).
