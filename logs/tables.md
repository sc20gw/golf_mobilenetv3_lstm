## Overall comparison (PCE@5, higher is better)

| tag | val_best | argmax_s16 | argmax_s8 | argmax_s4 | dp_s8 | dp_s4 |
|---|---:|---:|---:|---:|---:|---:|
| bi256_aug_clswt_e20 | 0.7078 | 28.25 | 41.43 | 51.0 | 41.46 | 51.0 |
| bi256_aug_early_e20 | 0.5808 | 29.61 | 42.21 | 51.39 | 42.25 | 51.75 |
| bi256_aug_lr3e-4_e30 | 0.4458 | 25.04 | 38.04 | 46.29 | 38.0 | 45.89 |
| bi_256_lr3e-4_e30 |  | 28.93 | 41.93 | 48.75 | 41.89 | 48.25 |

## Per-event PCE@5 on test set

| event | PCE% | hits | total |
|---|---:|---:|---:|
| 0-Address | 0.57 | 2 | 350 |
| 1-Toe-up | 10.00 | 35 | 350 |
| 2-Mid-backswing | 24.86 | 87 | 350 |
| 3-Top | 34.57 | 121 | 350 |
| 4-Mid-downswing | 60.00 | 210 | 350 |
| 5-Impact | 87.14 | 305 | 350 |
| 6-Mid-follow-through | 96.86 | 339 | 350 |
| 7-Finish | 97.14 | 340 | 350 |
