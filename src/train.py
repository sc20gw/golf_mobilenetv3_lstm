# src/train.py
# Training script for MobileNetV3 + (Bi)LSTM on GolfDB (softmax, 9 classes: 0=bg, 1..8 events)
# - Supports AMP, pretrained backbone, optional freezing, and lightweight data augmentation (via dataset.py).
# - Class weights are set to emphasize early events moderately.

import argparse, os, time, math
import torch
import torch.nn as nn
from torch.optim import AdamW
from model import MobileNetV3LSTM
from train_utils import set_seed, create_dataloaders, freeze_backbone

def save_checkpoint(path, model, epoch, val_loss, args_short):
    """Save minimal checkpoint (weights + a few hyper-params for eval-time model restore)."""
    ckpt = {
        "epoch": epoch,
        "val_loss": float(val_loss),
        "model_state": model.state_dict(),
        "args": args_short,  # used by eval scripts to restore lstm configs
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)

def main():
    # ---- Arguments ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--frames_root", default="data/golfdb_frames")

    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)

    ap.add_argument("--lstm_hidden", type=int, default=512)
    ap.add_argument("--lstm_layers", type=int, default=1)
    ap.add_argument("--bidirectional", action="store_true")

    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--augment", action="store_true")  # enable lightweight augmentation (train only)

    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=160)
    ap.add_argument("--save_dir", default="models_softmax")
    ap.add_argument("--print_every", type=int, default=20)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---- Setup ----
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Model ----
    model = MobileNetV3LSTM(
        num_events=9,                      # 0..8 (0=background, 1..8=events)
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
        pretrained=args.pretrained,
    ).to(device)

    # Optionally freeze CNN backbone (train LSTM + head only)
    nf, nt = freeze_backbone(model, args.freeze_backbone)
    if args.freeze_backbone:
        print(f"Backbone frozen (params {nf}/{nt}). Training LSTM + classifier only.")

    # ---- Loss & Optimizer ----
    # Class weights: bg=0.1; early four events get moderate up-weights to improve early-phase recall.
    # Order: [bg, Address, Toe-up, Mid-backswing, Top, Mid-down, Impact, Mid-FT, Finish]
    class_weights = torch.tensor(
        [0.1, 2.0, 1.5, 1.25, 1.10, 1.0, 1.0, 1.0, 1.0],
        dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4
    )

    # ---- Data ----
    # create_dataloaders() will pass augment=True only to the train split if --augment is set.
    train_loader, val_loader = create_dataloaders(args)

    # AMP (new API, avoids deprecation warnings)
    scaler = torch.amp.GradScaler(device_type) if (args.use_amp and device_type == "cuda") else None

    # ---- Training loop ----
    best_val = math.inf
    best_path = os.path.join(args.save_dir, "checkpoint_best.pth")
    last_path = os.path.join(args.save_dir, "checkpoint_latest.pth")

    # minimal args stored into ckpt for eval scripts (so they can rebuild the same LSTM)
    args_short = {
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "bidirectional": args.bidirectional,
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (x, y) in enumerate(train_loader, start=1):
            # x: (B,T,3,H,W) ; y: (B,T) longs in [0..8]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast(device_type):
                    logits = model(x)                        # (B,T,9)
                    loss = criterion(logits.view(-1, 9), y.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits.view(-1, 9), y.view(-1))
                loss.backward()
                optimizer.step()

            running += loss.item()
            if step % args.print_every == 0:
                avg = running / args.print_every
                print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch}/{args.epochs} "
                      f"Step {step}/{len(train_loader)}  TrainLoss {avg:.4f}", flush=True)
                running = 0.0

        # ---- Validation ----
        model.eval()
        val_loss = 0.0; n_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if scaler is not None:
                    with torch.amp.autocast(device_type):
                        logits = model(x)
                        loss = criterion(logits.view(-1, 9), y.view(-1))
                else:
                    logits = model(x)
                    loss = criterion(logits.view(-1, 9), y.view(-1))
                val_loss += loss.item(); n_batches += 1
        val_loss /= max(1, n_batches)
        print(f"Epoch {epoch} done. ValLoss {val_loss:.4f}", flush=True)

        # ---- Save checkpoints ----
        save_checkpoint(last_path, model, epoch, val_loss, args_short)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path, model, epoch, val_loss, args_short)

    print("Training finished.")
    print(f"Best ValLoss: {best_val:.4f}")
    print(f"Saved latest: {last_path}")
    print(f"Saved best:   {best_path}")

if __name__ == "__main__":
    main()
