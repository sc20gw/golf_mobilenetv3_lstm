# src/train_utils.py
# Utilities for dataloaders, seeding, and backbone freezing.

import os, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import GolfSwingDataset

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # faster on GPU for fixed shapes

def create_dataloaders(args):
    """
    Build train/val dataloaders. Pass `augment` only for train split.
    """
    train_ds = GolfSwingDataset(
        csv_path=args.train_csv,
        frames_root=args.frames_root,
        split="train",
        seq_len=args.seq_len,
        width=args.width, height=args.height,
        bbox_margin=0.10,
        label_radius=1,
        center_prob=0.7,
        one_based=True,
        augment=getattr(args, "augment", False),
    )
    val_ds = GolfSwingDataset(
        csv_path=args.val_csv,
        frames_root=args.frames_root,
        split="val",
        seq_len=args.seq_len,
        width=args.width, height=args.height,
        bbox_margin=0.10,
        label_radius=1,
        center_prob=0.0,
        one_based=True,
        augment=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader

def freeze_backbone(model, freeze: bool):
    """
    If True: freeze conv backbone; keep LSTM + classifier trainable.
    """
    if not freeze:
        return 0, 0
    n_freeze = 0
    for name, p in model.backbone.named_parameters():
        p.requires_grad_(False); n_freeze += 1
    n_total = sum(1 for _ in model.backbone.parameters())
    return n_freeze, n_total
