# src/utils.py
"""
General utilities: checkpoint IO and PCE metric helpers.
"""

import os
import torch
import numpy as np


def save_checkpoint(state: dict, save_dir: str, name: str = "checkpoint_latest.pth"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    torch.save(state, path)
    return path


def load_checkpoint(path: str, map_location=None):
    return torch.load(path, map_location=map_location)


def pred_frame_indices_from_logits(logits_2d: np.ndarray):
    """
    logits_2d: (T, E) array
    Return list of length E with argmax over time for each event.
    """
    T, E = logits_2d.shape
    return [int(np.argmax(logits_2d[:, e])) for e in range(E)]


def compute_pce_for_video(pred_indices, gt_indices, k=5):
    """
    pred_indices: list length E
    gt_indices: list length E (use -1 if missing)
    Return (correct_count, total_valid)
    """
    correct = 0
    total = 0
    for p, g in zip(pred_indices, gt_indices):
        if g is None or int(g) < 0:
            continue
        total += 1
        if abs(int(p) - int(g)) <= int(k):
            correct += 1
    return correct, total
