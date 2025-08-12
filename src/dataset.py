# src/dataset.py
# -*- coding: utf-8 -*-
"""
GolfDB dataset for softmax training (9 classes: 0=background, 1..8=events).

Each CSV row is expected to contain:
- frames_dir : path to a folder with extracted frames (frame_XXXX.jpg)
- events     : list-like of 8 integers (frame indices of the 8 swing events)
- bbox       : optional [x, y, w, h] in normalized coordinates (relative to full frame)

This dataset returns:
- frames : FloatTensor (T, 3, H, W), ImageNet-normalized
- target : LongTensor  (T,) with values in [0..8]

Key features:
- Event-centered window sampling (train only)
- Optional *early-bias* so that early events (1..4) are sampled more often as centers
- Label dilation with radius=1 (configurable)
- Optional bbox crop with margin
- Lightweight per-clip augmentation (flip + small affine), train only
"""

import os
import re
import cv2
import json
import ast
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------- Normalization (ImageNet) ----------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ---------- Small utilities ----------
def _natural_sort_key(fn: str):
    """Sort file names by the last number inside (frame_12.jpg < frame_100.jpg)."""
    m = re.findall(r"\d+", fn)
    return (0, int(m[-1])) if m else (1, fn.lower())

def list_images_sorted(folder: str):
    """List image files in numeric order."""
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]
    files = sorted(files, key=_natural_sort_key)
    return [os.path.join(folder, f) for f in files]

def parse_list_literal(text):
    """Parse a JSON or Python-literal list from a CSV cell."""
    if isinstance(text, (list, tuple)):
        return list(text)
    try:
        return json.loads(text)
    except Exception:
        return ast.literal_eval(text)

def clamp_int(v, lo, hi):
    return max(lo, min(hi, v))

def safe_join(frames_root: str, frames_dir: str):
    """
    If frames_dir is not an existing absolute path, join with frames_root/basename.
    Keeps absolute/existing paths intact.
    """
    if os.path.isabs(frames_dir) and os.path.exists(frames_dir):
        return frames_dir
    if os.path.exists(frames_dir):
        return frames_dir
    return os.path.join(frames_root, os.path.basename(frames_dir))


# ---------- Geometry & augmentation ----------
def crop_by_bbox(img_bgr: np.ndarray, bbox, margin: float = 0.1):
    """Crop by normalized bbox [x, y, w, h] with extra margin; fallback to full image if invalid."""
    if bbox is None:
        return img_bgr
    x, y, w, h = bbox
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        return img_bgr
    H, W = img_bgr.shape[:2]
    x1 = clamp_int(int((x - margin) * W), 0, W - 1)
    y1 = clamp_int(int((y - margin) * H), 0, H - 1)
    x2 = clamp_int(int((x + w + margin) * W), 1, W)
    y2 = clamp_int(int((y + h + margin) * H), 1, H)
    if x2 <= x1 or y2 <= y1:
        return img_bgr
    return img_bgr[y1:y2, x1:x2, :]

def _affine_params_small(h, w):
    """Sample a small affine transform: ±5° rotation, ±5% scale and translation."""
    angle = random.uniform(-5, 5)
    scale = 1.0 + random.uniform(-0.05, 0.05)
    tx = random.uniform(-0.05, 0.05) * w
    ty = random.uniform(-0.05, 0.05) * h
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    M[:, 2] += (tx, ty)
    return M

def augment_clip_consistently(frames_rgb: np.ndarray):
    """
    Apply the SAME random augmentation to all frames of the clip.
    frames_rgb: (T, H, W, 3) in [0,1], float32
    Augmentations: 50% horizontal flip; small affine (rot/scale/shift).
    """
    T, H, W, _ = frames_rgb.shape
    out = frames_rgb

    # 50% horizontal flip (consistent across the clip)
    if random.random() < 0.5:
        out = np.ascontiguousarray(out[:, :, ::-1, :])

    # small affine (rotation/scale/shift), consistent across the clip
    M = _affine_params_small(H, W)
    warped = np.empty_like(out)
    for t in range(T):
        warped[t] = cv2.warpAffine(out[t], M, (W, H),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
    return warped


# ---------- Label building ----------
def build_frame_targets_softmax(local_events, seq_len: int, radius: int = 1):
    """
    Create per-frame softmax targets with dilation.
    local_events: list of (event_id, local_frame_idx) for events inside [0, T).
                  event_id in [1..8] (softmax class id)
    Return LongTensor of shape (T,), values in [0..8]
      - background=0
      - each event is dilated by 'radius' on both sides
      - if overlapping, choose the event with smaller |t - le|
        (ties -> smaller event id)
    """
    target = np.zeros((seq_len,), dtype=np.int64)
    best_dist = np.full((seq_len,), 1e9, dtype=np.float32)

    for ev_id, le in local_events:
        lo = max(0, le - radius)
        hi = min(seq_len - 1, le + radius)
        for t in range(lo, hi + 1):
            d = abs(t - le)
            if d < best_dist[t] or (d == best_dist[t] and ev_id < target[t]):
                target[t] = ev_id
                best_dist[t] = d

    return torch.from_numpy(target)  # (T,)


# ---------- Dataset ----------
class GolfSwingDataset(Dataset):
    """
    Softmax training/eval dataset for GolfDB.

    Args
    ----
    csv_path : str, path to split CSV (train/val/test)
    frames_root : str, root dir for frames (used if CSV has relative frames_dir)
    split : "train" | "val" | "test"
    seq_len : int, number of frames per sample window (e.g., 64)
    width, height : target spatial size (e.g., 160x160)
    bbox_margin : float, extend bbox crop by this fraction on each side
    label_radius : int, event dilation radius (default 1)
    center_prob : float, prob. to center the sampled window at a random event (train only)
    one_based : bool, whether CSV events are 1-based indices; if True, convert to 0-based
    augment : bool, enable consistent per-clip augmentation (train only)
    early_bias : bool, if True, prefer early events (1..4) when choosing the center event (train only)
    early_bias_weight : float, multiplicative weight for early events in sampling
    """

    def __init__(
        self,
        csv_path: str,
        frames_root: str = "data/golfdb_frames",
        split: str = "train",
        seq_len: int = 64,
        width: int = 160,
        height: int = 160,
        bbox_margin: float = 0.10,
        label_radius: int = 1,
        center_prob: float = 0.7,
        one_based: bool = True,
        augment: bool = False,
        early_bias: bool = True,          # NEW: prefer events 1..4 when centering
        early_bias_weight: float = 3.0,   # NEW: weight multiplier for events 1..4
    ):
        super().__init__()
        assert split in ("train", "val", "test")
        self.csv_path = csv_path
        self.frames_root = frames_root
        self.split = split
        self.seq_len = int(seq_len)
        self.W = int(width)
        self.H = int(height)
        self.bbox_margin = float(bbox_margin)
        self.label_radius = int(label_radius)
        self.center_prob = float(center_prob if split == "train" else 0.0)  # no bias for val/test
        self.one_based = bool(one_based)
        self.augment = bool(augment if split == "train" else False)         # never augment val/test
        self.early_bias = bool(early_bias if split == "train" else False)   # train only
        self.early_bias_weight = float(early_bias_weight)

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(self.csv_path)

        df = pd.read_csv(self.csv_path)
        needed = {"frames_dir", "events"}
        if not needed.issubset(set(df.columns)):
            raise ValueError(f"{self.csv_path} must contain columns: {needed}")

        self.rows = df.to_dict(orient="records")

    def __len__(self):
        return len(self.rows)

    def _read_clip(self, frames_dir: str, bbox):
        """
        Sample a start index and read a (seq_len) clip from frames_dir.
        Returns:
          frames_rgb : (T, H, W, 3) in [0,1], float32 (RGB)
          local_events : list of (event_id, local_idx) inside [0..T)
        """
        frames_dir = safe_join(self.frames_root, frames_dir)
        files = list_images_sorted(frames_dir)
        n = len(files)
        if n == 0:
            raise RuntimeError(f"No images in {frames_dir}")

        # Parse bbox
        if bbox is not None:
            bbox = parse_list_literal(bbox)
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                bbox = None

        # Parse global events (convert to 0-based if needed, clamp to [0..n-1])
        row_events = []
        ev_list = parse_list_literal(self.cur_row["events"])
        ev_list = list(ev_list)[:8] + [-1] * max(0, 8 - len(ev_list))
        for i, e in enumerate(ev_list):
            try:
                e = int(e)
            except Exception:
                e = -1
            if e >= 0:
                if self.one_based and e > 0:
                    e = e - 1
                e = clamp_int(e, 0, max(0, n - 1))
                row_events.append((i + 1, e))  # (event_id in [1..8], frame_idx)

        # Choose window start
        if self.split == "train" and self.center_prob > 0 and len(row_events) > 0 and random.random() < self.center_prob:
            # >>> Early-bias sampling: upweight events 1..4 (Address..Top)
            if self.early_bias:
                weights = [self.early_bias_weight if (1 <= ev_id <= 4) else 1.0 for (ev_id, _) in row_events]
            else:
                weights = [1.0] * len(row_events)
            ev_id, ev_idx = random.choices(row_events, weights=weights, k=1)[0]
            ideal = ev_idx - self.seq_len // 2 + random.randint(-4, 4)
            start = clamp_int(ideal, 0, max(0, n - self.seq_len))
        else:
            if self.split == "train":
                start = random.randint(0, max(0, n - self.seq_len))
            else:
                # For val/test, pick the middle window deterministically
                start = clamp_int((n - self.seq_len) // 2, 0, max(0, n - self.seq_len))

        # Collect T paths; pad by repeating the last frame if the video is too short
        clip_files = files[start:min(n, start + self.seq_len)]
        if len(clip_files) < self.seq_len:
            clip_files += [files[-1]] * (self.seq_len - len(clip_files))

        # Read + crop + resize -> RGB [0,1]
        frames = []
        for p in clip_files:
            im = cv2.imread(p, cv2.IMREAD_COLOR)  # BGR
            if im is None:
                raise FileNotFoundError(p)
            im = crop_by_bbox(im, bbox, margin=self.bbox_margin)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            frames.append(im.astype(np.float32) / 255.0)
        frames_rgb = np.stack(frames, axis=0)  # (T,H,W,3)

        # Augmentation (train only), apply the SAME transform to all frames in this clip
        if self.augment:
            frames_rgb = augment_clip_consistently(frames_rgb)

        # Build local event list inside this window
        local_events = []
        for ev_id, idx in row_events:
            le = idx - start
            if 0 <= le < self.seq_len:
                local_events.append((ev_id, int(le)))

        return frames_rgb, local_events

    def _to_tensor_normalized(self, frames_rgb: np.ndarray):
        """
        Convert (T, H, W, 3) [0,1] to FloatTensor (T, 3, H, W) normalized by ImageNet stats.
        """
        x = frames_rgb.transpose(0, 3, 1, 2)  # (T,3,H,W)
        x = (x - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
        return torch.from_numpy(x.astype(np.float32))

    def __getitem__(self, idx: int):
        self.cur_row = self.rows[idx]
        frames_dir = self.cur_row["frames_dir"]
        bbox = self.cur_row["bbox"] if "bbox" in self.cur_row else None

        frames_rgb, local_events = self._read_clip(frames_dir, bbox)

        # Targets with dilation radius
        target = build_frame_targets_softmax(local_events, self.seq_len, radius=self.label_radius)

        # To normalized tensor
        frames = self._to_tensor_normalized(frames_rgb)  # (T,3,H,W)

        return frames, target

    def __repr__(self) -> str:
        return (f"GolfSwingDataset(split={self.split}, seq_len={self.seq_len}, "
                f"size=({self.H},{self.W}), augment={self.augment}, "
                f"center_prob={self.center_prob}, early_bias={self.early_bias}, "
                f"early_bias_weight={self.early_bias_weight})")
