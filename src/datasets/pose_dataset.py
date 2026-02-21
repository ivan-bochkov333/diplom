"""
Unified pose dataset for loading (image, position, orientation) triplets
from a CSV + images directory structure.
"""

import os
import csv

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from src.utils.normalize import compute_normalization_params, normalize_poses


class PoseSample:
    __slots__ = ("frame", "xyz", "q", "scene", "_img_path")

    def __init__(self, frame: str, xyz: np.ndarray, q: np.ndarray, scene: str = ""):
        self.frame = frame
        self.xyz = xyz.astype(np.float32)
        self.q = q.astype(np.float32)
        self.scene = scene


class PoseDataset(Dataset):
    def __init__(self, samples, images_dir, transform=None, scene_id: int = 0):
        self.samples = samples
        self.images_dir = images_dir
        self.transform = transform
        self.scene_id = scene_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = os.path.join(self.images_dir, f"{s.frame}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.images_dir, f"{s.frame}.png")
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        xyz = torch.from_numpy(s.xyz)
        q = torch.from_numpy(s.q)
        return img, xyz, q, self.scene_id


def load_samples(csv_path: str, images_dir: str, scene: str = "",
                 normalize: bool = False):
    """
    Load pose samples from CSV and optionally normalize positions.
    Returns (samples, norm_params) where norm_params is (center, scale) or None.
    """
    samples = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = row["frame"]

            found = False
            for ext in (".jpg", ".png"):
                if os.path.exists(os.path.join(images_dir, f"{frame}{ext}")):
                    found = True
                    break
            if not found:
                continue

            xyz = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])],
                           dtype=np.float32)
            q = np.array([float(row["qw"]), float(row["qx"]),
                          float(row["qy"]), float(row["qz"])], dtype=np.float32)
            q = q / (np.linalg.norm(q) + 1e-8)

            samples.append(PoseSample(frame, xyz, q, scene))

    samples.sort(key=lambda s: s.frame)

    norm_params = None
    if normalize and samples:
        positions = np.array([s.xyz for s in samples])
        center, scale = compute_normalization_params(positions)
        norm_params = {"center": center, "scale": scale}
        for s in samples:
            s.xyz = normalize_poses(s.xyz.reshape(1, 3), center, scale).flatten()

    return samples, norm_params
