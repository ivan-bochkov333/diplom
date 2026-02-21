"""
7Scenes dataset loader (Microsoft Research).
Each scene has train/test splits with RGB images and ground-truth 4x4 pose matrices.

Expected directory structure after download:
  7scenes/
    chess/
      seq-01/ ... seq-06/
        frame-XXXXXX.color.png
        frame-XXXXXX.pose.txt
    fire/
    ...
"""

import os
import re
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms

from .pose_dataset import PoseSample


SEVEN_SCENES_NAMES = [
    "chess", "fire", "heads", "office",
    "pumpkin", "redkitchen", "stairs",
]

# Official train/test splits (sequence numbers)
SPLITS = {
    "chess":      {"train": [1, 2, 4, 6],       "test": [3, 5]},
    "fire":       {"train": [1, 2],              "test": [3, 4]},
    "heads":      {"train": [2],                 "test": [1]},
    "office":     {"train": [1, 3, 4, 5, 8, 10], "test": [2, 6, 7, 9]},
    "pumpkin":    {"train": [2, 3, 6, 8],        "test": [1, 7]},
    "redkitchen": {"train": [1, 2, 5, 7, 8, 11, 13], "test": [3, 4, 6, 12, 14]},
    "stairs":     {"train": [2, 3, 5, 6],        "test": [1, 4]},
}


def _pose_matrix_to_quat_pos(pose_4x4: np.ndarray):
    """
    Convert 4x4 camera-to-world pose matrix to (position, quaternion).
    Quaternion format: (qw, qx, qy, qz).
    """
    R = pose_4x4[:3, :3]
    t = pose_4x4[:3, 3]

    # Rotation matrix to quaternion (Shepperd's method)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    if q[0] < 0:
        q = -q

    return t.astype(np.float32), q


def _load_scene_split(scene_root, scene_name, split, seq_numbers):
    """Load all frames from given sequences for a scene."""
    samples = []
    for seq_num in seq_numbers:
        seq_dir = os.path.join(scene_root, f"seq-{seq_num:02d}")
        if not os.path.isdir(seq_dir):
            continue

        pose_files = sorted([
            f for f in os.listdir(seq_dir)
            if f.endswith(".pose.txt")
        ])

        for pose_file in pose_files:
            frame_id = pose_file.replace(".pose.txt", "")
            img_file = f"{frame_id}.color.png"
            img_path = os.path.join(seq_dir, img_file)

            if not os.path.exists(img_path):
                continue

            pose_path = os.path.join(seq_dir, pose_file)
            pose_4x4 = np.loadtxt(pose_path).reshape(4, 4)

            if np.any(np.isinf(pose_4x4)) or np.any(np.isnan(pose_4x4)):
                continue

            t, q = _pose_matrix_to_quat_pos(pose_4x4)
            sample = PoseSample(
                frame=os.path.join(f"seq-{seq_num:02d}", frame_id),
                xyz=t, q=q, scene=scene_name,
            )
            sample._img_path = img_path
            samples.append(sample)

    return samples


class SevenScenesSceneDataset(Dataset):
    """Dataset for a single 7Scenes scene (train or test split)."""

    def __init__(self, samples, transform=None, scene_id: int = 0):
        self.samples = samples
        self.transform = transform
        self.scene_id = scene_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s._img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        xyz = torch.from_numpy(s.xyz)
        q = torch.from_numpy(s.q)
        return img, xyz, q, self.scene_id


def get_transforms_7scenes(mode: str = "train", image_size: int = 224):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class SevenScenesDataset:
    """
    Full 7Scenes dataset manager.
    Provides per-scene train/test datasets and multi-scene combined datasets.
    """

    def __init__(self, root: str, scenes: list = None,
                 image_size: int = 224, normalize: bool = False):
        self.root = root
        self.scenes = scenes or SEVEN_SCENES_NAMES
        self.image_size = image_size
        self.scene_datasets = {}
        self.norm_params = {}

        for i, scene_name in enumerate(self.scenes):
            scene_root = os.path.join(root, scene_name)
            if not os.path.isdir(scene_root):
                print(f"Warning: scene directory not found: {scene_root}")
                continue

            train_samples = _load_scene_split(
                scene_root, scene_name, "train",
                SPLITS[scene_name]["train"]
            )
            test_samples = _load_scene_split(
                scene_root, scene_name, "test",
                SPLITS[scene_name]["test"]
            )

            if normalize:
                from src.utils.normalize import compute_normalization_params, normalize_poses
                all_pos = np.array([s.xyz for s in train_samples + test_samples])
                center, scale = compute_normalization_params(all_pos)
                self.norm_params[scene_name] = {"center": center, "scale": scale}
                for s in train_samples + test_samples:
                    s.xyz = normalize_poses(s.xyz.reshape(1, 3), center, scale).flatten()

            self.scene_datasets[scene_name] = {
                "train_samples": train_samples,
                "test_samples": test_samples,
                "scene_id": i,
            }

    def get_scene_datasets(self, scene_name: str):
        """Get (train_dataset, test_dataset) for a single scene."""
        info = self.scene_datasets[scene_name]
        train_ds = SevenScenesSceneDataset(
            info["train_samples"],
            transform=get_transforms_7scenes("train", self.image_size),
            scene_id=info["scene_id"],
        )
        test_ds = SevenScenesSceneDataset(
            info["test_samples"],
            transform=get_transforms_7scenes("test", self.image_size),
            scene_id=info["scene_id"],
        )
        return train_ds, test_ds

    def get_multiscene_datasets(self, exclude_scenes=None):
        """
        Get combined train/test datasets across multiple scenes.
        Optionally exclude some scenes (for held-out fine-tuning evaluation).
        """
        exclude = set(exclude_scenes or [])
        train_datasets = []
        test_datasets = []

        for scene_name, info in self.scene_datasets.items():
            if scene_name in exclude:
                continue
            train_ds = SevenScenesSceneDataset(
                info["train_samples"],
                transform=get_transforms_7scenes("train", self.image_size),
                scene_id=info["scene_id"],
            )
            test_ds = SevenScenesSceneDataset(
                info["test_samples"],
                transform=get_transforms_7scenes("test", self.image_size),
                scene_id=info["scene_id"],
            )
            train_datasets.append(train_ds)
            test_datasets.append(test_ds)

        return ConcatDataset(train_datasets), ConcatDataset(test_datasets)
