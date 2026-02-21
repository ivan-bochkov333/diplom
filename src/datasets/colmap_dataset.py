"""
Dataset loader for custom COLMAP-generated datasets.
Wraps the unified PoseDataset with COLMAP-specific directory conventions.
"""

import os

from torchvision import transforms

from .pose_dataset import PoseDataset, load_samples


def get_transforms(mode: str = "train", image_size: int = 224):
    """Get augmentation transforms for train or val/test."""
    if mode == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
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


class ColmapDataset:
    """
    Convenience wrapper for COLMAP-generated scene data.
    Expects: data_root/images/ and data_root/poses.csv
    """

    def __init__(self, data_root: str, scene_name: str = "",
                 normalize: bool = False, image_size: int = 224,
                 train_split: float = 0.8, scene_id: int = 0):
        self.data_root = data_root
        self.images_dir = os.path.join(data_root, "images")
        self.csv_path = os.path.join(data_root, "poses.csv")
        self.scene_name = scene_name or os.path.basename(data_root)

        all_samples, self.norm_params = load_samples(
            self.csv_path, self.images_dir, scene=self.scene_name,
            normalize=normalize
        )

        n_train = int(len(all_samples) * train_split)
        self.train_samples = all_samples[:n_train]
        self.val_samples = all_samples[n_train:]

        self.train_dataset = PoseDataset(
            self.train_samples, self.images_dir,
            transform=get_transforms("train", image_size),
            scene_id=scene_id,
        )
        self.val_dataset = PoseDataset(
            self.val_samples, self.images_dir,
            transform=get_transforms("val", image_size),
            scene_id=scene_id,
        )
