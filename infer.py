#!/usr/bin/env python3
"""
Inference script: predict camera poses for a directory of images.
No ground truth required — purely inference mode.

Usage:
    python infer.py --checkpoint outputs/posenet/best.pth --images_dir new_dataset/images
    python infer.py --checkpoint outputs/posenet/best.pth --images_dir new_dataset/images --gt_csv new_dataset/poses.csv
"""

import os
import csv
import argparse

import yaml
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from src.models import build_model
from src.datasets.colmap_dataset import get_transforms
from src.utils.metrics import compute_metrics
from src.utils.visualization import plot_trajectory_3d


class ImageFolderDataset(Dataset):
    def __init__(self, images_dir, transform):
        self.images_dir = images_dir
        self.transform = transform
        self.frames = []
        for fname in sorted(os.listdir(images_dir)):
            if fname.lower().endswith((".jpg", ".png")):
                self.frames.append(os.path.splitext(fname)[0])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        for ext in (".jpg", ".png"):
            path = os.path.join(self.images_dir, f"{frame}{ext}")
            if os.path.exists(path):
                break
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, frame


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Infer camera poses")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--gt_csv", type=str, default=None,
                        help="Optional: compare with ground truth poses")
    parser.add_argument("--scene_id", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    device = get_device()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint["config"]

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    transform = get_transforms("val", cfg["data"].get("image_size", 224))
    dataset = ImageFolderDataset(args.images_dir, transform)

    if len(dataset) == 0:
        print(f"No images found in {args.images_dir}")
        return

    print(f"Model: {cfg['model']['name']}")
    print(f"Running inference on {len(dataset)} images...")

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    out_csv = args.out_csv or os.path.join(os.path.dirname(args.images_dir), "pred_poses.csv")
    predictions = {}

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "tx", "ty", "tz", "qw", "qx", "qy", "qz"])

        with torch.no_grad():
            for imgs, frames in loader:
                imgs = imgs.to(device, non_blocking=True)
                scene_ids = torch.full((imgs.shape[0],), args.scene_id,
                                       dtype=torch.long, device=device)

                xyz_pred, q_pred = model(imgs, scene_id=scene_ids)
                xyz_pred = xyz_pred.cpu().numpy()
                q_pred = q_pred.cpu().numpy()

                for frame, xyz, q in zip(frames, xyz_pred, q_pred):
                    writer.writerow([frame, xyz[0], xyz[1], xyz[2],
                                     q[0], q[1], q[2], q[3]])
                    predictions[frame] = {"xyz": xyz, "q": q}

    print(f"Predictions saved to {out_csv}")

    # Compare with ground truth if provided
    if args.gt_csv and os.path.exists(args.gt_csv):
        print(f"Comparing with ground truth from {args.gt_csv}...")
        gt = {}
        with open(args.gt_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = row["frame"]
                xyz = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])])
                q = np.array([float(row["qw"]), float(row["qx"]),
                              float(row["qy"]), float(row["qz"])])
                q = q / (np.linalg.norm(q) + 1e-8)
                gt[frame] = {"xyz": xyz, "q": q}

        pred_xyz_list, pred_q_list, gt_xyz_list, gt_q_list = [], [], [], []
        for frame, pred in predictions.items():
            if frame in gt:
                pred_xyz_list.append(pred["xyz"])
                pred_q_list.append(pred["q"])
                gt_xyz_list.append(gt[frame]["xyz"])
                gt_q_list.append(gt[frame]["q"])

        if pred_xyz_list:
            metrics = compute_metrics(pred_xyz_list, pred_q_list,
                                      gt_xyz_list, gt_q_list)
            print(f"Matched frames: {len(pred_xyz_list)}")
            print(f"  Median position error: {metrics['median_pos']:.4f}")
            print(f"  Mean position error:   {metrics['mean_pos']:.4f}")
            print(f"  Median rotation error: {metrics['median_rot']:.2f} deg")
            print(f"  Mean rotation error:   {metrics['mean_rot']:.2f} deg")

            if args.plot:
                out_dir = os.path.dirname(out_csv)
                gt_xyz = np.array(gt_xyz_list)
                pred_xyz = np.array(pred_xyz_list)
                plot_trajectory_3d(gt_xyz, pred_xyz,
                                   title=f"{cfg['model']['name']} Inference",
                                   save_path=os.path.join(out_dir, "infer_trajectory.png"))
                print(f"Trajectory plot saved.")
        else:
            print("No matching frames between predictions and GT.")


if __name__ == "__main__":
    main()
