#!/usr/bin/env python3
"""
Evaluation script: loads a trained model and computes metrics against ground truth.

Usage:
    python evaluate.py --checkpoint outputs/posenet/best.pth --data_root my_test_1_dataset
    python evaluate.py --checkpoint outputs/atloc/best.pth --data_root data/7scenes --data_type 7scenes_single --scene chess
"""

import os
import csv
import argparse

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from src.models import build_model
from src.datasets.colmap_dataset import ColmapDataset, get_transforms
from src.datasets.seven_scenes import SevenScenesDataset
from src.utils.metrics import compute_metrics
from src.utils.visualization import plot_trajectory_3d, plot_cumulative_error


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(args):
    device = get_device()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint["config"]

    # Override data settings from CLI
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.data_type:
        cfg["data"]["type"] = args.data_type
    if args.scene:
        cfg["data"]["scene"] = args.scene

    # Build model
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    data_cfg = cfg["data"]
    transform = get_transforms("val", data_cfg.get("image_size", 224))

    # Build evaluation dataset
    if data_cfg["type"] == "colmap":
        ds = ColmapDataset(
            data_root=data_cfg["root"],
            normalize=data_cfg.get("normalize", False),
            image_size=data_cfg.get("image_size", 224),
            train_split=0.0,  # use all data for evaluation
        )
        eval_dataset = ds.val_dataset
    elif data_cfg["type"] in ("7scenes_single", "7scenes"):
        root = data_cfg.get("seven_scenes_root", data_cfg.get("root"))
        scene_name = data_cfg.get("scene", "chess")
        seven = SevenScenesDataset(
            root=root, scenes=[scene_name],
            image_size=data_cfg.get("image_size", 224),
            normalize=data_cfg.get("normalize", False),
        )
        _, eval_dataset = seven.get_scene_datasets(scene_name)
    else:
        raise ValueError(f"Unknown data type: {data_cfg['type']}")

    loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=4,
                        pin_memory=True)

    print(f"Model: {cfg['model']['name']}")
    print(f"Evaluating on {len(eval_dataset)} samples...")

    pred_xyz_list = []
    pred_q_list = []
    gt_xyz_list = []
    gt_q_list = []

    with torch.no_grad():
        for imgs, xyz_gt, q_gt, scene_ids in loader:
            imgs = imgs.to(device, non_blocking=True)
            scene_ids = scene_ids.to(device, non_blocking=True)

            xyz_pred, q_pred = model(imgs, scene_id=scene_ids)

            for i in range(imgs.shape[0]):
                pred_xyz_list.append(xyz_pred[i].cpu().numpy())
                pred_q_list.append(q_pred[i].cpu().numpy())
                gt_xyz_list.append(xyz_gt[i].numpy())
                gt_q_list.append(q_gt[i].numpy())

    metrics = compute_metrics(pred_xyz_list, pred_q_list, gt_xyz_list, gt_q_list)

    print(f"\nResults:")
    print(f"  Median position error: {metrics['median_pos']:.4f} m")
    print(f"  Mean position error:   {metrics['mean_pos']:.4f} m")
    print(f"  Median rotation error: {metrics['median_rot']:.2f} deg")
    print(f"  Mean rotation error:   {metrics['mean_rot']:.2f} deg")

    # Save predictions to CSV
    out_dir = os.path.dirname(args.checkpoint)
    out_csv = os.path.join(out_dir, "eval_predictions.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "pred_tx", "pred_ty", "pred_tz",
                          "pred_qw", "pred_qx", "pred_qy", "pred_qz",
                          "gt_tx", "gt_ty", "gt_tz",
                          "gt_qw", "gt_qx", "gt_qy", "gt_qz",
                          "pos_err", "rot_err_deg"])
        for i in range(len(pred_xyz_list)):
            p = pred_xyz_list[i]
            g = gt_xyz_list[i]
            pq = pred_q_list[i]
            gq = gt_q_list[i]
            writer.writerow([
                i,
                p[0], p[1], p[2], pq[0], pq[1], pq[2], pq[3],
                g[0], g[1], g[2], gq[0], gq[1], gq[2], gq[3],
                metrics["pos_errors"][i], metrics["rot_errors"][i],
            ])
    print(f"Predictions saved to {out_csv}")

    # Visualizations
    if args.plot:
        gt_xyz = np.array(gt_xyz_list)
        pred_xyz = np.array(pred_xyz_list)

        traj_path = os.path.join(out_dir, "trajectory_3d.png")
        plot_trajectory_3d(gt_xyz, pred_xyz, title=f"{cfg['model']['name']} Trajectory",
                           save_path=traj_path)
        print(f"Trajectory plot saved to {traj_path}")

        pos_cdf_path = os.path.join(out_dir, "position_error_cdf.png")
        plot_cumulative_error(metrics["pos_errors"], cfg["model"]["name"],
                              xlabel="Position Error (m)",
                              title="Cumulative Position Error",
                              save_path=pos_cdf_path)

        rot_cdf_path = os.path.join(out_dir, "rotation_error_cdf.png")
        plot_cumulative_error(metrics["rot_errors"], cfg["model"]["name"],
                              xlabel="Rotation Error (deg)",
                              title="Cumulative Rotation Error",
                              save_path=rot_cdf_path)
        print(f"CDF plots saved to {out_dir}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate pose regression model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--data_type", type=str, default=None,
                        choices=["colmap", "7scenes", "7scenes_single"])
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no_plot", action="store_false", dest="plot")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
