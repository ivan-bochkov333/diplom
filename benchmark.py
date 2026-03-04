#!/usr/bin/env python3
"""
Benchmark script: evaluate all trained models and produce a comparison table.

Usage:
    python benchmark.py --outputs_dir outputs --data_root my_test_1_dataset
    python benchmark.py --outputs_dir outputs --data_root new_video_test --measure_speed --update_readme README.md
"""

import os
import argparse
import json
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import build_model
from src.datasets.colmap_dataset import ColmapDataset, get_transforms
from src.datasets.seven_scenes import SevenScenesDataset
from src.utils.metrics import compute_metrics
from src.utils.visualization import plot_comparison_curves


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_checkpoints(outputs_dir):
    """Find all best.pth checkpoints in output subdirectories."""
    checkpoints = {}
    if not os.path.isdir(outputs_dir):
        return checkpoints

    for model_name in sorted(os.listdir(outputs_dir)):
        ckpt = os.path.join(outputs_dir, model_name, "best.pth")
        if os.path.isfile(ckpt):
            checkpoints[model_name] = ckpt
    return checkpoints


def build_eval_dataset(args):
    """Build evaluation dataset based on CLI args."""
    if args.data_type == "colmap":
        ds = ColmapDataset(
            data_root=args.data_root,
            image_size=224, train_split=0.0,
        )
        return ds.val_dataset
    elif args.data_type in ("7scenes_single",):
        seven = SevenScenesDataset(
            root=args.seven_scenes_root, scenes=[args.scene], image_size=224,
        )
        _, test_ds = seven.get_scene_datasets(args.scene)
        return test_ds
    else:
        raise ValueError(f"Unknown data_type: {args.data_type}")


def measure_inference_speed(model, device, image_size=224, num_warmup=10, num_runs=100):
    """Return mean ms per frame and FPS (batch_size=1)."""
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    scene_id = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(num_warmup):
            model(x, scene_id=scene_id)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_runs):
            model(x, scene_id=scene_id)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    elapsed = t1 - t0
    ms_per_frame = 1000.0 * elapsed / num_runs
    fps = num_runs / elapsed
    return ms_per_frame, fps


def make_markdown_table(results):
    """Generate markdown table from results dict (with optional speed keys)."""
    rows = []
    rows.append("| Модель | Med. позиция (м) | Mean позиция (м) | Med. поворот (°) | Mean поворот (°) | мс/кадр | FPS |")
    rows.append("|--------|------------------|------------------|-------------------|-------------------|---------|-----|")
    for name, m in sorted(results.items(), key=lambda x: x[1].get("median_pos", float("inf"))):
        med_pos = m.get("median_pos")
        mean_pos = m.get("mean_pos")
        med_rot = m.get("median_rot")
        mean_rot = m.get("mean_rot")
        ms = m.get("ms_per_frame")
        fps = m.get("fps")
        med_pos_s = f"{med_pos:.4f}" if med_pos is not None else "—"
        mean_pos_s = f"{mean_pos:.4f}" if mean_pos is not None else "—"
        med_rot_s = f"{med_rot:.2f}" if med_rot is not None else "—"
        mean_rot_s = f"{mean_rot:.2f}" if mean_rot is not None else "—"
        ms_s = f"{ms:.1f}" if ms is not None else "—"
        fps_s = f"{fps:.1f}" if fps is not None else "—"
        rows.append(f"| {name} | {med_pos_s} | {mean_pos_s} | {med_rot_s} | {mean_rot_s} | {ms_s} | {fps_s} |")
    return "\n".join(rows)


def update_readme_with_table(readme_path, table_md, marker_start="<!-- BENCHMARK_TABLE -->", marker_end="<!-- /BENCHMARK_TABLE -->"):
    """Replace content between markers in README with the new table."""
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = re.compile(
        re.escape(marker_start) + r".*?" + re.escape(marker_end),
        re.DOTALL,
    )
    replacement = marker_start + "\n\n" + table_md + "\n\n" + marker_end
    if pattern.search(content):
        new_content = pattern.sub(replacement, content)
    else:
        new_content = content.rstrip() + "\n\n## Сравнение архитектур (бенчмарк)\n\n"
        new_content += "Оценка на одном и том же тестовом наборе (точность и скорость инференса).\n\n"
        new_content += marker_start + "\n\n" + table_md + "\n\n" + marker_end + "\n"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def main():
    parser = argparse.ArgumentParser(description="Benchmark all models")
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--data_root", type=str, default="my_test_1_dataset")
    parser.add_argument("--data_type", type=str, default="colmap",
                        choices=["colmap", "7scenes_single"])
    parser.add_argument("--scene", type=str, default="chess")
    parser.add_argument("--seven_scenes_root", type=str, default="data/7scenes")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--measure_speed", action="store_true", help="Measure inference time (ms/frame, FPS)")
    parser.add_argument("--num_speed_runs", type=int, default=100, help="Number of forward passes for speed measurement")
    parser.add_argument("--update_readme", type=str, default=None, metavar="PATH", help="Update README.md with markdown table (path to README)")
    args = parser.parse_args()

    device = get_device()
    checkpoints = find_checkpoints(args.outputs_dir)

    if not checkpoints:
        print(f"No checkpoints found in {args.outputs_dir}/*/best.pth")
        return

    print(f"Found {len(checkpoints)} models: {list(checkpoints.keys())}")
    print(f"Evaluating on: {args.data_type} / {args.data_root or args.scene}")
    print()

    eval_dataset = build_eval_dataset(args)
    loader = DataLoader(eval_dataset, batch_size=32, shuffle=False,
                        num_workers=4, pin_memory=True)

    results = {}
    all_pos_errors = {}
    all_rot_errors = {}

    for model_name, ckpt_path in checkpoints.items():
        print(f"Evaluating {model_name}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        cfg = checkpoint["config"]

        model = build_model(cfg).to(device)
        model.load_state_dict(checkpoint["model"], strict=False)
        model.eval()

        pred_xyz_list, pred_q_list = [], []
        gt_xyz_list, gt_q_list = [], []

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

        results[model_name] = {
            "median_pos": metrics["median_pos"],
            "mean_pos": metrics["mean_pos"],
            "median_rot": metrics["median_rot"],
            "mean_rot": metrics["mean_rot"],
        }
        all_pos_errors[model_name] = metrics["pos_errors"]
        all_rot_errors[model_name] = metrics["rot_errors"]

        if args.measure_speed:
            image_size = cfg["data"].get("image_size", 224)
            ms_per_frame, fps = measure_inference_speed(
                model, device, image_size=image_size, num_runs=args.num_speed_runs
            )
            results[model_name]["ms_per_frame"] = ms_per_frame
            results[model_name]["fps"] = fps
            print(f"  Speed: {ms_per_frame:.1f} ms/frame, {fps:.1f} FPS")

        print(f"  Pos: {metrics['median_pos']:.4f}m (median), "
              f"{metrics['mean_pos']:.4f}m (mean)")
        print(f"  Rot: {metrics['median_rot']:.2f}° (median), "
              f"{metrics['mean_rot']:.2f}° (mean)")
        print()

    # Print comparison table
    print("=" * 80)
    header = f"{'Model':<20} {'Med Pos (m)':<14} {'Mean Pos (m)':<14} {'Med Rot (°)':<14} {'Mean Rot (°)':<14}"
    if args.measure_speed and any("ms_per_frame" in m for m in results.values()):
        header += f" {'ms/frame':<12} {'FPS':<8}"
    print(header)
    print("-" * 80)
    for name, m in sorted(results.items(), key=lambda x: x[1]["median_pos"]):
        line = f"{name:<20} {m['median_pos']:<14.4f} {m['mean_pos']:<14.4f} {m['median_rot']:<14.2f} {m['mean_rot']:<14.2f}"
        if args.measure_speed and m.get("ms_per_frame") is not None:
            line += f" {m['ms_per_frame']:<12.1f} {m['fps']:<8.1f}"
        print(line)
    print("=" * 80)

    # Save results
    results_path = os.path.join(args.outputs_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if args.update_readme:
        table_md = make_markdown_table(results)
        update_readme_with_table(args.update_readme, table_md)
        print(f"README updated: {args.update_readme}")

    # Plots
    if args.plot and len(all_pos_errors) > 1:
        plot_comparison_curves(
            all_pos_errors,
            xlabel="Position Error (m)",
            title="Position Error CDF — Model Comparison",
            save_path=os.path.join(args.outputs_dir, "benchmark_pos_cdf.png"),
        )
        plot_comparison_curves(
            all_rot_errors,
            xlabel="Rotation Error (deg)",
            title="Rotation Error CDF — Model Comparison",
            save_path=os.path.join(args.outputs_dir, "benchmark_rot_cdf.png"),
        )
        print(f"Comparison plots saved to {args.outputs_dir}")


if __name__ == "__main__":
    main()
