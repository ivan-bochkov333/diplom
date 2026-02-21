#!/usr/bin/env python3
"""
Unified training script for all pose regression models.
Config-driven: reads base.yaml + model-specific overrides.

Usage:
    python train.py --config configs/posenet.yaml
    python train.py --config configs/atloc.yaml --data.root my_test_1_dataset
    python train.py --config configs/ms_transformer.yaml --finetune --checkpoint outputs/ms_transformer/best.pth
"""

import os
import sys
import copy
import math
import time
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from src.models import build_model
from src.losses import PoseLoss
from src.datasets.colmap_dataset import ColmapDataset, get_transforms
from src.datasets.seven_scenes import SevenScenesDataset
from src.utils.metrics import batch_rotation_error_deg


def load_config(config_path: str, overrides: list = None):
    """Load base config, merge with model config, apply CLI overrides."""
    base_path = os.path.join(os.path.dirname(config_path), "base.yaml")
    cfg = {}
    if os.path.exists(base_path):
        with open(base_path) as f:
            cfg = yaml.safe_load(f)

    with open(config_path) as f:
        model_cfg = yaml.safe_load(f)

    cfg = _deep_merge(cfg, model_cfg)

    if overrides:
        for ov in overrides:
            if "=" in ov:
                key, val = ov.split("=", 1)
            else:
                key, val = ov.lstrip("-").split(".", 1) if "." in ov else (ov, "true")
                if "=" in val:
                    parts = val.split("=", 1)
                    key = key + "." + parts[0]
                    val = parts[1]
                else:
                    continue

            keys = key.lstrip("-").split(".")
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = _parse_value(val)

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _parse_value(val: str):
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.lower() == "null" or val.lower() == "none":
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(cfg):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    batch_size = train_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 4)

    if data_cfg["type"] == "colmap":
        ds = ColmapDataset(
            data_root=data_cfg["root"],
            normalize=data_cfg.get("normalize", False),
            image_size=data_cfg.get("image_size", 224),
            train_split=data_cfg.get("train_split", 0.8),
        )
        train_loader = DataLoader(
            ds.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            ds.val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True,
        )
        return train_loader, val_loader

    elif data_cfg["type"] == "7scenes":
        root = data_cfg["seven_scenes_root"]
        scenes = data_cfg.get("scenes", None)
        seven = SevenScenesDataset(
            root=root, scenes=scenes,
            image_size=data_cfg.get("image_size", 224),
            normalize=data_cfg.get("normalize", False),
        )
        train_ds, val_ds = seven.get_multiscene_datasets()
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True,
        )
        return train_loader, val_loader

    elif data_cfg["type"] == "7scenes_single":
        root = data_cfg["seven_scenes_root"]
        scene_name = data_cfg["scene"]
        seven = SevenScenesDataset(
            root=root, scenes=[scene_name],
            image_size=data_cfg.get("image_size", 224),
            normalize=data_cfg.get("normalize", False),
        )
        train_ds, val_ds = seven.get_scene_datasets(scene_name)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True,
        )
        return train_loader, val_loader

    else:
        raise ValueError(f"Unknown data type: {data_cfg['type']}")


def build_scheduler(optimizer, cfg, steps_per_epoch):
    sched_name = cfg["training"].get("scheduler", "cosine")
    epochs = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"].get("warmup_epochs", 5)

    if sched_name == "cosine":
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs * steps_per_epoch,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(epochs - warmup_epochs) * steps_per_epoch,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs * steps_per_epoch],
        )
        return scheduler
    elif sched_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        return None


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, val_loss) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train(cfg, finetune_checkpoint=None):
    force_cpu = cfg.get("training", {}).get("force_cpu", False)
    device = get_device(force_cpu=force_cpu)
    print(f"Device: {device}")

    out_dir = os.path.join(
        cfg["output"]["dir"],
        cfg["model"]["name"],
    )
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Data
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = build_model(cfg).to(device)

    if finetune_checkpoint:
        print(f"Loading checkpoint for fine-tuning: {finetune_checkpoint}")
        state = torch.load(finetune_checkpoint, map_location=device)
        model.load_state_dict(state["model"], strict=False)
        if hasattr(model, "prepare_finetune"):
            ft_cfg = cfg.get("finetune", {})
            model.prepare_finetune(new_num_scenes=ft_cfg.get("num_scenes", 1))
            model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg['model']['name']}")
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Loss
    learn_weights = cfg["training"].get("learn_loss_weights", True)
    criterion = PoseLoss(learn_weights=learn_weights).to(device)

    # Optimizer — include loss parameters if learnable
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if learn_weights:
        params += list(criterion.parameters())

    train_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        params, lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )

    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    use_amp = train_cfg.get("amp", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    early_stop = EarlyStopping(
        patience=train_cfg.get("patience", 20),
        min_delta=train_cfg.get("min_delta", 1e-4),
    )

    # TensorBoard
    writer = None
    if cfg["output"].get("tensorboard", True):
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb_logs"))
        except ImportError:
            print("TensorBoard not available, skipping logging.")

    best_val_loss = float("inf")
    epochs = train_cfg["epochs"]

    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ---- TRAIN ----
        model.train()
        criterion.train()
        train_loss_sum = 0.0
        train_pos_sum = 0.0
        train_ori_sum = 0.0
        n_train = 0

        for imgs, xyz_gt, q_gt, scene_ids in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            xyz_gt = xyz_gt.to(device, non_blocking=True)
            q_gt = q_gt.to(device, non_blocking=True)
            scene_ids = scene_ids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast(enabled=True):
                    xyz_pred, q_pred = model(imgs, scene_id=scene_ids)
                    loss, lp, lo = criterion(xyz_pred, xyz_gt, q_pred, q_gt)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                xyz_pred, q_pred = model(imgs.float(), scene_id=scene_ids)
                loss, lp, lo = criterion(xyz_pred, xyz_gt, q_pred, q_gt)
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            train_loss_sum += loss.item()
            train_pos_sum += lp.item()
            train_ori_sum += lo.item()
            n_train += 1

        # ---- VALIDATION ----
        model.eval()
        criterion.eval()
        val_loss_sum = 0.0
        val_pos_sum = 0.0
        val_ori_sum = 0.0
        val_angle_sum = 0.0
        n_val = 0

        with torch.no_grad():
            for imgs, xyz_gt, q_gt, scene_ids in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                xyz_gt = xyz_gt.to(device, non_blocking=True)
                q_gt = q_gt.to(device, non_blocking=True)
                scene_ids = scene_ids.to(device, non_blocking=True)

                xyz_pred, q_pred = model(imgs.float(), scene_id=scene_ids)
                loss, lp, lo = criterion(xyz_pred, xyz_gt, q_pred, q_gt)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                angle_err = batch_rotation_error_deg(q_pred.float(), q_gt.float()).mean().item()

                val_loss_sum += loss.item()
                val_pos_sum += lp.item()
                val_ori_sum += lo.item()
                val_angle_sum += angle_err
                n_val += 1

        # ---- Metrics ----
        train_loss = train_loss_sum / max(n_train, 1)
        train_pos = train_pos_sum / max(n_train, 1)
        train_ori = train_ori_sum / max(n_train, 1)
        val_loss = val_loss_sum / max(n_val, 1)
        val_pos = val_pos_sum / max(n_val, 1)
        val_ori = val_ori_sum / max(n_val, 1)
        val_angle = val_angle_sum / max(n_val, 1)
        elapsed = time.time() - t0

        weights = criterion.get_effective_weights()
        lr_current = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{epochs} ({elapsed:.1f}s) | "
            f"lr={lr_current:.2e} | "
            f"train: loss={train_loss:.4f} pos={train_pos:.4f} ori={train_ori:.4f} | "
            f"val: loss={val_loss:.4f} pos={val_pos:.4f} ori={val_ori:.4f} "
            f"angle={val_angle:.2f}deg | "
            f"w_pos={weights['w_pos']:.2f} w_ori={weights['w_ori']:.2f}"
        )

        if writer:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/pos_loss", train_pos, epoch)
            writer.add_scalar("train/ori_loss", train_ori, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/pos_loss", val_pos, epoch)
            writer.add_scalar("val/ori_loss", val_ori, epoch)
            writer.add_scalar("val/angle_deg", val_angle, epoch)
            writer.add_scalar("lr", lr_current, epoch)
            writer.add_scalar("loss_weights/w_pos", weights["w_pos"], epoch)
            writer.add_scalar("loss_weights/w_ori", weights["w_ori"], epoch)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if cfg["output"].get("save_best", True):
                save_path = os.path.join(out_dir, "best.pth")
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "criterion": criterion.state_dict(),
                    "val_loss": val_loss,
                    "val_angle": val_angle,
                    "config": cfg,
                }, save_path)

        if early_stop.step(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    # Save final
    save_path = os.path.join(out_dir, "last.pth")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "criterion": criterion.state_dict(),
        "val_loss": val_loss,
        "config": cfg,
    }, save_path)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in {out_dir}")

    if writer:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train pose regression model")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--finetune", type=str, default=None,
                        help="Path to checkpoint for fine-tuning")
    args, unknown = parser.parse_known_args()

    cfg = load_config(args.config, unknown)
    train(cfg, finetune_checkpoint=args.finetune)


if __name__ == "__main__":
    main()
