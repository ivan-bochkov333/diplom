#!/usr/bin/env python3
import os
import csv
import math
from typing import List, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# ===========================
# Dataset
# ===========================

class PoseSample:
    def __init__(self, frame: str, xyz: np.ndarray, q: np.ndarray):
        self.frame = frame
        self.xyz = xyz.astype(np.float32)
        self.q = q.astype(np.float32)


class PoseDataset(Dataset):
    def __init__(self, samples: List[PoseSample], images_dir: str, transform):
        self.samples = samples
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = os.path.join(self.images_dir, f"{s.frame}.jpg")
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        xyz = torch.from_numpy(s.xyz)
        q = torch.from_numpy(s.q)
        return img, xyz, q


def load_samples(csv_path: str, images_dir: str) -> List[PoseSample]:
    samples: List[PoseSample] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = row["frame"]
            img_path = os.path.join(images_dir, f"{frame}.jpg")
            if not os.path.exists(img_path):
                # пропускаем позы без картинки
                continue

            tx = float(row["tx"])
            ty = float(row["ty"])
            tz = float(row["tz"])
            qw = float(row["qw"])
            qx = float(row["qx"])
            qy = float(row["qy"])
            qz = float(row["qz"])

            xyz = np.array([tx, ty, tz], dtype=np.float32)
            q = np.array([qw, qx, qy, qz], dtype=np.float32)
            # на всякий случай нормируем кватернион
            q = q / (np.linalg.norm(q) + 1e-8)

            samples.append(PoseSample(frame, xyz, q))

    # сортируем по номеру кадра
    samples.sort(key=lambda s: int(s.frame))
    return samples


# ===========================
# Модель
# ===========================

class MiniPoseNet(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()

        # ResNet18 backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # выкидываем последний FC, оставляем фичи [B, 512, 1, 1]
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        feat_dim = 512

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )

        self.fc_xyz = nn.Linear(256, 3)
        self.fc_q = nn.Linear(256, 4)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.fc(feat)

        xyz = self.fc_xyz(feat)
        q = self.fc_q(feat)
        # нормируем кватернион
        q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-8)
        return xyz, q


# ===========================
# Лоссы
# ===========================

def position_loss(x_pred, x_gt):
    # MSE по позиции
    return nn.functional.mse_loss(x_pred, x_gt)


def orientation_loss(q_pred, q_gt):
    """
    Кватернионы должны быть нормализованы.
    Лосс = 1 - |<q_pred, q_gt>|, эквивалентен ошибке по углу.
    """
    # нормируем ещё раз на всякий случай
    q_pred = q_pred / (q_pred.norm(dim=1, keepdim=True) + 1e-8)
    q_gt = q_gt / (q_gt.norm(dim=1, keepdim=True) + 1e-8)

    dot = torch.sum(q_pred * q_gt, dim=1).abs()
    loss = 1.0 - dot
    return loss.mean()


def orientation_angle_deg(q_pred, q_gt):
    """
    Средняя угловая ошибка в градусах для батча.
    """
    q_pred = q_pred / (q_pred.norm(dim=1, keepdim=True) + 1e-8)
    q_gt = q_gt / (q_gt.norm(dim=1, keepdim=True) + 1e-8)
    dot = torch.sum(q_pred * q_gt, dim=1)
    dot = torch.clamp(dot, -1.0, 1.0)
    angle = 2 * torch.acos(dot.abs())  # [0, pi]
    return angle.mean().item() * 180.0 / math.pi


# ===========================
# Тренировка
# ===========================

def train(
    data_root: str = "my_test_1_dataset",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    beta_ori: float = 1.0,
):

    device = torch.device("cpu")  # при желании можно поменять на cuda

    images_dir = os.path.join(data_root, "images")
    csv_path = os.path.join(data_root, "poses.csv")

    print(f"Загружаю датасет из {csv_path}")
    all_samples = load_samples(csv_path, images_dir)
    print(f"Всего валидных сэмплов: {len(all_samples)}")

    # train/val split по кадрам (80/20)
    n_total = len(all_samples)
    n_train = int(0.8 * n_total)
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:]

    print(f"Train: {len(train_samples)}  |  Val: {len(val_samples)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # можно добавить нормировку под ImageNet, но это не критично
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = PoseDataset(train_samples, images_dir, transform)
    val_ds = PoseDataset(val_samples, images_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MiniPoseNet(freeze_backbone=True).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    print("Начинаю обучение...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_pos_sum = 0.0
        train_ori_sum = 0.0
        n_train_batches = 0

        for imgs, xyz_gt, q_gt in train_loader:
            imgs = imgs.to(device)
            xyz_gt = xyz_gt.to(device)
            q_gt = q_gt.to(device)

            optimizer.zero_grad()
            xyz_pred, q_pred = model(imgs)

            lp = position_loss(xyz_pred, xyz_gt)
            lo = orientation_loss(q_pred, q_gt)
            loss = lp + beta_ori * lo

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_pos_sum += lp.item()
            train_ori_sum += lo.item()
            n_train_batches += 1

        # валидация
        model.eval()
        val_loss_sum = 0.0
        val_pos_sum = 0.0
        val_ori_sum = 0.0
        val_angle_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for imgs, xyz_gt, q_gt in val_loader:
                imgs = imgs.to(device)
                xyz_gt = xyz_gt.to(device)
                q_gt = q_gt.to(device)

                xyz_pred, q_pred = model(imgs)

                lp = position_loss(xyz_pred, xyz_gt)
                lo = orientation_loss(q_pred, q_gt)
                loss = lp + beta_ori * lo

                val_loss_sum += loss.item()
                val_pos_sum += lp.item()
                val_ori_sum += lo.item()
                val_angle_sum += orientation_angle_deg(q_pred, q_gt)
                n_val_batches += 1

        train_loss = train_loss_sum / max(n_train_batches, 1)
        train_pos = train_pos_sum / max(n_train_batches, 1)
        train_ori = train_ori_sum / max(n_train_batches, 1)

        val_loss = val_loss_sum / max(n_val_batches, 1)
        val_pos = val_pos_sum / max(n_val_batches, 1)
        val_ori = val_ori_sum / max(n_val_batches, 1)
        val_angle = val_angle_sum / max(n_val_batches, 1)

        print(
            f"Epoch {epoch:03d} | "
            f"train: loss={train_loss:.4f}, pos={train_pos:.4f}, ori={train_ori:.4f} | "
            f"val: loss={val_loss:.4f}, pos={val_pos:.4f}, ori={val_ori:.4f}, angle={val_angle:.2f}°"
        )

    # сохраняем веса
    out_path = os.path.join(data_root, "posenet.pth")
    torch.save(model.state_dict(), out_path)
    print(f"Модель сохранена в {out_path}")


if __name__ == "__main__":
    # при желании можно поменять параметры
    train(
        data_root="my_test_1_dataset",
        epochs=30,
        batch_size=32,
        lr=1e-4,
        beta_ori=1.0,
    )
