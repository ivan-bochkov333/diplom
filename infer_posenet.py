#!/usr/bin/env python3
import os
import csv
import math
import argparse
from typing import Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset


# ===========================
# Модель (та же, что в train_posenet.py)
# ===========================

class MiniPoseNet(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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
        q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-8)
        return xyz, q


# ===========================
# Dataset для инференса
# ===========================

class ImageOnlyDataset(Dataset):
    def __init__(self, frames, images_dir, transform):
        self.frames = frames
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        img_path = os.path.join(self.images_dir, f"{frame}.jpg")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, frame


def load_gt_poses(gt_csv_path: str) -> Dict[str, dict]:
    """
    Читаем GT poses.csv → dict frame -> {xyz, q}
    """
    gt = {}
    with open(gt_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = row["frame"]
            tx = float(row["tx"])
            ty = float(row["ty"])
            tz = float(row["tz"])
            qw = float(row["qw"])
            qx = float(row["qx"])
            qy = float(row["qy"])
            qz = float(row["qz"])

            xyz = np.array([tx, ty, tz], dtype=np.float32)
            q = np.array([qw, qx, qy, qz], dtype=np.float32)
            q = q / (np.linalg.norm(q) + 1e-8)
            gt[frame] = {"xyz": xyz, "q": q}
    return gt


def orientation_angle_deg(q_pred: np.ndarray, q_gt: np.ndarray) -> float:
    """
    Угловая ошибка между двумя кватернионами в градусах.
    q_pred, q_gt: shape (4,)
    """
    q_pred = q_pred / (np.linalg.norm(q_pred) + 1e-8)
    q_gt = q_gt / (np.linalg.norm(q_gt) + 1e-8)
    dot = float(np.clip(np.abs(np.dot(q_pred, q_gt)), -1.0, 1.0))
    angle = 2 * math.acos(dot)  # [0, pi]
    return angle * 180.0 / math.pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="my_test_1_dataset",
                        help="Папка с images/ и (опционально) poses.csv")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Путь к posenet.pth (если не задан, берем data_root/posenet.pth)")
    parser.add_argument("--out_csv", type=str, default=None,
                        help="Куда сохранить предсказания (по умолчанию data_root/pred_poses.csv)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Если указан флаг, и есть poses.csv, то посчитать ошибки относительно GT")
    args = parser.parse_args()

    data_root = args.data_root
    images_dir = os.path.join(data_root, "images")
    gt_csv_path = os.path.join(data_root, "poses.csv")
    ckpt_path = args.checkpoint or os.path.join(data_root, "posenet.pth")
    out_csv = args.out_csv or os.path.join(data_root, "pred_poses.csv")

    device = torch.device("cpu")

    # 1. Список кадров
    frames = []
    for fname in sorted(os.listdir(images_dir)):
        if fname.lower().endswith(".jpg"):
            frame = os.path.splitext(fname)[0]
            frames.append(frame)

    if not frames:
        print(f"Не найдено .jpg картинок в {images_dir}")
        return

    print(f"Нашёл {len(frames)} кадров для инференса.")

    # 2. Трансформации такие же, как при обучении
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    ds = ImageOnlyDataset(frames, images_dir, transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    # 3. Модель
    model = MiniPoseNet(freeze_backbone=True).to(device)
    if not os.path.exists(ckpt_path):
        print(f"Не найден чекпоинт {ckpt_path}")
        return

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 4. Инференс и запись в CSV
    print(f"Делаю инференс, сохраню в {out_csv}")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "tx", "ty", "tz", "qw", "qx", "qy", "qz"])

        all_preds = {}

        with torch.no_grad():
            for imgs, batch_frames in loader:
                imgs = imgs.to(device)
                xyz_pred, q_pred = model(imgs)

                xyz_pred = xyz_pred.cpu().numpy()
                q_pred = q_pred.cpu().numpy()

                for frame, xyz, q in zip(batch_frames, xyz_pred, q_pred):
                    writer.writerow([frame, xyz[0], xyz[1], xyz[2], q[0], q[1], q[2], q[3]])
                    all_preds[frame] = {"xyz": xyz, "q": q}

    print("Инференс завершён.")

    # 5. Оценка, если есть GT и запрошен --evaluate
    if args.evaluate and os.path.exists(gt_csv_path):
        print("Читаю GT из poses.csv и считаю ошибки...")
        gt = load_gt_poses(gt_csv_path)

        n = 0
        pos_err_sum = 0.0
        ang_err_sum = 0.0

        for frame, pred in all_preds.items():
            if frame not in gt:
                continue
            xyz_p = pred["xyz"]
            q_p = pred["q"]
            xyz_g = gt[frame]["xyz"]
            q_g = gt[frame]["q"]

            pos_err = float(np.linalg.norm(xyz_p - xyz_g))
            ang_err = orientation_angle_deg(q_p, q_g)

            pos_err_sum += pos_err
            ang_err_sum += ang_err
            n += 1

        if n > 0:
            print(f"Число сравнимых кадров: {n}")
            print(f"Средняя позиционная ошибка: {pos_err_sum / n:.4f} (в единицах COLMAP)")
            print(f"Средняя угловая ошибка:      {ang_err_sum / n:.2f} градусов")
        else:
            print("Нет пересечения кадров между предсказаниями и GT.")
    elif args.evaluate:
        print(f"GT poses.csv не найден в {gt_csv_path}, пропускаю оценку.")


if __name__ == "__main__":
    main()
