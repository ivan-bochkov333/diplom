#!/usr/bin/env python3
"""
Рантайм-инференс: один кадр → позиция и ориентация камеры на обученной сцене.

Использование:
  # из файла
  python run_inference_single.py --checkpoint outputs/posenet/best.pth --image path/to/frame.jpg

  # из кода (импорт)
  from run_inference_single import load_pose_model, predict_pose
  model, transform, device = load_pose_model("outputs/posenet/best.pth")
  xyz, quat = predict_pose(model, transform, device, "frame.jpg")  # или PIL Image / numpy
"""

import argparse
import os

import torch
from PIL import Image
import numpy as np

from src.models import build_model
from src.datasets.colmap_dataset import get_transforms


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pose_model(checkpoint_path: str):
    """Загружает модель и препроцессинг. Вызвать один раз при старте."""
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    image_size = cfg["data"].get("image_size", 224)
    transform = get_transforms("val", image_size)

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, transform, device


def predict_pose(model, transform, device, image_input, scene_id: int = 0):
    """
    Предсказание позы по одному кадру.

    image_input: путь к файлу (str), PIL.Image или numpy array (H,W,3) RGB.
    Возвращает: (xyz, quat) — numpy arrays формы (3,) и (4,) — позиция и кватернион (qw,qx,qy,qz).
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
    else:
        raise TypeError("image_input: path (str), PIL.Image or numpy array (H,W,3)")

    x = transform(img).unsqueeze(0).to(device)
    sid = torch.full((1,), scene_id, dtype=torch.long, device=device)

    with torch.no_grad():
        xyz, q = model(x, scene_id=sid)

    xyz = xyz[0].cpu().numpy()
    q = q[0].cpu().numpy()
    q = q / (np.linalg.norm(q) + 1e-8)
    return xyz, q


def main():
    parser = argparse.ArgumentParser(description="Single-frame pose inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--scene_id", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"File not found: {args.image}")
        return

    model, transform, device = load_pose_model(args.checkpoint)
    xyz, quat = predict_pose(model, transform, device, args.image, scene_id=args.scene_id)

    print("Position (tx, ty, tz):", xyz.tolist())
    print("Orientation quaternion (qw, qx, qy, qz):", quat.tolist())
    print("\nДля использования в рантайме импортируйте load_pose_model и predict_pose из этого файла.")


if __name__ == "__main__":
    main()
