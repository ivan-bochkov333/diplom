#!/usr/bin/env python3
import sys
import os
import numpy as np


def read_images_txt(path):
    """
    Парсим images.txt от COLMAP.
    Берём только ЗАГОЛОВКИ изображений (первая строка из пары).
    Возвращаем словарь: name -> {q, t}
    """
    images = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            elems = line.split()
            # Заголовочная строка: IMAGE_ID qw qx qy qz tx ty tz CAMERA_ID NAME
            if len(elems) < 10:
                continue

            # Вторая строка (список 2D-3D точек) начинается с x (float),
            # поэтому int(elems[0]) там упадёт -> мы её пропустим.
            try:
                _image_id = int(elems[0])
            except ValueError:
                # Это не строка заголовка, пропускаем
                continue

            qw, qx, qy, qz = map(float, elems[1:5])
            tx, ty, tz = map(float, elems[5:8])
            name = elems[9]

            images[name] = {
                "q": np.array([qw, qx, qy, qz], dtype=float),
                "t": np.array([tx, ty, tz], dtype=float),
            }

    return images


def q_to_R(q):
    """
    Кватернион (qw, qx, qy, qz) -> матрица поворота 3x3.
    Формула как в COLMAP (world -> camera).
    """
    qw, qx, qy, qz = q
    R = np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=float,
    )
    return R


def main():
    if len(sys.argv) < 3:
        print("Использование: extract_colmap_poses.py images.txt output.csv")
        sys.exit(1)

    images_txt = sys.argv[1]
    out_csv = sys.argv[2]

    images = read_images_txt(images_txt)

    with open(out_csv, "w") as f:
        f.write("frame,tx,ty,tz,qw,qx,qy,qz\n")

        for name in sorted(images.keys()):
            data = images[name]
            q = data["q"]
            t = data["t"]

            # COLMAP: X_cam = R * X_world + t
            # Центр камеры в мировых координатах:
            #   C = - R^T * t
            R = q_to_R(q)
            C = -R.T @ t

            frame = os.path.splitext(name)[0]

            f.write(f"{frame},{C[0]},{C[1]},{C[2]},{q[0]},{q[1]},{q[2]},{q[3]}\n")

    print(f"Saved poses to {out_csv}")


if __name__ == "__main__":
    main()
