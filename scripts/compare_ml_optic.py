#!/usr/bin/env python3
"""
Сравнение предсказаний ML (poses из infer / COLMAP-координаты) с эталоном оптического трекинга (AL_Optic.csv).

Позы в разных системах координат и с разным масштабом COLMAP → перед метриками выполняется
similarity transform (Umeyama): p_opt ≈ s * R @ p_ml + t по всем спаренным точкам.

Базовый минимум метрик (как у научника):
  1) максимальное отклонение по позиции (м);
  2) среднее отклонение по позиции (м);
  3) максимальное по сферическому расстоянию ориентации (°, SO(3));
  4) среднее по сферическому расстоянию ориентации (°, SO(3)).

Дополнительно в JSON: медианы по позиции и по углу; при --kabsch_orientation_calibration — углы после R_kabsch;
всегда (кроме --no_relative_orientation) — метрики относительного поворота между соседними кадрами.

Калибровка «камера (COLMAP) ↔ датчик оптики»:
  - В poses.csv / pred: кватернион как в COLMAP, R_wc: X_cam = R @ X_world.
  - Оси датчика на креплении ≠ оси камеры OpenCV — задаётся R_sensor_to_camera (вектор из СК датчика в СК камеры).
  - Ориентация датчика в мире (colmap): R_wc^T @ R_sensor_to_camera; после Umeyama: R_align @ R_wc^T @ R_sensor_to_camera.
  - Если в AL_Optic кватернион в том же смысле, что COLMAP (world→тело), базис тела в мире = R(q)^T.

Использование:
  python scripts/compare_ml_optic.py --optic_csv AL_Optic.csv --pred_csv test_v2_scene/pred_poses.csv --frames_fps 5
  python scripts/compare_ml_optic.py ... --rig_preset diagram_xy_swap
  python scripts/compare_ml_optic.py ... --sensor_to_camera_rowmajor 0 1 0 1 0 0 0 0 1
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

# --- Quaternion / rotation (совместимо с extract_colmap_poses: qw,qx,qy,qz -> R world->camera) ---


def q_to_R_wxyz(q):
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


def R_to_quat_wxyz(R):
    """Rotation matrix -> unit quaternion (qw, qx, qy, qz)."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)


def quat_angle_deg(q1, q2):
    """Угол между двумя кватернионами (wxyz), градусы — геодезия на SO(3)."""
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(0.0, d))
    return 2.0 * np.degrees(np.arccos(d))


def project_to_so3(M: np.ndarray) -> np.ndarray:
    """Ближайшая правильная ортогональная матрица с det=+1 (если исходная почти перестановка осей)."""
    U, _, Vt = np.linalg.svd(M.astype(float))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def parse_sensor_to_camera_rowmajor(flat: list[float]) -> np.ndarray:
    if len(flat) != 9:
        raise ValueError("--sensor_to_camera_rowmajor: нужно 9 чисел (строка за строкой)")
    M = np.array(flat, dtype=float).reshape(3, 3)
    return project_to_so3(M)


RIG_PRESETS = {
    # Схема: датчик X вниз ≈ ось Y камеры OpenCV (вниз), датчик Y вправо ≈ X камеры, Z вперёд ≈ Z камеры.
    # Сырые колонки [X_s|Y_s|Z_s] в базисе камеры дают det=-1; проецируем на SO(3).
    "diagram_xy_swap": project_to_so3(
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    ),
    "identity": np.eye(3, dtype=float),
}


def kabsch_rotation_align_predictions(R_pred: np.ndarray, R_opt: np.ndarray) -> np.ndarray:
    """
    Постоянная R_k ∈ SO(3), минимизирующая Σ_i ||R_k R_pred[i] - R_opt[i]||_F^2
    при ортонормированных R_pred[i], R_opt[i]: R_k = UV^T для M = Σ R_opt[i] @ R_pred[i].T.
    """
    n = R_pred.shape[0]
    if n < 1:
        return np.eye(3, dtype=float)
    M = np.zeros((3, 3), dtype=float)
    for i in range(n):
        M += R_opt[i] @ R_pred[i].T
    return project_to_so3(M)


def umeyama(src, dst):
    """
    Similarity transform: dst ≈ s * R @ src + t (src = ML, dst = оптика).
    Kabsch / Umeyama по центрированным точкам.
    """
    n = src.shape[0]
    if n < 3:
        raise ValueError("Нужно минимум 3 точки для выравнивания Umeyama")

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d
    H = xs.T @ xd
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt_adj = Vt.copy()
        Vt_adj[2, :] *= -1
        R = Vt_adj.T @ U.T
    denom = np.sum(xs ** 2)
    s = float(np.trace(R @ H) / denom) if denom > 1e-12 else 1.0
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def load_optic_csv(path, quat_order):
    """Читает AL_Optic.csv: Timestamp, x_Optic..z_Optic, q0..q3."""
    times = []
    pos = []
    quat = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = float(row["Timestamp"])
            except (KeyError, ValueError):
                continue
            x = float(row["x_Optic"])
            y = float(row["y_Optic"])
            z = float(row["z_Optic"])
            q0 = float(row["q0_Optic"])
            q1 = float(row["q1_Optic"])
            q2 = float(row["q2_Optic"])
            q3 = float(row["q3_Optic"])
            if quat_order == "wxyz":
                qw, qx, qy, qz = q0, q1, q2, q3
            else:
                qx, qy, qz, qw = q0, q1, q2, q3
            q = np.array([qw, qx, qy, qz], dtype=float)
            q /= np.linalg.norm(q) + 1e-12
            times.append(ts)
            pos.append([x, y, z])
            quat.append(q)
    return np.array(times), np.array(pos), np.array(quat)


def load_pred_csv(path):
    """pred_poses.csv: frame,tx,ty,tz,qw,qx,qy,qz"""
    frames = []
    pos = []
    quat = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = row["frame"].strip()
            try:
                idx = int(frame)
            except ValueError:
                idx = len(frames)
            tx, ty, tz = float(row["tx"]), float(row["ty"]), float(row["tz"])
            qw, qx, qy, qz = (
                float(row["qw"]),
                float(row["qx"]),
                float(row["qy"]),
                float(row["qz"]),
            )
            q = np.array([qw, qx, qy, qz], dtype=float)
            q /= np.linalg.norm(q) + 1e-12
            frames.append(idx)
            pos.append([tx, ty, tz])
            quat.append(q)
    return np.array(frames), np.array(pos), np.array(quat)


def interp_position(times_opt, pos_opt, t_query):
    """Линейная интерполяция позиции по времени."""
    return np.stack(
        [
            np.interp(t_query, times_opt, pos_opt[:, 0]),
            np.interp(t_query, times_opt, pos_opt[:, 1]),
            np.interp(t_query, times_opt, pos_opt[:, 2]),
        ],
        axis=1,
    )


def slerp_batch(times_opt, quat_opt, t_query):
    """Интерполяция кватерниона по времени (slerp между соседними отсчётами оптики)."""
    q = np.zeros((len(t_query), 4))
    for i in range(len(t_query)):
        tq = t_query[i]
        idx = int(np.searchsorted(times_opt, tq, side="left"))
        idx = np.clip(idx, 0, len(times_opt) - 1)
        idx_prev = max(0, idx - 1)
        dt = times_opt[idx] - times_opt[idx_prev]
        if dt < 1e-9:
            q[i] = quat_opt[idx]
            continue
        a = (tq - times_opt[idx_prev]) / dt
        a = float(np.clip(a, 0.0, 1.0))
        qa, qb = quat_opt[idx_prev], quat_opt[idx]
        d = abs(np.dot(qa, qb))
        if d > 0.9999:
            q[i] = qa
            continue
        theta = np.arccos(np.clip(d, 0, 1))
        s0 = np.sin((1 - a) * theta) / np.sin(theta)
        s1 = np.sin(a * theta) / np.sin(theta)
        sign = 1.0 if np.dot(qa, qb) >= 0 else -1.0
        q[i] = s0 * qa + s1 * sign * qb
        q[i] /= np.linalg.norm(q[i]) + 1e-12
    return q


def main():
    ap = argparse.ArgumentParser(description="ML vs optical tracking comparison")
    ap.add_argument("--optic_csv", type=str, required=True)
    ap.add_argument("--pred_csv", type=str, required=True)
    ap.add_argument("--frames_fps", type=float, required=True, help="FPS извлечения кадров (как в make_dataset.sh)")
    ap.add_argument("--time_offset_sec", type=float, default=0.0, help="Сдвиг: t_кадра += offset относительно Timestamp оптики")
    ap.add_argument(
        "--optic_quat_order",
        type=str,
        default="wxyz",
        choices=["wxyz", "xyzw"],
        help="Порядок q0..q3 в AL_Optic.csv",
    )
    ap.add_argument("--frame_start_index", type=int, default=1, help="Первый кадр 000001 -> индекс 1, время (index-1)/fps")
    ap.add_argument("--export_json", type=str, default=None, help="Сохранить метрики в JSON")
    ap.add_argument(
        "--rig_preset",
        type=str,
        default="diagram_xy_swap",
        choices=list(RIG_PRESETS.keys()),
        help="Жёсткое вращение датчик→камера (COLMAP/OpenCV: X вправо, Y вниз, Z вперёд). По умолчанию — схема X_s→Y_c, Y_s→X_c, Z_s→Z_c.",
    )
    ap.add_argument(
        "--sensor_to_camera_rowmajor",
        type=float,
        nargs=9,
        default=None,
        metavar=("r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"),
        help="Переопределить R_sensor_to_camera (9 чисел, строка за строкой); проецируется на SO(3). Имеет приоритет над --rig_preset.",
    )
    ap.add_argument(
        "--lever_cam",
        type=float,
        nargs=3,
        default=None,
        metavar=("dx", "dy", "dz"),
        help="Рычаг в метрах в СК камеры: вектор от оптического центра камеры к началу датчика (X вправо, Y вниз, Z вперёд). "
        "Позиция датчика в СК оптики после выравнивания: p_cam_aligned + R_align @ R_wc^T @ lever.",
    )
    ap.add_argument(
        "--optic_rotation_body_to_world_quat",
        action="store_true",
        help="По умолчанию q в AL_Optic как в COLMAP (world→тело): базис датчика в мире = R(q)^T. "
        "Если у вашего файла кватернион наоборот (тело→мир), укажите этот флаг.",
    )
    ap.add_argument(
        "--kabsch_orientation_calibration",
        action="store_true",
        help="Оценить постоянный поворот R_kabsch по всей траектории и пересчитать углы (полезно при "
        "несовпадении осей экспорта оптики и цепочки COLMAP/крепления). Базовые метрики без R_kabsch печатаются как раньше.",
    )
    ap.add_argument(
        "--no_relative_orientation",
        action="store_true",
        help="Не печатать метрики относительного поворота между соседними кадрами.",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.optic_csv):
        print(f"Нет файла: {args.optic_csv}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.pred_csv):
        print(f"Нет файла: {args.pred_csv}", file=sys.stderr)
        sys.exit(1)

    if args.sensor_to_camera_rowmajor is not None:
        R_sensor_to_camera = parse_sensor_to_camera_rowmajor(list(args.sensor_to_camera_rowmajor))
    else:
        R_sensor_to_camera = np.asarray(RIG_PRESETS[args.rig_preset], dtype=float)

    lever_cam = np.zeros(3, dtype=float)
    if args.lever_cam is not None:
        lever_cam = np.array(args.lever_cam, dtype=float)

    times_o, pos_o, quat_o = load_optic_csv(args.optic_csv, args.optic_quat_order)
    frames, pos_ml, quat_ml = load_pred_csv(args.pred_csv)

    order = np.argsort(frames)
    frames, pos_ml, quat_ml = frames[order], pos_ml[order], quat_ml[order]

    # Время кадра i (000001 -> 1): t = (i - frame_start_index) / fps + offset
    t_frames = (frames - args.frame_start_index) / args.frames_fps + args.time_offset_sec

    pos_o_i = interp_position(times_o, pos_o, t_frames)
    quat_o_i = slerp_batch(times_o, quat_o, t_frames)

    # Umeyama: центр камеры (COLMAP) ↔ позиция датчика в СК оптики (см. --lever_cam).
    s, R_align, t_align = umeyama(pos_ml, pos_o_i)

    pos_cam_aln = (s * (R_align @ pos_ml.T).T) + t_align
    n = len(frames)
    R_wc_all = np.stack([q_to_R_wxyz(quat_ml[i]) for i in range(n)], axis=0)
    if np.any(lever_cam != 0.0):
        off = np.einsum("ij,njk,k->ni", R_align, np.transpose(R_wc_all, (0, 2, 1)), lever_cam)
        pos_ml_aln = pos_cam_aln + off
    else:
        pos_ml_aln = pos_cam_aln

    pos_err = np.linalg.norm(pos_ml_aln - pos_o_i, axis=1)

    # Ориентация датчика в мире: R_wc^T @ R_sensor_to_camera → в СК оптики: R_align @ R_wc^T @ R_sensor_to_camera.
    R_pred_all = np.zeros((n, 3, 3), dtype=float)
    R_opt_all = np.zeros((n, 3, 3), dtype=float)
    rot_err = np.zeros(n, dtype=float)
    for i in range(n):
        R_wc = R_wc_all[i]
        R_pred_all[i] = R_align @ R_wc.T @ R_sensor_to_camera
        R_o = q_to_R_wxyz(quat_o_i[i])
        R_opt_all[i] = R_o if args.optic_rotation_body_to_world_quat else R_o.T
        q_pred = R_to_quat_wxyz(R_pred_all[i])
        q_opt = R_to_quat_wxyz(R_opt_all[i])
        rot_err[i] = quat_angle_deg(q_pred, q_opt)

    R_kabsch = None
    rot_err_cal = None
    if args.kabsch_orientation_calibration and n >= 1:
        R_kabsch = kabsch_rotation_align_predictions(R_pred_all, R_opt_all)
        rot_err_cal = np.zeros(n, dtype=float)
        for i in range(n):
            q_pred = R_to_quat_wxyz(R_kabsch @ R_pred_all[i])
            q_opt = R_to_quat_wxyz(R_opt_all[i])
            rot_err_cal[i] = quat_angle_deg(q_pred, q_opt)

    rel_err = None
    if not args.no_relative_orientation and n >= 2:
        rel_err = np.zeros(n - 1, dtype=float)
        for i in range(n - 1):
            d_pred = R_pred_all[i + 1] @ R_pred_all[i].T
            d_opt = R_opt_all[i + 1] @ R_opt_all[i].T
            rel_err[i] = quat_angle_deg(R_to_quat_wxyz(d_pred), R_to_quat_wxyz(d_opt))

    print("=== Калибровка жёсткого крепления (датчик → камера OpenCV/COLMAP) ===")
    print(f"  R_sensor_to_camera =\n{R_sensor_to_camera}")
    if np.any(lever_cam != 0.0):
        print(f"  lever_cam (м): {lever_cam}")
    print()
    print("=== Выравнивание (Umeyama): p_opt ≈ s * R @ p_cam_colmap + t ===")
    print(f"  scale s = {s:.6f}")
    print(f"  R =\n{R_align}")
    print(f"  t = {t_align}")
    print(f"  Пар кадров: {len(frames)}")
    print()
    pos_med = float(np.median(pos_err))
    rot_med = float(np.median(rot_err))
    pos_rmse = float(np.sqrt(np.mean(pos_err**2)))

    print("=== Базовые метрики (эталон — оптика; после Umeyama и калибровки крепления) ===")
    if np.any(lever_cam != 0.0):
        print("  Позиция: центр датчика (p_cam_aligned + R_align @ R_wc^T @ lever_cam).")
    else:
        print("  Позиция: центр камеры (для центра датчика задайте --lever_cam dx dy dz, м).")
    print()
    print("  Позиция — евклидово расстояние до эталона, м:")
    print(f"    максимальное отклонение:  {pos_err.max():.6f}")
    print(f"    среднее отклонение:       {pos_err.mean():.6f}")
    print()
    print("  Ориентация — сферическое расстояние на SO(3) (угол между ориентациями), °:")
    print(f"    максимальное:             {rot_err.max():.4f}")
    print(f"    среднее:                  {rot_err.mean():.4f}")
    print()
    print("  Дополнительно: медиана по позиции (м) = {:.6f}; по углу (°) = {:.4f}; RMSE позиции (м) = {:.6f}".format(
        pos_med, rot_med, pos_rmse
    ))
    print()

    if rel_err is not None:
        print("=== Относительная ориентация (соседние кадры), ° ===")
        print("  Угол между приращениями поворота ML и оптики: ΔR = R[i+1] @ R[i]^T.")
        print(f"    максимальное:             {rel_err.max():.4f}")
        print(f"    среднее:                  {rel_err.mean():.4f}")
        print(f"    медиана:                  {np.median(rel_err):.4f}")
        print()

    if R_kabsch is not None and rot_err_cal is not None:
        print("=== Ориентация после оценки постоянного поворота R_kabsch (калибровка осей по траектории) ===")
        print("  R_kabsch оценивается по всем кадрам (Kabsch в SO(3)): снимает часть систематического")
        print("  сдвига между экспортом оптики и цепочкой COLMAP/крепления. Не заменяет измерение рычага.")
        print(f"  R_kabsch =\n{R_kabsch}")
        print()
        print("  Сферическое расстояние на SO(3), ° (после R_kabsch @ R_pred):")
        print(f"    максимальное:             {rot_err_cal.max():.4f}")
        print(f"    среднее:                  {rot_err_cal.mean():.4f}")
        print(f"    медиана:                  {np.median(rot_err_cal):.4f}")
        print()

    print("Эталон: оптика (AL_Optic). ML: PoseNet/COLMAP → ориентация датчика + Umeyama.")
    print("Большие абсолютные углы: --kabsch_orientation_calibration, --optic_rotation_body_to_world_quat, --sensor_to_camera_rowmajor.")

    if args.export_json:
        import json

        out = {
            "rig_preset": args.rig_preset,
            "sensor_to_camera": R_sensor_to_camera.tolist(),
            "lever_cam_m": lever_cam.tolist(),
            "scale": s,
            "rotation": R_align.tolist(),
            "translation": t_align.tolist(),
            "rotation_align": R_align.tolist(),
            "translation_align": t_align.tolist(),
            "num_pairs": len(frames),
            "position_error_m": {
                "max_deviation": float(pos_err.max()),
                "mean_deviation": float(pos_err.mean()),
                "median": pos_med,
                "rmse": pos_rmse,
            },
            "orientation_spherical_distance_deg": {
                "max": float(rot_err.max()),
                "mean": float(rot_err.mean()),
                "median": rot_med,
            },
            "position_m": {
                "max": float(pos_err.max()),
                "mean": float(pos_err.mean()),
                "median": pos_med,
                "rmse": pos_rmse,
            },
            "orientation_deg": {
                "max": float(rot_err.max()),
                "mean": float(rot_err.mean()),
                "median": rot_med,
            },
        }
        if rel_err is not None:
            out["relative_orientation_delta_deg"] = {
                "max": float(rel_err.max()),
                "mean": float(rel_err.mean()),
                "median": float(np.median(rel_err)),
            }
        if R_kabsch is not None and rot_err_cal is not None:
            out["kabsch_rotation"] = R_kabsch.tolist()
            out["orientation_spherical_distance_after_kabsch_deg"] = {
                "max": float(rot_err_cal.max()),
                "mean": float(rot_err_cal.mean()),
                "median": float(np.median(rot_err_cal)),
            }

        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"JSON: {args.export_json}")


if __name__ == "__main__":
    main()
