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

Дополнительно в JSON: медианы по позиции и по углу; при --kabsch_orientation_calibration — углы после R_kabsch слева;
при --kabsch_orientation_calibration_right — после R_pred @ R_r; при --probe_orientation_conventions — таблицы полуоборотов π и Kabsch;
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
import math
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


def R_to_quat_wxyz_batch(R: np.ndarray) -> np.ndarray:
    """Матрицы вращения (N,3,3) → кватернионы (N,4) wxyz; устойчиво для пакета."""
    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 2:
        return R_to_quat_wxyz(R).reshape(1, 4)
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    N = R.shape[0]
    qw = np.empty(N)
    qx = np.empty(N)
    qy = np.empty(N)
    qz = np.empty(N)
    c0 = tr > 0
    if np.any(c0):
        S = np.sqrt(tr[c0] + 1.0) * 2
        qw[c0] = 0.25 * S
        qx[c0] = (m21[c0] - m12[c0]) / S
        qy[c0] = (m02[c0] - m20[c0]) / S
        qz[c0] = (m10[c0] - m01[c0]) / S
    c1 = ~c0 & (m00 > m11) & (m00 > m22)
    c2 = ~c0 & ~c1 & (m11 > m22)
    c3 = ~c0 & ~c1 & ~c2
    if np.any(c1):
        S = np.sqrt(1.0 + m00[c1] - m11[c1] - m22[c1]) * 2
        qw[c1] = (m21[c1] - m12[c1]) / S
        qx[c1] = 0.25 * S
        qy[c1] = (m01[c1] + m10[c1]) / S
        qz[c1] = (m02[c1] + m20[c1]) / S
    if np.any(c2):
        S = np.sqrt(1.0 + m11[c2] - m00[c2] - m22[c2]) * 2
        qw[c2] = (m02[c2] - m20[c2]) / S
        qx[c2] = (m01[c2] + m10[c2]) / S
        qy[c2] = 0.25 * S
        qz[c2] = (m12[c2] + m21[c2]) / S
    if np.any(c3):
        S = np.sqrt(1.0 + m22[c3] - m00[c3] - m11[c3]) * 2
        qw[c3] = (m10[c3] - m01[c3]) / S
        qx[c3] = (m02[c3] + m20[c3]) / S
        qy[c3] = (m12[c3] + m21[c3]) / S
        qz[c3] = 0.25 * S
    q = np.stack([qw, qx, qy, qz], axis=1)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    return q


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


def R_euler_zyx_deg(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Поворот R = Rz(rz) Ry(ry) Rx(rx), углы в градусах (правило ZYX)."""
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return Rz @ Ry @ Rx


def grid_search_sensor_to_camera_box(
    A_stack: np.ndarray,
    R_opt_all: np.ndarray,
    rx_list: list[int],
    ry_list: list[int],
    rz_list: list[int],
    chunk_size: int = 2048,
) -> tuple[np.ndarray, tuple[int, int, int], float]:
    """
    Перебор R_sensor_to_camera = Rz*Ry*Rx на декартовом произведении списков углов (градусы).
    Минимизируется средняя угловая ошибка между A_stack[i] @ R_sc и R_opt_all[i].
    """
    n = A_stack.shape[0]
    if not rx_list or not ry_list or not rz_list:
        return np.eye(3, dtype=float), (0, 0, 0), float("inf")
    Lx, Ly, Lz = len(rx_list), len(ry_list), len(rz_list)
    total = Lx * Ly * Lz
    q_opt = np.stack([R_to_quat_wxyz(R_opt_all[i]) for i in range(n)])

    def euler_idx_to_triple(idx: int) -> tuple[int, int, int]:
        iz = idx % Lz
        idx //= Lz
        iy = idx % Ly
        ix = idx // Ly
        return rx_list[ix], ry_list[iy], rz_list[iz]

    best_mean = float("inf")
    best_idx = 0
    cs = max(32, int(chunk_size))

    for chunk_start in range(0, total, cs):
        chunk_end = min(chunk_start + cs, total)
        B = chunk_end - chunk_start
        R_chunk = np.zeros((B, 3, 3), dtype=float)
        for j in range(B):
            rx, ry, rz = euler_idx_to_triple(chunk_start + j)
            R_chunk[j] = R_euler_zyx_deg(float(rx), float(ry), float(rz))
        R_pred = np.einsum("nij,bjk->nbik", A_stack, R_chunk)
        R_bn = np.transpose(R_pred, (1, 0, 2, 3))
        flat = R_bn.reshape(-1, 3, 3)
        q_pred = R_to_quat_wxyz_batch(flat).reshape(B, n, 4).transpose(1, 0, 2)
        d = np.abs(np.sum(q_pred * q_opt[:, None, :], axis=2))
        d = np.clip(d, 0.0, 1.0)
        ang = 2.0 * np.degrees(np.arccos(d))
        mean_per = ang.mean(axis=0)
        local_j = int(np.argmin(mean_per))
        m = float(mean_per[local_j])
        if m < best_mean:
            best_mean = m
            best_idx = chunk_start + local_j

    rx_b, ry_b, rz_b = euler_idx_to_triple(best_idx)
    best_R = R_euler_zyx_deg(float(rx_b), float(ry_b), float(rz_b))
    return project_to_so3(best_R), (rx_b, ry_b, rz_b), best_mean


def grid_search_sensor_to_camera(
    A_stack: np.ndarray,
    R_opt_all: np.ndarray,
    step_deg: int,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, tuple[int, int, int], float]:
    """
    Полный перебор R_sensor_to_camera = R_euler_zyx_deg(rx,ry,rz) с шагом step_deg по [-180,180).
    """
    step = max(1, int(step_deg))
    rx_list = list(range(-180, 180, step))
    if not rx_list:
        rx_list = [0]
    return grid_search_sensor_to_camera_box(A_stack, R_opt_all, rx_list, rx_list, rx_list, chunk_size)


def grid_search_sensor_to_camera_local(
    A_stack: np.ndarray,
    R_opt_all: np.ndarray,
    center_rx: int,
    center_ry: int,
    center_rz: int,
    half_width_deg: int,
    step_deg: int,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, tuple[int, int, int], float]:
    """
    Локальная сетка вокруг (center_rx, center_ry, center_rz) в градусах: по каждой оси
    от -half_width до +half_width с шагом step_deg; углы нормализуются в (-180, 180].
    """
    step = max(1, int(step_deg))
    half = max(0, int(half_width_deg))

    def axis_values(c: int) -> list[int]:
        seen: set[int] = set()
        for d in range(-half, half + 1, step):
            a = int(((int(c) + d + 180) % 360) - 180)
            seen.add(a)
        return sorted(seen)

    rx_list = axis_values(center_rx)
    ry_list = axis_values(center_ry)
    rz_list = axis_values(center_rz)
    return grid_search_sensor_to_camera_box(A_stack, R_opt_all, rx_list, ry_list, rz_list, chunk_size)


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


def kabsch_rotation_align_predictions_right(R_pred: np.ndarray, R_opt: np.ndarray) -> np.ndarray:
    """
    Постоянная R_r ∈ SO(3), минимизирующая Σ_i ||R_pred[i] R_r - R_opt[i]||_F^2
    (ошибка справа на предсказании): M = Σ R_pred[i].T @ R_opt[i] → project_to_so3(M).
    """
    n = R_pred.shape[0]
    if n < 1:
        return np.eye(3, dtype=float)
    M = np.zeros((3, 3), dtype=float)
    for i in range(n):
        M += R_pred[i].T @ R_opt[i]
    return project_to_so3(M)


def orientation_errors_between(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """Покадровое сферическое расстояние между матрицами R_a[i] и R_b[i], градусы."""
    n = R_a.shape[0]
    errs = np.zeros(n, dtype=float)
    for i in range(n):
        errs[i] = quat_angle_deg(R_to_quat_wxyz(R_a[i]), R_to_quat_wxyz(R_b[i]))
    return errs


def rotation_geodesic_from_identity_deg(R: np.ndarray) -> float:
    """Угол поворота R ∈ SO(3) относительно единицы: acos((tr R - 1) / 2), градусы."""
    tr = float(np.clip((np.trace(R.astype(float)) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(tr)))


# Полуобороты π вокруг осей мира (для проверки «не хватает 180°»).
_HALF_TURN_LABELS_AND_R = (
    ("I", np.eye(3, dtype=float)),
    ("Rx(pi)", np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)),
    ("Ry(pi)", np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)),
    ("Rz(pi)", np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)),
)


def rotation_errors_deg(R_left: np.ndarray, R_pred: np.ndarray, R_opt: np.ndarray) -> np.ndarray:
    """Угловая ошибка (SO(3), градусы) для R_left @ R_pred[i] против R_opt[i]."""
    n = R_pred.shape[0]
    errs = np.zeros(n, dtype=float)
    for i in range(n):
        q_pred = R_to_quat_wxyz(R_left @ R_pred[i])
        q_opt = R_to_quat_wxyz(R_opt[i])
        errs[i] = quat_angle_deg(q_pred, q_opt)
    return errs


def estimate_const_rotation_ransac(
    R_pred: np.ndarray,
    R_opt: np.ndarray,
    iters: int,
    thresh_deg: float,
    min_samples: int,
    seed: int,
):
    """
    Оценка постоянного поворота R0 (R0 @ R_pred ≈ R_opt) через RANSAC + МНК.
    Возвращает (R0, inliers_mask, inlier_errors_deg).
    """
    n = R_pred.shape[0]
    if n < 3:
        R0 = kabsch_rotation_align_predictions(R_pred, R_opt)
        errs = rotation_errors_deg(R0, R_pred, R_opt)
        inl = np.ones(n, dtype=bool)
        return R0, inl, errs

    rng = np.random.default_rng(seed)
    sample_size = max(3, min(min_samples, n))

    best_inliers = None
    best_count = -1
    best_mean = float("inf")

    for _ in range(max(1, iters)):
        idx = rng.choice(n, size=sample_size, replace=False)
        R_cand = kabsch_rotation_align_predictions(R_pred[idx], R_opt[idx])
        errs = rotation_errors_deg(R_cand, R_pred, R_opt)
        inliers = errs <= thresh_deg
        count = int(np.sum(inliers))
        mean_err = float(errs[inliers].mean()) if count > 0 else float("inf")
        if count > best_count or (count == best_count and mean_err < best_mean):
            best_count = count
            best_mean = mean_err
            best_inliers = inliers

    if best_inliers is None or np.sum(best_inliers) < 3:
        best_inliers = np.ones(n, dtype=bool)

    R0 = kabsch_rotation_align_predictions(R_pred[best_inliers], R_opt[best_inliers])
    errs = rotation_errors_deg(R0, R_pred, R_opt)
    return R0, best_inliers, errs


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
        "--kabsch_orientation_calibration_right",
        action="store_true",
        help="Оценить постоянный R_r в смысле R_pred @ R_r ≈ R_opt (МНК в SO(3)) и печатать метрики.",
    )
    ap.add_argument(
        "--probe_orientation_conventions",
        action="store_true",
        help="Перебор полуоборотов π (I, Rx, Ry, Rz) на эталоне и предсказании; Kabsch слева и справа; "
        "угол поворота R_k от единицы. Для проверки гипотезы «константный полуоборот / другая сторона умножения».",
    )
    ap.add_argument(
        "--no_relative_orientation",
        action="store_true",
        help="Не печатать метрики относительного поворота между соседними кадрами.",
    )
    ap.add_argument(
        "--calibrate_const_rotation",
        action="store_true",
        help="Оценить константный поворот R0 по началу траектории (RANSAC+МНК), затем считать ошибки на оставшихся кадрах.",
    )
    ap.add_argument(
        "--calib_fraction",
        type=float,
        default=0.2,
        help="Доля начальных кадров для оценки R0 (используется с --calibrate_const_rotation).",
    )
    ap.add_argument(
        "--ransac_iters",
        type=int,
        default=300,
        help="Число итераций RANSAC для оценки R0.",
    )
    ap.add_argument(
        "--ransac_thresh_deg",
        type=float,
        default=20.0,
        help="Порог инлайера RANSAC по угловой ошибке, градусы.",
    )
    ap.add_argument(
        "--ransac_min_samples",
        type=int,
        default=8,
        help="Размер случайной подвыборки на итерацию RANSAC.",
    )
    ap.add_argument(
        "--ransac_seed",
        type=int,
        default=42,
        help="Сид генератора RANSAC.",
    )
    ap.add_argument(
        "--grid_search_sensor_deg",
        type=int,
        default=0,
        help="Перебор R_sensor_to_camera как Rz*Ry*Rx на сетке с шагом (градусы); 0=выкл. "
        "SO(3) непрерывен — это грубая сетка, число вариантов ~(360/step)³. Несовместимо с --sensor_to_camera_rowmajor.",
    )
    ap.add_argument(
        "--grid_local_refine_deg",
        type=int,
        default=0,
        help="После глобальной сетки (или вокруг --grid_local_center_deg) — локальное уточнение с шагом (°). "
        "0=выкл. Нужен центр: либо три угла из --grid_local_center_deg, либо предшествующий --grid_search_sensor_deg.",
    )
    ap.add_argument(
        "--grid_local_half_width_deg",
        type=int,
        default=10,
        help="Полуширина локальной сетки по каждой оси Euler (градусы), вместе с --grid_local_refine_deg.",
    )
    ap.add_argument(
        "--grid_local_center_deg",
        type=int,
        nargs=3,
        default=None,
        metavar=("rx", "ry", "rz"),
        help="Центр локальной сетки ZYX (°). Если не задан — берутся лучшие углы после --grid_search_sensor_deg.",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.optic_csv):
        print(f"Нет файла: {args.optic_csv}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.pred_csv):
        print(f"Нет файла: {args.pred_csv}", file=sys.stderr)
        sys.exit(1)

    if args.grid_search_sensor_deg > 0 and args.sensor_to_camera_rowmajor is not None:
        print(
            "Нельзя одновременно --grid_search_sensor_deg и --sensor_to_camera_rowmajor",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.grid_local_refine_deg > 0 and args.sensor_to_camera_rowmajor is not None:
        print(
            "Нельзя одновременно --grid_local_refine_deg и --sensor_to_camera_rowmajor",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.grid_local_refine_deg > 0 and args.grid_local_center_deg is None and args.grid_search_sensor_deg <= 0:
        print(
            "Для --grid_local_refine_deg без --grid_search_sensor_deg задайте --grid_local_center_deg rx ry rz",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.sensor_to_camera_rowmajor is not None:
        R_sensor_to_camera = parse_sensor_to_camera_rowmajor(list(args.sensor_to_camera_rowmajor))
    elif args.grid_search_sensor_deg > 0 or args.grid_local_refine_deg > 0:
        R_sensor_to_camera = np.eye(3, dtype=float)  # временно, пересчитается ниже
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

    # Базис датчика в мире оптики (не зависит от R_sensor_to_camera).
    R_opt_all = np.zeros((n, 3, 3), dtype=float)
    for i in range(n):
        R_o = q_to_R_wxyz(quat_o_i[i])
        R_opt_all[i] = R_o if args.optic_rotation_body_to_world_quat else R_o.T

    # A[i] = R_align @ R_wc^T  →  R_pred[i] = A[i] @ R_sensor_to_camera
    A_stack = np.einsum("ij,njk->nik", R_align, np.transpose(R_wc_all, (0, 2, 1)))

    grid_search_meta = None
    euler_after_global: tuple[int, int, int] | None = None
    if args.grid_search_sensor_deg > 0:
        step = max(1, int(args.grid_search_sensor_deg))
        n_ang = len(list(range(-180, 180, step)))
        n_total = n_ang**3
        print(f"=== Перебор R_sensor_to_camera (Euler ZYX, шаг {step}°), ~{n_total} комбинаций ===")
        R_sensor_to_camera, euler_best, mean_fit = grid_search_sensor_to_camera(
            A_stack, R_opt_all, step
        )
        euler_after_global = euler_best
        grid_search_meta = {
            "step_deg": step,
            "euler_zyx_best_deg": list(euler_best),
            "mean_orientation_error_deg_on_grid_objective": float(mean_fit),
            "num_grid_points": int(n_total),
        }
        print(f"  Лучшие rx,ry,rz (°): {euler_best[0]}, {euler_best[1]}, {euler_best[2]}")
        print(f"  Средняя угловая ошибка на сетке (целевая): {mean_fit:.4f}°")
        print(f"  R_sensor_to_camera =\n{R_sensor_to_camera}")
        print()

    if args.grid_local_refine_deg > 0:
        step_l = max(1, int(args.grid_local_refine_deg))
        half_w = max(0, int(args.grid_local_half_width_deg))
        if args.grid_local_center_deg is not None:
            cx, cy, cz = (int(x) for x in args.grid_local_center_deg)
        elif euler_after_global is not None:
            cx, cy, cz = euler_after_global
        else:
            cx, cy, cz = 0, 0, 0
        rx_axis = len(
            sorted(
                {int(((cx + d + 180) % 360) - 180) for d in range(-half_w, half_w + 1, step_l)}
            )
        )
        n_local = rx_axis**3
        print(
            f"=== Локальное уточнение Euler ZYX вокруг ({cx},{cy},{cz})°, "
            f"±{half_w}°, шаг {step_l}° (~{n_local} комбинаций) ==="
        )
        R_loc, euler_loc, mean_loc = grid_search_sensor_to_camera_local(
            A_stack, R_opt_all, cx, cy, cz, half_w, step_l
        )
        R_sensor_to_camera = R_loc
        loc_meta = {
            "half_width_deg": half_w,
            "step_deg": step_l,
            "center_deg": [cx, cy, cz],
            "euler_zyx_best_deg": list(euler_loc),
            "mean_orientation_error_deg_on_grid_objective": float(mean_loc),
            "num_grid_points": int(n_local),
        }
        if grid_search_meta is not None:
            grid_search_meta["local_refine"] = loc_meta
        else:
            grid_search_meta = {"local_refine_only": True, **loc_meta}
        print(f"  После уточнения rx,ry,rz (°): {euler_loc[0]}, {euler_loc[1]}, {euler_loc[2]}")
        print(f"  Средняя угловая ошибка (целевая): {mean_loc:.4f}°")
        print(f"  R_sensor_to_camera =\n{R_sensor_to_camera}")
        print()

    R_pred_all = np.zeros((n, 3, 3), dtype=float)
    rot_err = np.zeros(n, dtype=float)
    for i in range(n):
        R_pred_all[i] = A_stack[i] @ R_sensor_to_camera
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

    R_kabsch_r = None
    rot_err_cal_r = None
    if args.kabsch_orientation_calibration_right and n >= 1:
        R_kabsch_r = kabsch_rotation_align_predictions_right(R_pred_all, R_opt_all)
        R_pred_r = np.einsum("nij,jk->nik", R_pred_all, R_kabsch_r)
        rot_err_cal_r = orientation_errors_between(R_pred_r, R_opt_all)

    probe_report: dict | None = None
    if args.probe_orientation_conventions and n >= 1:
        probe_report = {}

        def _row_dict(name: str, errs: np.ndarray) -> dict:
            return {
                "name": name,
                "mean_deg": float(errs.mean()),
                "median_deg": float(np.median(errs)),
                "max_deg": float(errs.max()),
            }

        rows_lo = []
        for name, Rpi in _HALF_TURN_LABELS_AND_R:
            R_opt_m = np.einsum("ij,njk->nik", Rpi, R_opt_all)
            e = orientation_errors_between(R_pred_all, R_opt_m)
            rows_lo.append(_row_dict(name, e))
        probe_report["compare_Rpred_vs_Rpi_Ropt"] = rows_lo

        rows_rp = []
        for name, Rpi in _HALF_TURN_LABELS_AND_R:
            R_pred_m = np.einsum("nij,jk->nik", R_pred_all, Rpi)
            e = orientation_errors_between(R_pred_m, R_opt_all)
            rows_rp.append(_row_dict(name, e))
        probe_report["compare_Rpred_Rpi_vs_Ropt"] = rows_rp

        R_kl = kabsch_rotation_align_predictions(R_pred_all, R_opt_all)
        R_pred_kl = np.einsum("ij,njk->nik", R_kl, R_pred_all)
        e_kl = orientation_errors_between(R_pred_kl, R_opt_all)
        probe_report["kabsch_left_Rk_Rpred"] = {
            "R_k": R_kl.tolist(),
            "rotation_angle_from_identity_deg": rotation_geodesic_from_identity_deg(R_kl),
            "mean_deg": float(e_kl.mean()),
            "median_deg": float(np.median(e_kl)),
            "max_deg": float(e_kl.max()),
        }

        R_kr = kabsch_rotation_align_predictions_right(R_pred_all, R_opt_all)
        R_pred_kr = np.einsum("nij,jk->nik", R_pred_all, R_kr)
        e_kr = orientation_errors_between(R_pred_kr, R_opt_all)
        probe_report["kabsch_right_Rpred_Rr"] = {
            "R_r": R_kr.tolist(),
            "rotation_angle_from_identity_deg": rotation_geodesic_from_identity_deg(R_kr),
            "mean_deg": float(e_kr.mean()),
            "median_deg": float(np.median(e_kr)),
            "max_deg": float(e_kr.max()),
        }

    rel_err = None
    if not args.no_relative_orientation and n >= 2:
        rel_err = np.zeros(n - 1, dtype=float)
        for i in range(n - 1):
            d_pred = R_pred_all[i + 1] @ R_pred_all[i].T
            d_opt = R_opt_all[i + 1] @ R_opt_all[i].T
            rel_err[i] = quat_angle_deg(R_to_quat_wxyz(d_pred), R_to_quat_wxyz(d_opt))

    split_info = None
    if args.calibrate_const_rotation:
        frac = float(np.clip(args.calib_fraction, 0.0, 1.0))
        n_cal = int(round(n * frac))
        n_cal = max(3, min(n - 1, n_cal))
        cal_idx = np.arange(n_cal)
        eval_idx = np.arange(n_cal, n)

        R0, inliers_mask, cal_err_before = estimate_const_rotation_ransac(
            R_pred_all[cal_idx],
            R_opt_all[cal_idx],
            iters=args.ransac_iters,
            thresh_deg=args.ransac_thresh_deg,
            min_samples=args.ransac_min_samples,
            seed=args.ransac_seed,
        )
        cal_err_after = rotation_errors_deg(R0, R_pred_all[cal_idx], R_opt_all[cal_idx])
        eval_err_before = rotation_errors_deg(np.eye(3), R_pred_all[eval_idx], R_opt_all[eval_idx])
        eval_err_after = rotation_errors_deg(R0, R_pred_all[eval_idx], R_opt_all[eval_idx])
        pos_eval = pos_err[eval_idx]

        split_info = {
            "n_total": n,
            "n_calib": int(n_cal),
            "n_eval": int(len(eval_idx)),
            "R0": R0,
            "cal_inliers": int(np.sum(inliers_mask)),
            "cal_inlier_ratio": float(np.mean(inliers_mask)),
            "cal_err_before": cal_err_before,
            "cal_err_after": cal_err_after,
            "eval_err_before": eval_err_before,
            "eval_err_after": eval_err_after,
            "pos_eval": pos_eval,
        }

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
        ang_id = rotation_geodesic_from_identity_deg(R_kabsch)
        print(f"  Угол поворота R_kabsch относительно I (геодезия SO(3)): {ang_id:.4f}°")
        print()
        print("  Сферическое расстояние на SO(3), ° (после R_kabsch @ R_pred):")
        print(f"    максимальное:             {rot_err_cal.max():.4f}")
        print(f"    среднее:                  {rot_err_cal.mean():.4f}")
        print(f"    медиана:                  {np.median(rot_err_cal):.4f}")
        print()

    if R_kabsch_r is not None and rot_err_cal_r is not None:
        print("=== Ориентация после R_pred @ R_kabsch_right (Kabsch справа) ===")
        print(f"  R_kabsch_right =\n{R_kabsch_r}")
        print(f"  Угол поворота относительно I: {rotation_geodesic_from_identity_deg(R_kabsch_r):.4f}°")
        print()
        print("  Сферическое расстояние на SO(3), °:")
        print(f"    максимальное:             {rot_err_cal_r.max():.4f}")
        print(f"    среднее:                  {rot_err_cal_r.mean():.4f}")
        print(f"    медиана:                  {np.median(rot_err_cal_r):.4f}")
        print()

    if probe_report is not None:
        print("=== Проверка конвенций (гипотеза «π» / сторона умножения), ° ===")
        print("  Сравниваем R_pred с вариантами эталона R_opt и предсказания (полуобороты в СК мира оптики).")
        print(
            "  Для полуоборотов R_π=R_π^T угол(R_pred, R_π R_opt) совпадает с углом(R_π R_pred, R_opt) "
            "(один относительный поворот R_pred^T R_π R_opt)."
        )
        print()

        def _print_probe_table(title: str, rows: list) -> None:
            print(f"  {title}")
            print(f"    {'имя':<8}  {'mean':>8}  {'median':>8}  {'max':>8}")
            best = min(rows, key=lambda r: r["mean_deg"])
            for r in sorted(rows, key=lambda x: x["mean_deg"]):
                mark = " *" if r["name"] == best["name"] else ""
                print(
                    f"    {r['name']:<8}  {r['mean_deg']:8.4f}  {r['median_deg']:8.4f}  {r['max_deg']:8.4f}{mark}"
                )
            print()

        _print_probe_table("R_pred vs (R_π @ R_opt)  [= (R_π @ R_pred) vs R_opt при R_π=R_π^T]:", probe_report["compare_Rpred_vs_Rpi_Ropt"])
        _print_probe_table("(R_pred @ R_π) vs R_opt (другая сторона умножения):", probe_report["compare_Rpred_Rpi_vs_Ropt"])

        kl = probe_report["kabsch_left_Rk_Rpred"]
        kr = probe_report["kabsch_right_Rpred_Rr"]
        print("  Kabsch слева: R_k @ R_pred ≈ R_opt")
        print(f"    mean/median/max °: {kl['mean_deg']:.4f} / {kl['median_deg']:.4f} / {kl['max_deg']:.4f}")
        print(f"    угол(R_k) от I:    {kl['rotation_angle_from_identity_deg']:.4f}°")
        print()
        print("  Kabsch справа: R_pred @ R_r ≈ R_opt")
        print(f"    mean/median/max °: {kr['mean_deg']:.4f} / {kr['median_deg']:.4f} / {kr['max_deg']:.4f}")
        print(f"    угол(R_r) от I:    {kr['rotation_angle_from_identity_deg']:.4f}°")
        print()

    if split_info is not None:
        print("=== Константный поворот по схеме МНК + RANSAC (калибровка на начале, оценка на остатке) ===")
        print(
            f"  Калибровка: {split_info['n_calib']} кадров из {split_info['n_total']} "
            f"({split_info['n_calib'] / split_info['n_total']:.1%}), "
            f"инлайеры RANSAC: {split_info['cal_inliers']} ({split_info['cal_inlier_ratio']:.1%})"
        )
        print(f"  R0 =\n{split_info['R0']}")
        print()
        print("  На ОСТАТКЕ кадров (честная оценка после фиксации R0):")
        print(f"    Позиция max/mean, м:      {split_info['pos_eval'].max():.6f} / {split_info['pos_eval'].mean():.6f}")
        print(
            f"    Ориентация ДО R0 max/mean/med, °:    "
            f"{split_info['eval_err_before'].max():.4f} / {split_info['eval_err_before'].mean():.4f} / {np.median(split_info['eval_err_before']):.4f}"
        )
        print(
            f"    Ориентация ПОСЛЕ R0 max/mean/med, °: "
            f"{split_info['eval_err_after'].max():.4f} / {split_info['eval_err_after'].mean():.4f} / {np.median(split_info['eval_err_after']):.4f}"
        )
        print()
        print("  (Те же 4 пункта научника по ориентации, но на eval-участке и ПОСЛЕ фиксации R0:)")
        print("    максимальное по сферическому расстоянию, °:  {:.4f}".format(split_info["eval_err_after"].max()))
        print("    среднее по сферическому расстоянию, °:      {:.4f}".format(split_info["eval_err_after"].mean()))
        print("  Позиция на eval (макс./среднее), м:          {:.6f} / {:.6f}".format(
            split_info["pos_eval"].max(), split_info["pos_eval"].mean()
        ))
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
        if grid_search_meta is not None:
            out["grid_search_sensor_to_camera"] = grid_search_meta
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
            out["kabsch_left_angle_from_identity_deg"] = rotation_geodesic_from_identity_deg(R_kabsch)
        if R_kabsch_r is not None and rot_err_cal_r is not None:
            out["kabsch_rotation_right"] = R_kabsch_r.tolist()
            out["orientation_spherical_distance_after_kabsch_right_deg"] = {
                "max": float(rot_err_cal_r.max()),
                "mean": float(rot_err_cal_r.mean()),
                "median": float(np.median(rot_err_cal_r)),
            }
            out["kabsch_right_angle_from_identity_deg"] = rotation_geodesic_from_identity_deg(R_kabsch_r)
        if probe_report is not None:
            out["orientation_probe"] = probe_report
        if split_info is not None:
            out["const_rotation_ransac_lsq"] = {
                "R0": split_info["R0"].tolist(),
                "n_total": split_info["n_total"],
                "n_calib": split_info["n_calib"],
                "n_eval": split_info["n_eval"],
                "calib_inliers": split_info["cal_inliers"],
                "calib_inlier_ratio": split_info["cal_inlier_ratio"],
                "eval_position_m": {
                    "max": float(split_info["pos_eval"].max()),
                    "mean": float(split_info["pos_eval"].mean()),
                    "median": float(np.median(split_info["pos_eval"])),
                },
                "eval_orientation_before_deg": {
                    "max": float(split_info["eval_err_before"].max()),
                    "mean": float(split_info["eval_err_before"].mean()),
                    "median": float(np.median(split_info["eval_err_before"])),
                },
                "eval_orientation_after_deg": {
                    "max": float(split_info["eval_err_after"].max()),
                    "mean": float(split_info["eval_err_after"].mean()),
                    "median": float(np.median(split_info["eval_err_after"])),
                },
            }

        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"JSON: {args.export_json}")


if __name__ == "__main__":
    main()
