#!/usr/bin/env python3
"""
Подбор настроек сравнения ML vs оптика для минимума среднего сферического расстояния ориентации (°).

Перебирает:
  - порядок кватерниона в CSV (wxyz / xyzw);
  - смысл кватерниона оптики (флаг body_to_world как в compare_ml_optic);
  - time_offset_sec (синхронизация видео↔оптика);
  - сетку Euler ZYX для R_sensor_to_camera (грубо, затем плотнее).

Запуск из корня репозитория:
  python scripts/tune_ml_optic_orientation.py \\
    --optic_csv AL_Optic.csv --pred_csv test_v2_scene/pred_poses.csv --frames_fps 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import compare_ml_optic as c  # noqa: E402


def eval_orientation(
    optic_csv: str,
    pred_csv: str,
    frames_fps: float,
    frame_start_index: int,
    time_offset_sec: float,
    optic_quat_order: str,
    optic_rotation_body_to_world_quat: bool,
    grid_step_deg: int,
    lever_cam: np.ndarray | None = None,
) -> dict:
    times_o, pos_o, quat_o = c.load_optic_csv(optic_csv, optic_quat_order)
    frames, pos_ml, quat_ml = c.load_pred_csv(pred_csv)
    order = np.argsort(frames)
    frames, pos_ml, quat_ml = frames[order], pos_ml[order], quat_ml[order]

    t_frames = (frames - frame_start_index) / frames_fps + time_offset_sec
    pos_o_i = c.interp_position(times_o, pos_o, t_frames)
    quat_o_i = c.slerp_batch(times_o, quat_o, t_frames)

    s, R_align, t_align = c.umeyama(pos_ml, pos_o_i)
    n = len(frames)
    R_wc_all = np.stack([c.q_to_R_wxyz(quat_ml[i]) for i in range(n)], axis=0)
    lc = np.zeros(3, dtype=float) if lever_cam is None else np.asarray(lever_cam, dtype=float)

    R_opt_all = np.zeros((n, 3, 3), dtype=float)
    for i in range(n):
        R_o = c.q_to_R_wxyz(quat_o_i[i])
        R_opt_all[i] = R_o if optic_rotation_body_to_world_quat else R_o.T

    A_stack = np.einsum("ij,njk->nik", R_align, np.transpose(R_wc_all, (0, 2, 1)))
    R_sc, euler_best, mean_grid_obj = c.grid_search_sensor_to_camera(
        A_stack, R_opt_all, grid_step_deg
    )

    rot_err = np.zeros(n, dtype=float)
    for i in range(n):
        R_pred = A_stack[i] @ R_sc
        rot_err[i] = c.quat_angle_deg(c.R_to_quat_wxyz(R_pred), c.R_to_quat_wxyz(R_opt_all[i]))

    return {
        "mean_deg": float(rot_err.mean()),
        "max_deg": float(rot_err.max()),
        "median_deg": float(np.median(rot_err)),
        "mean_grid_objective_deg": float(mean_grid_obj),
        "euler_zyx_best_deg": list(euler_best),
        "R_sensor_to_camera": R_sc.tolist(),
        "scale": float(s),
        "R_align": R_align.tolist(),
        "t_align": t_align.tolist(),
        "num_frames": int(n),
        "time_offset_sec": float(time_offset_sec),
        "optic_quat_order": optic_quat_order,
        "optic_rotation_body_to_world_quat": bool(optic_rotation_body_to_world_quat),
        "grid_step_deg": int(grid_step_deg),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--optic_csv", type=str, default=str(ROOT / "AL_Optic.csv"))
    ap.add_argument("--pred_csv", type=str, default=str(ROOT / "test_v2_scene" / "pred_poses.csv"))
    ap.add_argument("--frames_fps", type=float, default=30.0, help="FPS извлечения кадров для test_v2.mp4")
    ap.add_argument("--frame_start_index", type=int, default=1)
    ap.add_argument("--export_json", type=str, default=str(ROOT / "outputs" / "tuned_ml_optic_orientation.json"))
    args = ap.parse_args()

    for path, label in [(args.optic_csv, "optic_csv"), (args.pred_csv, "pred_csv")]:
        p = Path(path)
        if not p.is_file():
            print(f"Нет файла {label}: {path}", file=sys.stderr)
            sys.exit(1)

    Path(args.export_json).parent.mkdir(parents=True, exist_ok=True)

    orders = ["wxyz", "xyzw"]
    bodies = [False, True]
    offsets_coarse = np.arange(-0.40, 0.401, 0.05, dtype=float)

    print("Фаза 1: сетка 15° по Euler, time_offset с шагом 0.05 с, все порядки кватерниона …")
    best = None
    best_mean = float("inf")
    for order in orders:
        for body in bodies:
            for off in offsets_coarse:
                r = eval_orientation(
                    args.optic_csv,
                    args.pred_csv,
                    args.frames_fps,
                    args.frame_start_index,
                    float(off),
                    order,
                    body,
                    15,
                )
                if r["mean_deg"] < best_mean:
                    best_mean = r["mean_deg"]
                    best = r
                    print(
                        f"  лучше: mean={r['mean_deg']:.4f}° max={r['max_deg']:.4f}° "
                        f"order={order} body_to_world={body} off={off:.3f} euler={r['euler_zyx_best_deg']}"
                    )

    assert best is not None
    b_off = best["time_offset_sec"]
    print(f"\nФаза 2: уточнение time_offset около {b_off:.3f} с (шаг 0.02), сетка 15° …")
    for off in np.arange(b_off - 0.12, b_off + 0.121, 0.02, dtype=float):
        r = eval_orientation(
            args.optic_csv,
            args.pred_csv,
            args.frames_fps,
            args.frame_start_index,
            float(off),
            best["optic_quat_order"],
            best["optic_rotation_body_to_world_quat"],
            15,
        )
        if r["mean_deg"] < best_mean:
            best_mean = r["mean_deg"]
            best = r
            print(
                f"  лучше: mean={r['mean_deg']:.4f}° off={off:.3f} euler={r['euler_zyx_best_deg']}"
            )

    print("\nФаза 3: полная сетка Euler с шагом 5° при лучшем offset …")
    r5 = eval_orientation(
        args.optic_csv,
        args.pred_csv,
        args.frames_fps,
        args.frame_start_index,
        best["time_offset_sec"],
        best["optic_quat_order"],
        best["optic_rotation_body_to_world_quat"],
        5,
    )
    if r5["mean_deg"] < best_mean:
        best = r5
        best_mean = r5["mean_deg"]
    print(
        f"  после 5°: mean={r5['mean_deg']:.4f}° max={r5['max_deg']:.4f}° "
        f"euler={r5['euler_zyx_best_deg']}"
    )

    out = {
        "video_note": "Предполагается pred_poses из того же FPS, что и test_v2.mp4 (здесь 30).",
        "best": best,
    }
    with open(args.export_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== Итог (минимум средней угловой ошибки, °) ===")
    print(f"  mean={best['mean_deg']:.4f}  max={best['max_deg']:.4f}  median={best['median_deg']:.4f}")
    print(f"  optic_quat_order={best['optic_quat_order']}")
    print(f"  optic_rotation_body_to_world_quat={best['optic_rotation_body_to_world_quat']}")
    print(f"  time_offset_sec={best['time_offset_sec']}")
    print(f"  euler_zyx_best_deg (rx,ry,rz)={best['euler_zyx_best_deg']}")
    print(f"  grid_step_deg (последний проход)={best['grid_step_deg']}")
    print(f"\nКоманда compare_ml_optic (без повторного grid — подставьте matrix из JSON):")
    print(
        "  Используйте --grid_search_sensor_deg 5 с тем же offset/order/flag "
        "или --sensor_to_camera_rowmajor с nine значениями из best.R_sensor_to_camera."
    )
    print(f"\nJSON: {args.export_json}")


if __name__ == "__main__":
    main()
