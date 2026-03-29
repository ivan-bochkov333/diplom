#!/usr/bin/env python3
"""
Анализ резких поворотов по оптическому CSV: |ω| из колонок ang_vel_*.

Помогает понять, где на временной шкале «пики», и при необходимости:
  - выгрузить CSV без отсчётов с высокой угловой скоростью;
  - сохранить JSON с интервалами «плохого» времени для нарезки видео (ffmpeg).

Частота оптики в данных часто 250 Гц — порог подбирайте по отчёту (--report).

Примеры:
  python scripts/optic_angular_peaks.py -i AL_Optic.csv --report --top_peaks 15
  python scripts/optic_angular_peaks.py -i AL_Optic.csv -o AL_Optic_low_motion.csv --threshold 0.02
  python scripts/optic_angular_peaks.py -i AL_Optic.csv --threshold_quantile 0.995 -o AL_Optic_filt.csv \\
      --bad_intervals_json bad_motion.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys


def load_rows(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fn = reader.fieldnames or []
    return rows, fn


def omega_mag(row: dict) -> float | None:
    try:
        wx = float(row["ang_vel_x_Optic"])
        wy = float(row["ang_vel_y_Optic"])
        wz = float(row["ang_vel_z_Optic"])
    except (KeyError, ValueError):
        return None
    return math.sqrt(wx * wx + wy * wy + wz * wz)


def ts(row: dict) -> float | None:
    try:
        return float(row["Timestamp"])
    except (KeyError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description="Пики угловой скорости в AL_Optic.csv")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", default=None, help="CSV без строк с |ω| > threshold")
    ap.add_argument("--report", action="store_true", help="Статистика и топ пиков")
    ap.add_argument("--top_peaks", type=int, default=20)
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Порог |ω| (в тех же единицах, что в CSV; см. --report)",
    )
    ap.add_argument(
        "--threshold_quantile",
        type=float,
        default=None,
        help="Порог как квантиль по |ω| (например 0.995 — отрезать верхние 0.5%%)",
    )
    ap.add_argument(
        "--bad_intervals_json",
        type=str,
        default=None,
        help="Записать интервалы времени, где |ω| > threshold: [[t0,t1], ...]",
    )
    ap.add_argument(
        "--merge_gap_sec",
        type=float,
        default=0.05,
        help="Слияние соседних «плохих» отсчётов в один интервал (сек)",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"Нет файла: {args.input}", file=sys.stderr)
        sys.exit(1)

    rows, fieldnames = load_rows(args.input)
    omegas = []
    valid_idx = []
    for i, row in enumerate(rows):
        w = omega_mag(row)
        t = ts(row)
        if w is None or t is None:
            continue
        omegas.append(w)
        valid_idx.append(i)

    if not omegas:
        print("Нет валидных ang_vel_* / Timestamp", file=sys.stderr)
        sys.exit(1)

    arr = sorted(omegas)
    n = len(arr)

    def quantile(q: float) -> float:
        if not 0 <= q <= 1:
            raise ValueError("quantile in [0,1]")
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return arr[lo]
        return arr[lo] * (hi - pos) + arr[hi] * (pos - lo)

    if args.report:
        print(f"Отсчётов с |ω|: {n}")
        print(f"|ω| min/mean/max: {arr[0]:.8g} / {sum(arr)/n:.8g} / {arr[-1]:.8g}")
        for q in (0.5, 0.9, 0.95, 0.99, 0.995, 0.999):
            print(f"  квантиль {q:.3f}: {quantile(q):.8g}")

        indexed = sorted(
            [(omega_mag(rows[i]), ts(rows[i]), i) for i in valid_idx],
            key=lambda x: -x[0],
        )
        print(f"\nТоп-{args.top_peaks} по |ω| (Timestamp с, |ω|):")
        for w, t, _ in indexed[: args.top_peaks]:
            print(f"  t={t:.6f}  |ω|={w:.8g}")

    thresh = args.threshold
    if args.threshold_quantile is not None:
        thresh = quantile(args.threshold_quantile)
        print(f"\nПорог из квантиля {args.threshold_quantile}: {thresh:.8g}")

    if thresh is None and args.output:
        print("Для -o нужен --threshold или --threshold_quantile", file=sys.stderr)
        sys.exit(1)

    if thresh is not None and args.bad_intervals_json:
        bad_times = []
        for i in valid_idx:
            w = omega_mag(rows[i])
            t = ts(rows[i])
            if w is not None and t is not None and w > thresh:
                bad_times.append(t)
        bad_times.sort()
        intervals = []
        if bad_times:
            t0 = bad_times[0]
            t1 = bad_times[0]
            for t in bad_times[1:]:
                if t - t1 <= args.merge_gap_sec:
                    t1 = t
                else:
                    intervals.append([t0, t1])
                    t0 = t
                    t1 = t
            intervals.append([t0, t1])
        with open(args.bad_intervals_json, "w", encoding="utf-8") as f:
            json.dump(intervals, f, indent=2)
        print(f"Интервалов с |ω|>{thresh:.8g}: {len(intervals)} → {args.bad_intervals_json}")

    if args.output and thresh is not None:
        kept = 0
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                w = omega_mag(row)
                if w is None:
                    continue
                if w <= thresh:
                    writer.writerow(row)
                    kept += 1
        print(f"Сохранено строк (|ω| ≤ {thresh:.8g}): {kept} → {args.output}")

    if not args.report and thresh is None:
        print("Укажите --report и/или порог + -o / --bad_intervals_json", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
