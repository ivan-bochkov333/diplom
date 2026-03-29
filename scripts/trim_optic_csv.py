#!/usr/bin/env python3
"""
Обрезка AL_Optic.csv по времени в секундах (Timestamp), чтобы совпасть с обрезанным видео.

Типичный сценарий: вырезали test_v2_cuted.mp4 из начала исходника → в оптике оставляем
тот же отрезок по времени, что и длительность нового файла, начиная с --optic_time_start.

Примеры:
  python scripts/trim_optic_csv.py -i AL_Optic.csv -o AL_Optic_cuted.csv --match_video test_v2_cuted.mp4
  python scripts/trim_optic_csv.py -i AL_Optic.csv -o AL_Optic_slice.csv --t_min 5.0 --t_max 35.0
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys


def ffprobe_duration_sec(path: str) -> float:
    if not os.path.isfile(path):
        print(f"Нет файла видео: {path}", file=sys.stderr)
        sys.exit(1)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    try:
        return float(out)
    except ValueError:
        print(f"Не удалось прочитать длительность: {out!r}", file=sys.stderr)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="Обрезка CSV оптики по Timestamp")
    ap.add_argument("-i", "--input", required=True, help="Входной CSV (AL_Optic.csv)")
    ap.add_argument("-o", "--output", required=True, help="Выходной CSV")
    ap.add_argument("--t_min", type=float, default=None, help="Минимальный Timestamp (сек)")
    ap.add_argument("--t_max", type=float, default=None, help="Максимальный Timestamp (сек)")
    ap.add_argument(
        "--match_video",
        type=str,
        default=None,
        help="Взять длительность из видео (ffprobe); интервал [optic_time_start, start+duration)",
    )
    ap.add_argument(
        "--optic_time_start",
        type=float,
        default=0.0,
        help="Начало отрезка на шкале Timestamp оптики (сек), если видео вырезано не с нуля исходника",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"Нет файла: {args.input}", file=sys.stderr)
        sys.exit(1)

    t_min = args.t_min
    t_max = args.t_max
    if args.match_video is not None:
        d = ffprobe_duration_sec(args.match_video)
        t_min = args.optic_time_start
        t_max = args.optic_time_start + d
        print(f"Видео {args.match_video}: длительность {d:.6f} с")
        print(f"Оптика: Timestamp в [{t_min:.6f}, {t_max:.6f})")

    if t_min is None or t_max is None:
        print("Задайте --t_min и --t_max или --match_video", file=sys.stderr)
        sys.exit(1)
    if t_max <= t_min:
        print("t_max должен быть больше t_min", file=sys.stderr)
        sys.exit(1)

    kept = 0
    total = 0
    with open(args.input, newline="", encoding="utf-8") as fin, open(
        args.output, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        if not fieldnames or "Timestamp" not in fieldnames:
            print("В CSV нужна колонка Timestamp", file=sys.stderr)
            sys.exit(1)
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            total += 1
            try:
                ts = float(row["Timestamp"])
            except (KeyError, ValueError):
                continue
            if t_min <= ts < t_max:
                writer.writerow(row)
                kept += 1

    print(f"Строк прочитано: {total}, сохранено: {kept} → {args.output}")


if __name__ == "__main__":
    main()
