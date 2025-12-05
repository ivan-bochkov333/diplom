import csv
import os
import math

POSES_PATH = "new_dataset/poses.csv"
IMAGES_DIR = "new_dataset/images"

def main():
    rows = []
    with open(POSES_PATH, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"Всего поз в CSV: {len(rows)}")

    # 1) Проверим, что для каждого frame есть картинка
    missing_imgs = []
    for r in rows:
        frame = r["frame"]
        # наши кадры 000000.jpg, 000001.jpg и т.д.
        fname = f"{frame}.jpg"
        path = os.path.join(IMAGES_DIR, fname)
        if not os.path.exists(path):
            missing_imgs.append(fname)

    if missing_imgs:
        print(f"⚠ Нет файлов для {len(missing_imgs)} поз (первые 5): {missing_imgs[:5]}")
    else:
        print("✅ У всех поз есть соответствующий .jpg кадр")

    # 2) Проверим нормы кватернионов
    bad_quats = 0
    for r in rows:
        qw = float(r["qw"]); qx = float(r["qx"])
        qy = float(r["qy"]); qz = float(r["qz"])
        norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        if abs(norm - 1.0) > 1e-3:
            bad_quats += 1

    if bad_quats == 0:
        print("✅ Все кватернионы нормализованы (‖q‖≈1)")
    else:
        print(f"⚠ {bad_quats} кватернионов с ненормальной нормой")

    # 3) Посмотрим первые пару строк
    print("\nПервые 3 строки:")
    for r in rows[:3]:
        print(r)

    # 4) Оценим плавность траектории (разность соседних поз)
    import numpy as np

    C_prev = None
    diffs = []
    for r in rows:
        C = np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])])
        if C_prev is not None:
            diffs.append(np.linalg.norm(C - C_prev))
        C_prev = C

    if diffs:
        diffs = np.array(diffs)
        print("\nСтатистика по шагам траектории (расстояние между соседними позами):")
        print(f"  min = {diffs.min():.4f}, max = {diffs.max():.4f}, mean = {diffs.mean():.4f}")
    else:
        print("Мало поз для оценки траектории")

if __name__ == "__main__":
    main()
