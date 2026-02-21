#!/usr/bin/env bash
# Полный пайплайн: COLMAP (train + test) → обучение → оценка на test.
# Использование: bash scripts/run_full_pipeline.sh
# Опционально: bash scripts/run_full_pipeline.sh 2  — взять 2 кадра/сек (ещё быстрее COLMAP).

set -euo pipefail
cd "$(dirname "$0")/.."
FPS="${1:-5}"

echo "========== 1. COLMAP: new_video_train ($FPS кадр/сек) =========="
if [ ! -f new_video_train/poses.csv ]; then
  bash scripts/make_dataset.sh new_video_train.mp4 new_video_train "$FPS"
else
  echo "Пропуск: new_video_train/poses.csv уже есть"
fi

echo ""
echo "========== 2. COLMAP: new_video_test ($FPS кадр/сек) =========="
if [ ! -f new_video_test/poses.csv ]; then
  bash scripts/make_dataset.sh new_video_test.mp4 new_video_test "$FPS"
else
  echo "Пропуск: new_video_test/poses.csv уже есть"
fi

echo ""
echo "========== 3. Обучение на new_video_train =========="
source env/bin/activate
python train.py --config configs/new_video_train.yaml

echo ""
echo "========== 4. Оценка на new_video_test =========="
python evaluate.py \
  --checkpoint outputs/posenet/best.pth \
  --data_root new_video_test \
  --data_type colmap

echo ""
echo "========== Готово =========="
