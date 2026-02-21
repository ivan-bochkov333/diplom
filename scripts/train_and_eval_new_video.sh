#!/usr/bin/env bash
# Обучить модель на new_video_train и провалидировать на new_video_test.
# Требуется: COLMAP и ffmpeg установлены, видео new_video_train.mp4 и new_video_test.mp4 в корне проекта.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. Проверка датасетов ==="
if [ ! -f new_video_train/poses.csv ]; then
  echo "Запускаю COLMAP для train-видео..."
  bash scripts/make_dataset.sh new_video_train.mp4 new_video_train
else
  echo "new_video_train: poses.csv найден ($(wc -l < new_video_train/poses.csv) строк)"
fi

if [ ! -f new_video_test/poses.csv ]; then
  echo "Запускаю COLMAP для test-видео..."
  bash scripts/make_dataset.sh new_video_test.mp4 new_video_test
else
  echo "new_video_test: poses.csv найден ($(wc -l < new_video_test/poses.csv) строк)"
fi

echo ""
echo "=== 2. Обучение на new_video_train ==="
source env/bin/activate
python train.py --config configs/new_video_train.yaml

echo ""
echo "=== 3. Валидация на new_video_test ==="
python evaluate.py \
  --checkpoint outputs/posenet/best.pth \
  --data_root new_video_test \
  --data_type colmap

echo ""
echo "Готово. Метрики и графики — в выводе выше и в outputs/posenet/ (если указано в evaluate.py)."
