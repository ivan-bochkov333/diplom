#!/usr/bin/env bash
# Полный пайплайн для train_v2.mp4 / test_v2.mp4 и сравнение с AL_Optic.csv
#
# Положите в корень проекта:
#   train_v2.mp4  test_v2.mp4  AL_Optic.csv
#
# По умолчанию извлекаются ВСЕ кадры видео (нативный FPS), без прореживания.
# Если нужно ускорить COLMAP за счёт меньшего числа кадров:
#   DOWNSAMPLE_FPS=5 ./scripts/run_pipeline_v2.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

TRAIN_VIDEO="${TRAIN_VIDEO:-train_v2.mp4}"
TEST_VIDEO="${TEST_VIDEO:-test_v2.mp4}"
OPTIC_CSV="${OPTIC_CSV:-AL_Optic.csv}"

# FPS ролика для синхронизации с оптикой (по умолчанию — из метаданных test_v2)
probe_video_fps() {
  local vid="$1"
  local rate
  rate="$(ffprobe -v 0 -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0 "$vid" 2>/dev/null || true)"
  if [ -z "$rate" ]; then
    echo "30"
    return
  fi
  python3 -c "from fractions import Fraction; print(float(Fraction('${rate//\'/}')))"
}

if [ -n "${DOWNSAMPLE_FPS:-}" ]; then
  FRAMES_FPS="$DOWNSAMPLE_FPS"
  FPS_ARGS=("$DOWNSAMPLE_FPS")
else
  FRAMES_FPS="$(probe_video_fps "$TEST_VIDEO")"
  FPS_ARGS=()
fi

for f in "$TRAIN_VIDEO" "$TEST_VIDEO"; do
  if [ ! -f "$f" ]; then
    echo "Ошибка: нет файла $f (положите в $ROOT)"
    exit 1
  fi
done
if [ ! -f "$OPTIC_CSV" ]; then
  echo "Предупреждение: нет $OPTIC_CSV — шаг compare будет пропущен."
fi

if [ ${#FPS_ARGS[@]} -gt 0 ]; then
  echo "=== 1/4 COLMAP + датасет train (прореживание: ${FPS_ARGS[0]} кадр/с) ==="
  "$SCRIPT_DIR/make_dataset.sh" "$TRAIN_VIDEO" train_v2_scene "${FPS_ARGS[0]}"
  echo "=== 2/4 COLMAP + датасет test (${FPS_ARGS[0]} кадр/с) ==="
  "$SCRIPT_DIR/make_dataset.sh" "$TEST_VIDEO" test_v2_scene "${FPS_ARGS[0]}"
else
  echo "=== 1/4 COLMAP + датасет train (все кадры видео) ==="
  "$SCRIPT_DIR/make_dataset.sh" "$TRAIN_VIDEO" train_v2_scene
  echo "=== 2/4 COLMAP + датасет test (все кадры) ==="
  "$SCRIPT_DIR/make_dataset.sh" "$TEST_VIDEO" test_v2_scene
fi
echo "    (для оптики: --frames_fps=$FRAMES_FPS)"

echo "=== 3/4 Обучение PoseNet+ ==="
if [ -d "$ROOT/env/bin" ]; then
  # shellcheck source=/dev/null
  source "$ROOT/env/bin/activate"
fi
python train.py --config configs/train_v2.yaml

echo "=== 4/4 Инференс на test + сравнение с оптикой ==="
python infer.py --checkpoint outputs/posenet/best.pth \
  --images_dir test_v2_scene/images \
  --out_csv test_v2_scene/pred_poses.csv

if [ -f "$OPTIC_CSV" ]; then
  python "$SCRIPT_DIR/compare_ml_optic.py" \
    --optic_csv "$OPTIC_CSV" \
    --pred_csv test_v2_scene/pred_poses.csv \
    --frames_fps "$FRAMES_FPS" \
    --kabsch_orientation_calibration
else
  echo "Сравнение с оптикой пропущено (нет CSV)."
fi

echo "Готово. Предсказания: test_v2_scene/pred_poses.csv"
