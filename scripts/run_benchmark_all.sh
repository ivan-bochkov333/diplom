#!/usr/bin/env bash
# Обучает все 4 архитектуры на new_video_train и запускает бенчмарк с обновлением README.
# Использование: ./scripts/run_benchmark_all.sh [--skip-train]
#   --skip-train  не обучать, только бенчмарк (если чекпоинты уже есть)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

SKIP_TRAIN=false
for arg in "$@"; do
  if [ "$arg" = "--skip-train" ]; then
    SKIP_TRAIN=true
    break
  fi
done

if [ "$SKIP_TRAIN" = false ]; then
  echo "=== Training PoseNet+ ==="
  python train.py --config configs/new_video_train.yaml

  echo "=== Training AtLoc ==="
  python train.py --config configs/new_video_train_atloc.yaml

  echo "=== Training TransPoseNet ==="
  python train.py --config configs/new_video_train_transposenet.yaml

  echo "=== Training MS-Transformer ==="
  python train.py --config configs/new_video_train_ms_transformer.yaml
fi

echo "=== Benchmark (accuracy + speed) and update README ==="
python benchmark.py --outputs_dir outputs --data_root new_video_test --data_type colmap \
  --measure_speed --update_readme README.md

echo "Done. README.md updated with benchmark table."
