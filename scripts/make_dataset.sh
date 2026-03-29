#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Использование: $0 video_file [project_name] [fps]"
  echo "  fps — опционально: брать N кадров в секунду (быстрее COLMAP). Без fps — все кадры."
  exit 1
fi

VIDEO="$1"
PROJECT="${2:-colmap_project}"
FPS="${3:-}"

# 1. Проверки
if ! command -v colmap >/dev/null 2>&1; then
  echo "Ошибка: colmap не найден. Установи: brew install colmap"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Ошибка: ffmpeg не найден. Установи: brew install ffmpeg"
  exit 1
fi

echo "Видео:       $VIDEO"
echo "Проект:      $PROJECT"
echo "Рабочая папка: $PROJECT"
echo

# 2. Структура проекта
mkdir -p "$PROJECT"
mkdir -p "$PROJECT/images"
mkdir -p "$PROJECT/sparse"

DB_PATH="$PROJECT/database.db"
IMAGES_DIR="$PROJECT/images"
SPARSE_DIR="$PROJECT/sparse"

# 2.1. Полная очистка старых данных (ВАЖНО!)
rm -f "$DB_PATH"
rm -rf "$SPARSE_DIR"
mkdir -p "$SPARSE_DIR"

# 3. Видео -> кадры
echo "===> Извлекаю кадры из видео..."
rm -f "$IMAGES_DIR"/*.jpg
if [ -n "$FPS" ]; then
  echo "  (режим: ${FPS} кадр/сек)"
  ffmpeg -y -i "$VIDEO" -vf "fps=$FPS" "$IMAGES_DIR/%06d.jpg"
else
  ffmpeg -y -i "$VIDEO" "$IMAGES_DIR/%06d.jpg"
fi

# 4. COLMAP: feature extraction (CPU)
echo "===> COLMAP feature_extractor..."
colmap feature_extractor \
  --database_path "$DB_PATH" \
  --image_path "$IMAGES_DIR" \
  --ImageReader.single_camera 1

# 5. COLMAP: matching
MATCHER="${COLMAP_MATCHER:-exhaustive}"
if [ "$MATCHER" = "sequential" ]; then
  echo "===> COLMAP sequential_matcher (для видео)..."
  colmap sequential_matcher \
    --database_path "$DB_PATH" \
    --SequentialMatching.overlap 15
else
  echo "===> COLMAP exhaustive_matcher..."
  colmap exhaustive_matcher \
    --database_path "$DB_PATH"
fi

# 6. COLMAP: sparse reconstruction (mapper)
echo "===> COLMAP mapper..."
colmap mapper \
  --database_path "$DB_PATH" \
  --image_path "$IMAGES_DIR" \
  --output_path "$SPARSE_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Сначала TXT для всех подмоделей (иначе нет images.txt для подсчёта кадров)
echo "===> COLMAP model_converter -> TXT (все реконструкции)..."
for d in "$SPARSE_DIR"/*/; do
  [ -f "${d}images.bin" ] || continue
  colmap model_converter \
    --input_path "$d" \
    --output_path "$d" \
    --output_type TXT
done

MODEL_DIR="$(python3 "$SCRIPT_DIR/pick_largest_sparse_model.py" "$SPARSE_DIR")"
if [ -z "$MODEL_DIR" ] || [ ! -d "$MODEL_DIR" ]; then
  echo "Ошибка: нет реконструкции в $SPARSE_DIR. Проверь вывод colmap mapper."
  exit 1
fi
echo "===> Используем модель с наибольшим числом кадров: $MODEL_DIR"

# 8. Извлекаем позы в CSV
echo "===> Извлекаю позы в CSV..."
python3 "$SCRIPT_DIR/extract_colmap_poses.py" "$MODEL_DIR/images.txt" "$PROJECT/poses.csv"

echo
echo "Готово!"
echo "Кадры:   $IMAGES_DIR"
echo "Модель:  $MODEL_DIR"
echo "Позиции: $PROJECT/poses.csv"
