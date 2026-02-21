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

# 5. COLMAP: matching (CPU)
echo "===> COLMAP exhaustive_matcher..."
colmap exhaustive_matcher \
  --database_path "$DB_PATH"

# 6. COLMAP: sparse reconstruction (mapper)
echo "===> COLMAP mapper..."
colmap mapper \
  --database_path "$DB_PATH" \
  --image_path "$IMAGES_DIR" \
  --output_path "$SPARSE_DIR"

MODEL_DIR="$SPARSE_DIR/0"

if [ ! -d "$MODEL_DIR" ]; then
  echo "Ошибка: не найдено $MODEL_DIR. Проверь вывод colmap mapper."
  exit 1
fi

# 7. Конвертация модели в TXT
echo "===> COLMAP model_converter -> TXT..."
colmap model_converter \
  --input_path "$MODEL_DIR" \
  --output_path "$MODEL_DIR" \
  --output_type TXT

# 8. Извлекаем позы в CSV
echo "===> Извлекаю позы в CSV..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$SCRIPT_DIR/extract_colmap_poses.py" "$MODEL_DIR/images.txt" "$PROJECT/poses.csv"

echo
echo "Готово!"
echo "Кадры:   $IMAGES_DIR"
echo "Модель:  $MODEL_DIR"
echo "Позиции: $PROJECT/poses.csv"
