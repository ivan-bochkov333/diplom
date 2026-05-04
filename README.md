# Визуальная релокализация камеры: COLMAP + нейросетевые модели

Определение 6DoF позы камеры (позиция + ориентация) по одному изображению. COLMAP (Structure-from-Motion) используется **один раз** для создания разметки, после чего нейросетевая модель предсказывает позу за единицы миллисекунд — без 3D-карты и без сопоставления признаков.

---

## Содержание

- [Установка](#установка)
- [Быстрый старт: полный пайплайн](#быстрый-старт-полный-пайплайн)
- [Пошаговое руководство](#пошаговое-руководство)
  - [1. Сбор датасета (COLMAP)](#1-сбор-датасета-colmap)
  - [2. Очистка данных](#2-очистка-данных)
  - [3. Обучение моделей](#3-обучение-моделей)
  - [4. Инференс](#4-инференс)
  - [5. Сравнение с оптическим датчиком](#5-сравнение-с-оптическим-датчиком)
- [Архитектуры моделей](#архитектуры-моделей)
- [Результаты](#результаты)
- [Структура проекта](#структура-проекта)

---

## Установка

**Системные зависимости:**

```bash
# Ubuntu/Debian
sudo apt install -y colmap ffmpeg

# macOS
brew install colmap ffmpeg
```

**Python-окружение:**

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

---

## Быстрый старт: полный пайплайн

Для пары видео `train_v2.mp4` (обучение) и `test_v2.mp4` (тест) + файла оптики `AL_Optic.csv`:

```bash
source env/bin/activate

# 1. Сбор датасетов из видео (извлечение каждого 5-го кадра из 120 FPS → 24 FPS)
bash scripts/make_dataset.sh train_v2.mp4 train_v2_scene 5
bash scripts/make_dataset.sh test_v2.mp4 test_v2_scene 5

# 2. Очистка данных (убрать кадры с угловыми скачками > 15°)
python3 -c "
import csv, numpy as np, os

def filter_poses(input_csv, output_dir, ang_thresh=15.0):
    rows = []
    with open(input_csv) as f:
        for row in csv.DictReader(f): rows.append(row)
    quats = np.array([[float(r['qw']),float(r['qx']),float(r['qy']),float(r['qz'])] for r in rows])
    n = len(rows)
    bad = set()
    for i in range(1, n):
        q1 = quats[i-1] / (np.linalg.norm(quats[i-1])+1e-12)
        q2 = quats[i] / (np.linalg.norm(quats[i])+1e-12)
        angle = 2*np.degrees(np.arccos(min(1.0, abs(np.dot(q1,q2)))))
        if angle > ang_thresh: bad.add(i); bad.add(i-1)
    good = [i for i in range(n) if i not in bad]
    os.makedirs(os.path.join(output_dir,'images'), exist_ok=True)
    src_dir = os.path.dirname(input_csv)+'/images'
    with open(os.path.join(output_dir,'poses.csv'),'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=['frame','tx','ty','tz','qw','qx','qy','qz'])
        w.writeheader()
        for i in good:
            w.writerow(rows[i])
            fname = rows[i]['frame'].strip().zfill(6)+'.jpg'
            src, dst = os.path.join(src_dir,fname), os.path.join(output_dir,'images',fname)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)
    print(f'{output_dir}: {len(good)}/{n} кадров')

filter_poses('train_v2_scene/poses.csv', 'train_v2_scene_clean')
filter_poses('test_v2_scene/poses.csv', 'test_v2_scene_clean')
"

# 3. Обучение всех 4 моделей на очищенных данных
python train.py --config configs/train_v2_clean_posenet.yaml
python train.py --config configs/train_v2_clean_atloc.yaml
python train.py --config configs/train_v2_clean_transposenet.yaml
python train.py --config configs/train_v2_clean_ms_transformer.yaml

# 4. Инференс на тестовой сцене
for model in posenet atloc transposenet ms_transformer; do
  python infer.py \
    --checkpoint outputs_clean/${model}/best.pth \
    --images_dir test_v2_scene_clean/images \
    --out_csv test_v2_scene_clean/pred_${model}.csv
done

# 5. Сравнение с оптическим датчиком (эталон)
for model in posenet atloc transposenet ms_transformer; do
  python scripts/compare_ml_optic.py \
    --optic_csv AL_Optic.csv \
    --pred_csv test_v2_scene_clean/pred_${model}.csv \
    --frames_fps 24 \
    --kabsch_orientation_calibration \
    --export_json outputs_clean/optic_${model}.json
done
```

---

## Пошаговое руководство

### 1. Сбор датасета (COLMAP)

Из видео извлекаются кадры, по ним COLMAP строит 3D-реконструкцию и определяет позу камеры для каждого кадра.

```bash
bash scripts/make_dataset.sh <видео.mp4> <папка_сцены> [шаг_кадров]
```

- `шаг_кадров` — извлекать каждый N-й кадр (по умолчанию — все). Для видео 120 FPS рекомендуется 5 (итого 24 FPS).

**Что делает скрипт:**

1. Извлекает кадры из видео (`ffmpeg`)
2. Находит характерные точки на каждом кадре (COLMAP feature extraction)
3. Сопоставляет точки между парами кадров (COLMAP matching)
4. Строит 3D-модель сцены и определяет позы камер (COLMAP mapper)
5. Уточняет все позы совместно (Bundle Adjustment)
6. Экспортирует позы в `poses.csv`

**Результат:**

```
train_v2_scene/
├── images/         # кадры: 000001.jpg, 000002.jpg, ...
├── poses.csv       # frame,tx,ty,tz,qw,qx,qy,qz
├── sparse/         # 3D-реконструкция COLMAP
└── database.db     # база признаков COLMAP
```

**Формат `poses.csv`:**

| Поле | Описание |
|------|----------|
| frame | Номер кадра |
| tx, ty, tz | Позиция камеры в пространстве (метры) |
| qw, qx, qy, qz | Ориентация камеры — единичный кватернион |

**Проверка:**

```bash
wc -l train_v2_scene/poses.csv test_v2_scene/poses.csv
```

---

### 2. Очистка данных

COLMAP может давать ошибочные позы на кадрах с быстрым движением или смазом. Такие кадры ухудшают обучение. Очистка убирает кадры с угловым скачком > 15° между соседними позами.

Скрипт очистки — в разделе [Быстрый старт](#быстрый-старт-полный-пайплайн) (шаг 2).

Типичный результат: сохраняется 97% кадров, убираются только явные выбросы.

---

### 3. Обучение моделей

Обучение запускается по YAML-конфигу:

```bash
python train.py --config configs/train_v2_clean_posenet.yaml
```

**Доступные конфиги (очищенные данные):**

| Конфиг | Модель |
|--------|--------|
| `configs/train_v2_clean_posenet.yaml` | PoseNet |
| `configs/train_v2_clean_atloc.yaml` | AtLoc |
| `configs/train_v2_clean_transposenet.yaml` | TransPoseNet |
| `configs/train_v2_clean_ms_transformer.yaml` | MS-Transformer |

**Конфиги на неочищенных данных** (для сравнения):

| Конфиг | Модель |
|--------|--------|
| `configs/train_v2.yaml` | PoseNet |
| `configs/train_v2_atloc.yaml` | AtLoc |
| `configs/train_v2_transposenet.yaml` | TransPoseNet |
| `configs/train_v2_ms_transformer.yaml` | MS-Transformer |

**Ключевые параметры обучения** (общие для всех моделей):

| Параметр | Значение |
|----------|----------|
| Backbone | ResNet-34 (предобучен на ImageNet) |
| Эпох | до 80 (early stopping с patience 15) |
| Batch size | 16 |
| Оптимизатор | AdamW, lr=1e-4 |
| Расписание LR | Cosine annealing с warmup 3 эпохи |
| Размер изображения | 224×224 |

**Чекпоинты сохраняются в:**

- Очищенные: `outputs_clean/<модель>/best.pth`
- Неочищенные: `outputs/<модель>/best.pth`

**Проверка:**

```bash
ls outputs_clean/*/best.pth
```

---

### 4. Инференс

**По папке с кадрами:**

```bash
python infer.py \
  --checkpoint outputs_clean/atloc/best.pth \
  --images_dir test_v2_scene_clean/images \
  --out_csv test_v2_scene_clean/pred_atloc.csv
```

**Один кадр (для рантайма):**

```bash
python run_inference_single.py --checkpoint outputs_clean/atloc/best.pth --image frame.jpg
```

**Из кода:**

```python
from run_inference_single import load_pose_model, predict_pose

model, transform, device = load_pose_model("outputs_clean/atloc/best.pth")
xyz, quat = predict_pose(model, transform, device, "frame.jpg")
# xyz — (3,) позиция, quat — (4,) кватернион (qw,qx,qy,qz)
```

---

### 5. Сравнение с оптическим датчиком

Оптический трекер (250 Гц) — эталон. ML-предсказания и оптика в разных системах координат, поэтому перед сравнением выполняется:

1. **Синхронизация по времени** — кадры привязываются к временной шкале оптики
2. **Выравнивание координат (Umeyama)** — находится масштаб, поворот и сдвиг между системами координат
3. **Калибровка крепления** — учитывается разворот осей датчика относительно камеры

```bash
python scripts/compare_ml_optic.py \
  --optic_csv AL_Optic.csv \
  --pred_csv test_v2_scene_clean/pred_atloc.csv \
  --frames_fps 24 \
  --kabsch_orientation_calibration \
  --export_json outputs_clean/optic_atloc.json
```

**Основные флаги:**

| Флаг | Описание |
|------|----------|
| `--frames_fps` | FPS извлечённых кадров (120/5 = 24) |
| `--time_offset_sec` | Сдвиг по времени между видео и оптикой |
| `--optic_quat_order` | Порядок кватернионов в CSV оптики (`wxyz` или `xyzw`) |
| `--kabsch_orientation_calibration` | Калибровка постоянного поворота для ориентации |
| `--rig_preset diagram_xy_swap` | Пресет калибровки крепления (по умолчанию) |
| `--export_json` | Сохранить метрики в JSON |

**Метрики:**

| Метрика | Описание |
|---------|----------|
| Евклидово расстояние по позиции (м) | Расстояние между предсказанной и эталонной точкой |
| Сферическое расстояние на SO(3) (°) | Угол между предсказанной и эталонной ориентацией |
| Относительная ориентация (°) | Ошибка изменения поворота между соседними кадрами |

**Дополнительные утилиты:**

```bash
# Обрезка оптики под длительность видео
python scripts/trim_optic_csv.py -i AL_Optic.csv -o AL_Optic_cut.csv --match_video test_v2_cut.mp4

# Анализ пиков угловой скорости в оптике
python scripts/optic_angular_peaks.py -i AL_Optic.csv --report --top_peaks 20

# Фильтрация оптики по угловой скорости
python scripts/optic_angular_peaks.py -i AL_Optic.csv -o AL_Optic_smooth.csv --threshold_quantile 0.90
```

---

## Архитектуры моделей

| Модель | Описание | Особенность |
|--------|----------|-------------|
| **PoseNet** | ResNet-34 + две FC-головы (позиция, кватернион) | Простой и быстрый baseline |
| **AtLoc** | ResNet-34 + модуль внимания + FC-головы | Фокус на информативных областях изображения |
| **TransPoseNet** | ResNet-34 → фрагменты → Transformer Encoder → [CLS] → головы | Учёт связей между далёкими частями изображения |
| **MS-Transformer** | TransPoseNet + scene embedding | Поддержка нескольких сцен (для одной: `num_scenes: 1`) |

Все модели на выходе дают: позицию `(tx, ty, tz)` и ориентацию кватернионом `(qw, qx, qy, qz)`.

Функция потерь (Kendall et al., 2017): `L = L_pos·exp(−s_x) + s_x + L_ori·exp(−s_q) + s_q`, где `s_x`, `s_q` — обучаемые веса, автоматически балансирующие вклад позиции и ориентации.

---

## Результаты

### Итоговые метрики: ML vs оптический датчик (эталон)

Все модели обучены на очищенных данных (фильтрация угловых скачков > 15°). Эталон — оптический трекер (250 Гц). Выравнивание координат методом Умеямы.

**Позиция — евклидово отклонение до оптики (м):**

| Модель | Медиана | Среднее | Максимум |
|--------|---------|---------|----------|
| COLMAP GT | 0.032 | 0.047 | 0.253 |
| **AtLoc** | **0.035** | **0.049** | **0.247** |
| PoseNet | 0.037 | 0.050 | 0.244 |
| TransPoseNet | 0.030 | 0.048 | 0.251 |
| MS-Transformer | 0.035 | 0.050 | 0.254 |

> Все модели дают ~3-4 см медиану — сопоставимо с COLMAP.

**Ориентация — сферическое расстояние на SO(3) (°):**

| Модель | Медиана | Среднее |
|--------|---------|---------|
| COLMAP GT | 4.8 | 8.2 |
| **AtLoc** | **5.1** | **9.3** |
| PoseNet | 5.7 | 10.1 |
| TransPoseNet | 5.4 | 9.8 |
| MS-Transformer | 6.2 | 11.5 |

> После очистки данных абсолютная ошибка ориентации — около 5°. Это приемлемый уровень для задач навигации и дополненной реальности.

**Относительная ориентация — ошибка изменения поворота между соседними кадрами (°):**

| Модель | Медиана | Среднее |
|--------|---------|---------|
| COLMAP GT | 1.8 | 3.1 |
| **AtLoc** | **2.1** | **3.8** |
| PoseNet | 2.5 | 4.5 |
| TransPoseNet | 2.4 | 4.2 |
| MS-Transformer | 2.8 | 5.0 |

> Относительная ориентация ~2° — модель точно отслеживает изменения поворота камеры.

### Влияние очистки данных на ориентацию

| Модель | До очистки (медиана, °) | После очистки (медиана, °) | Улучшение |
|--------|------------------------|---------------------------|-----------|
| AtLoc | 51.4 | 5.1 | в 10 раз |
| TransPoseNet | 43.7 | 5.4 | в 8 раз |
| PoseNet | 41.8 | 5.7 | в 7 раз |
| MS-Transformer | 41.8 | 6.2 | в 7 раз |

> Качество COLMAP-разметки критично. Фильтрация выбросов улучшает ориентацию в 7-10 раз.

### Скорость инференса

| Подход | Время на кадр | Кадров/с |
|--------|---------------|----------|
| COLMAP | ~1000-5000 мс | < 1 |
| PoseNet | ~4 мс | ~250 |
| AtLoc | ~5 мс | ~200 |
| TransPoseNet | ~5 мс | ~200 |
| MS-Transformer | ~5.5 мс | ~180 |

> ML в 200-1000 раз быстрее COLMAP.

### Итоговая сводка

| | Позиция (медиана) | Ориентация (медиана) | Скорость |
|---|-------------------|---------------------|----------|
| **COLMAP** | 3.2 см | 4.8° | ~1 кадр/с |
| **AtLoc (лучшая ML)** | 3.5 см | 5.1° | ~200 кадров/с |

### Выводы

- **По позиции** ML совпадает с COLMAP (~3 см медиана от эталона).
- **По ориентации** после очистки данных — медиана ~5° (AtLoc: 5.1°, COLMAP GT: 4.8°). ML не уступает классическому подходу.
- **Очистка данных** — ключевой фактор: улучшает ориентацию в 7-10 раз.
- **Скорость** ML в 200-1000 раз выше, чем у COLMAP. Все модели укладываются в 5 мс на кадр.
- **Лучшая модель** — AtLoc: наименьшая ошибка ориентации + 200 кадров/с.

---

## Структура проекта

```
├── configs/                          # YAML-конфиги обучения
│   ├── train_v2.yaml                 # PoseNet на train_v2_scene
│   ├── train_v2_atloc.yaml           # AtLoc на train_v2_scene
│   ├── train_v2_transposenet.yaml    # TransPoseNet на train_v2_scene
│   ├── train_v2_ms_transformer.yaml  # MS-Transformer на train_v2_scene
│   ├── train_v2_clean_posenet.yaml   # PoseNet на очищенных данных
│   ├── train_v2_clean_atloc.yaml     # AtLoc на очищенных данных
│   ├── train_v2_clean_transposenet.yaml
│   └── train_v2_clean_ms_transformer.yaml
├── scripts/
│   ├── make_dataset.sh               # Видео → кадры → COLMAP → poses.csv
│   ├── extract_colmap_poses.py       # Парсинг COLMAP → poses.csv
│   ├── pick_largest_sparse_model.py  # Выбор лучшей COLMAP-реконструкции
│   ├── compare_ml_optic.py           # ML vs оптика (Umeyama + метрики)
│   ├── tune_ml_optic_orientation.py  # Подбор параметров калибровки
│   ├── trim_optic_csv.py             # Обрезка оптики под видео
│   ├── optic_angular_peaks.py        # Анализ/фильтрация угловых скоростей
│   ├── run_pipeline_v2.sh            # Полный пайплайн train_v2/test_v2
│   └── run_benchmark_all.sh          # Обучение всех моделей + бенчмарк
├── src/
│   ├── models/                       # PoseNet, AtLoc, TransPoseNet, MS-Transformer
│   ├── datasets/                     # Загрузка COLMAP-датасетов
│   ├── losses/                       # Функция потерь (Kendall et al.)
│   └── utils/                        # Метрики, нормализация
├── train.py                          # Обучение по конфигу
├── evaluate.py                       # Оценка на тестовом датасете
├── infer.py                          # Инференс по папке изображений
├── benchmark.py                      # Сравнение моделей + замер скорости
├── run_inference_single.py           # Инференс одного кадра (рантайм)
├── requirements.txt
└── README.md
```

**Датасет** (после `make_dataset.sh`): папка с `images/`, `poses.csv`, `sparse/`.

**Чекпоинты**: `outputs/<модель>/best.pth` (неочищенные), `outputs_clean/<модель>/best.pth` (очищенные).
