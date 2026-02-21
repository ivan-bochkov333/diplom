# Визуальная релокализация камеры: от COLMAP до Transformer-моделей

## 1. Постановка задачи

**Absolute Pose Regression (APR)** — задача определения 6DoF позы камеры (положение и ориентация) по одному RGB-изображению в координатах заданной сцены.

### Формальная постановка

Дано изображение \( I \in \mathbb{R}^{H \times W \times 3} \), требуется найти позу камеры:

$$
\mathbf{p} = (\mathbf{x}, \mathbf{q}), \quad \mathbf{x} \in \mathbb{R}^3, \quad \mathbf{q} \in \mathbb{H}_1
$$

где:
- **x** = (tx, ty, tz) — позиция центра камеры в мировой системе координат;
- **q** = (qw, qx, qy, qz) — ориентация камеры как единичный кватернион (||q|| = 1).

Модель \( f_\theta \) обучается минимизировать ошибку между предсказанной и истинной позой:

$$
f_\theta(I) = (\hat{\mathbf{x}}, \hat{\mathbf{q}}) \approx (\mathbf{x}^*, \mathbf{q}^*)
$$

### Функция потерь (Learnable-weighted, Kendall et al., 2017)

$$
\mathcal{L} = \mathcal{L}_x \cdot e^{-s_x} + s_x + \mathcal{L}_q \cdot e^{-s_q} + s_q
$$

где \( s_x, s_q \) — обучаемые параметры лог-дисперсии, автоматически балансирующие вклад позиционной и ориентационной ошибок:

- \( \mathcal{L}_x = \text{MSE}(\hat{\mathbf{x}}, \mathbf{x}^*) \) — позиционная ошибка;
- \( \mathcal{L}_q = 1 - |\langle \hat{\mathbf{q}}, \mathbf{q}^* \rangle| \) — ориентационная ошибка (эквивалентна угловой).

### Метрики

| Метрика | Формула | Описание |
|---------|---------|----------|
| Позиционная ошибка | \( e_x = \|\hat{\mathbf{x}} - \mathbf{x}^*\|_2 \) | Евклидово расстояние (м) |
| Угловая ошибка | \( e_q = 2 \arccos(|\langle \hat{\mathbf{q}}, \mathbf{q}^* \rangle|) \) | Угол между кватернионами (°) |

Основные метрики — **медианные** ошибки по всему тестовому набору.

---

## 2. Обзор подходов

### 2.1 Structure-based методы
Классические подходы (COLMAP, HLoc) выполняют сопоставление 2D-точек на изображении с 3D-точками карты сцены, а затем решают задачу PnP. Высокая точность, но требуют хранения 3D-карты и медленны при инференсе.

### 2.2 Absolute Pose Regression (APR)
Нейросети напрямую регрессируют позу из изображения (end-to-end). Компактные, быстрые, но менее точные. Основные работы:

- **PoseNet** (Kendall et al., 2015) — GoogLeNet backbone, первая APR-модель;
- **PoseNet с обучаемыми весами** (Kendall et al., 2017) — learnable loss weighting;
- **AtLoc** (Wang et al., 2020) — self-attention на feature maps;
- **MS-Transformer** (Shavit et al., 2021) — multi-scene transformer с scene embeddings.

### 2.3 Двухстадийный подход (Pre-train + Fine-tune)
Pre-train модели на нескольких сценах учит общие геометрические представления. Fine-tune на целевой сцене адаптирует модель с меньшим количеством данных и более быстрой сходимостью.

---

## 3. Архитектуры моделей

### 3.1 PoseNet+ (улучшенный CNN baseline)

```
Image → ResNet34 (partial unfreeze) → AvgPool → FC(512) → BN → ReLU → Dropout
                                                             ├→ FC(3)  → xyz
                                                             └→ FC(4)  → q (normalized)
```

Улучшения над оригинальным PoseNet:
- ResNet34 вместо GoogLeNet/ResNet18;
- Частичная разморозка backbone (последние 2 блока);
- BatchNorm в FC-голове;
- Xavier-инициализация pose heads.

### 3.2 AtLoc (Attention-guided)

```
Image → ResNet34 → Self-Attention Module → AvgPool → FC → Pose Heads
```

Self-Attention модуль (non-local block):
- Query, Key, Value — свёрточные проекции feature map;
- Attention weights вычисляются как softmax(Q^T K / √d);
- Позволяет модели фокусироваться на геометрически стабильных областях;
- Learnable residual scaling (γ · attention_out + input).

### 3.3 TransPoseNet (Transformer encoder)

```
Image → ResNet34 → 1×1 Conv → Flatten to tokens → [CLS] + Positional Encoding
      → Transformer Encoder (4 layers, 8 heads) → [CLS] output → FC → Pose Heads
```

- CNN feature map (7×7) разбивается на 49 patch-токенов;
- Learnable positional encoding;
- Transformer Encoder с pre-norm, GELU activation;
- [CLS]-токен агрегирует глобальную информацию.

### 3.4 MS-Transformer (Multi-Scene)

```
Image → ResNet34 → 1×1 Conv → Flatten → [CLS] + [SCENE] + Patch tokens + PosEnc
      → Transformer Encoder → [CLS] output → FC → Pose Heads
```

Расширение TransPoseNet:
- Learnable scene embedding — вектор, идентифицирующий сцену;
- Pre-train на нескольких сценах одновременно;
- Fine-tune: сброс scene embedding и pose heads, backbone + transformer замораживаются.

---

## 4. Pipeline

### 4.1 Создание датасета из видео (COLMAP)

```
Видео → ffmpeg (извлечение кадров) → COLMAP feature_extractor
      → exhaustive_matcher → mapper → model_converter (→ TXT)
      → extract_colmap_poses.py → poses.csv
```

```bash
./scripts/make_dataset.sh input_video.mp4 my_scene
```

### 4.2 Обучение модели

```bash
# PoseNet+ на собственных данных
python train.py --config configs/posenet.yaml

# AtLoc
python train.py --config configs/atloc.yaml

# TransPoseNet
python train.py --config configs/transposenet.yaml

# MS-Transformer: pre-train на 7Scenes
python train.py --config configs/ms_transformer.yaml

# MS-Transformer: fine-tune на своей сцене
python train.py --config configs/posenet.yaml --finetune outputs/ms_transformer/best.pth
```

### 4.3 Инференс

```bash
# Предсказание поз для новых изображений
python infer.py --checkpoint outputs/posenet/best.pth --images_dir new_scene/images

# С оценкой качества (если есть ground truth)
python infer.py --checkpoint outputs/posenet/best.pth --images_dir new_scene/images --gt_csv new_scene/poses.csv --plot
```

### 4.4 Бенчмарк всех моделей

```bash
# Сравнение на собственных данных
python benchmark.py --outputs_dir outputs --data_root my_test_1_dataset

# Сравнение на 7Scenes (одна сцена)
python benchmark.py --outputs_dir outputs --data_type 7scenes_single --scene chess --seven_scenes_root data/7scenes
```

---

## 5. Данные

### 5.1 Собственные данные

Видеозаписи помещений с iPhone, обработанные через COLMAP. Каждая сцена содержит:
- `images/` — извлечённые кадры (.jpg);
- `poses.csv` — позиции и ориентации камеры (из COLMAP);
- `sparse/` — разреженная 3D-реконструкция COLMAP.

### 5.2 7Scenes (Microsoft Research)

Стандартный бенчмарк для indoor visual relocalization:

| Сцена | Train | Test | Площадь |
|-------|-------|------|---------|
| Chess | 4000 | 2000 | 3×2×1 м |
| Fire | 2000 | 2000 | 2.5×1×1 м |
| Heads | 1000 | 1000 | 2×0.5×1 м |
| Office | 6000 | 4000 | 2.5×2×1.5 м |
| Pumpkin | 4000 | 2000 | 2.5×2×1 м |
| Red Kitchen | 7000 | 5000 | 4×3×1.5 м |
| Stairs | 2000 | 1000 | 2.5×2×1.5 м |

Скачивание:
```bash
./scripts/download_7scenes.sh data/7scenes          # все сцены
./scripts/download_7scenes.sh data/7scenes chess fire # только chess и fire
```

---

## 6. Структура проекта

```
project/
├── configs/                        # YAML-конфигурации экспериментов
│   ├── base.yaml                   # Базовые настройки (наследуются всеми)
│   ├── posenet.yaml
│   ├── atloc.yaml
│   ├── transposenet.yaml
│   └── ms_transformer.yaml
├── src/
│   ├── models/
│   │   ├── backbone.py             # ResNet / EfficientNet backbones
│   │   ├── posenet.py              # PoseNet+ (improved CNN)
│   │   ├── atloc.py                # AtLoc (attention-guided)
│   │   ├── transposenet.py         # TransPoseNet (transformer encoder)
│   │   └── ms_transformer.py       # MS-Transformer (multi-scene)
│   ├── datasets/
│   │   ├── pose_dataset.py         # Базовый Dataset для поз
│   │   ├── colmap_dataset.py       # Загрузчик COLMAP-датасетов
│   │   └── seven_scenes.py         # Загрузчик 7Scenes
│   ├── losses/
│   │   └── pose_loss.py            # Learnable-weighted loss
│   └── utils/
│       ├── metrics.py              # Позиционная / угловая ошибки
│       ├── normalize.py            # Нормализация координат (multi-scene)
│       └── visualization.py        # 3D-траектории, CDF, attention maps
├── scripts/
│   ├── make_dataset.sh             # Видео → COLMAP → poses.csv
│   ├── extract_colmap_poses.py     # Парсинг COLMAP images.txt
│   ├── check_dataset.py            # Валидация датасета
│   └── download_7scenes.sh         # Скачивание 7Scenes
├── train.py                        # Единый скрипт обучения
├── evaluate.py                     # Оценка модели
├── infer.py                        # Инференс на новых данных
├── benchmark.py                    # Сравнение всех моделей
├── requirements.txt
└── README.md
```

---

## 7. Установка

```bash
# Создание виртуального окружения
python3 -m venv env
source env/bin/activate

# Установка зависимостей Python
pip install -r requirements.txt

# Системные зависимости (macOS)
brew install colmap ffmpeg
```

---

## 8. Быстрый старт

```bash
# 1. Создать датасет из видео
./scripts/make_dataset.sh my_video.mp4 my_scene

# 2. Обучить PoseNet+
python train.py --config configs/posenet.yaml --data.root=my_scene

# 3. Оценить результат
python evaluate.py --checkpoint outputs/posenet/best.pth --data_root my_scene --plot

# 4. Инференс на новом видео той же сцены
./scripts/make_dataset.sh new_video.mp4 new_recording
python infer.py --checkpoint outputs/posenet/best.pth \
    --images_dir new_recording/images \
    --gt_csv new_recording/poses.csv --plot
```

---

## 9. Ссылки

1. Kendall A., Grimes M., Cipolla R. *PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization.* ICCV 2015.
2. Kendall A., Cipolla R. *Geometric Loss Functions for Camera Pose Regression.* CVPR 2017.
3. Wang B. et al. *AtLoc: Attention Guided Camera Localization.* AAAI 2020.
4. Shavit Y., Ferens R., Keller Y. *Learning Multi-Scene Absolute Pose Regression with Transformers.* ICCV 2021.
5. Schönberger J.L., Frahm J.M. *Structure-from-Motion Revisited.* CVPR 2016 (COLMAP).
