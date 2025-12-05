# Определение ориентации камеры по видео с помощью COLMAP и нейросетевой модели MiniPoseNet

Этот проект реализует полный pipeline для получения данных о положении и ориентации камеры по одному видео и обучения нейросетевой модели, способной предсказывать ориентацию камеры в новых видео.
В качестве источника ground truth используется система Structure-from-Motion COLMAP, которая восстанавливает траекторию камеры и 3D-структуру сцены.

Проект включает:
- автоматическую генерацию датасета из видео (make_dataset.sh);
- преобразование COLMAP-модели в обучающий датасет (extract_colmap_poses.py);
- обучение нейросети (train_posenet.py);
- инференс и сравнение с ground truth (infer_posenet.py).

## Структура проекта
project/
├── make_dataset.sh
├── extract_colmap_poses.py
├── train_posenet.py
├── infer_posenet.py
├── my_test_1_dataset/
│   ├── images/
│   ├── poses.csv
│   └── posenet.pth
└── new_dataset/

## Установка зависимостей
pip install torch torchvision numpy pillow
brew install colmap ffmpeg

## 1. Создание датасета
./make_dataset.sh input_video.mp4 my_dataset

## 2. Обучение модели
python train_posenet.py

## 3. Инференс модели
python infer_posenet.py --data_root new_dataset --checkpoint my_test_1_dataset/posenet.pth --evaluate
