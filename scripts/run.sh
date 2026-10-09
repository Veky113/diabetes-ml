#!/bin/bash

#bash скрипт для запуска обучения модели, принимает 2 параметра
#1 путь к данным по умолчанию data/dataset_diabetes/diabetic_data.csv
#2 количество деревьев по умолчанию 100

DATA_PATH="${1:-data/dataset_diabetes/diabetic_data.csv}"
N_ESTIMATORS="${2:-100}"

echo "запуск обучения модели"
echo "путь к данным $DATA_PATH"
echo "количество деревьев $N_ESTIMATORS"

#запуск скрипта через uv
uv run python src/main.py --data_path "$DATA_PATH" --n_estimators "$N_ESTIMATORS"

echo "обучение завершено"
echo "результаты и модель сохранены в папке results"