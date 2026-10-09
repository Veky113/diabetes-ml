#!/bin/bash

# Bash-скрипт для запуска обучения модели
# Принимает 2 параметра:
# 1. Путь к данным (по умолчанию: data/dataset_diabetes/diabetic_data.csv)
# 2. Количество деревьев (по умолчанию: 100)

DATA_PATH="${1:-data/dataset_diabetes/diabetic_data.csv}"
N_ESTIMATORS="${2:-100}"

echo "=== Запуск обучения модели ==="
echo "Путь к данным: $DATA_PATH"
echo "Количество деревьев: $N_ESTIMATORS"

# Запуск Python-скрипта через uv
uv run python src/main.py --data_path "$DATA_PATH" --n_estimators "$N_ESTIMATORS"

echo "=== Обучение завершено! ==="
echo "Результаты и модель сохранены в папке results/"