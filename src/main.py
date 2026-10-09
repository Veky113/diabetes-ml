import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import json
from datetime import datetime


def prepare_data(data_path: str) -> pd.DataFrame:
    """Загружает и подготавливает данные."""
    df = pd.read_csv(data_path)
    print(f"Исходный размер: {df.shape}")

    # Удаляем столбцы с большим количеством пропусков
    df = df.drop(columns=["encounter_id", "patient_nbr", "weight", "payer_code"])

    # Заменяем '?' на NaN и удаляем строки с пропусками
    df = df.replace("?", np.nan)
    df = df.dropna()

    # Кодируем категориальные признаки
    categorical_cols = df.select_dtypes(include=["object"]).columns
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    # Целевая переменная: readmitted (Yes/No/NO) -> бинарная
    df["readmitted_binary"] = (df["readmitted"] != "NO").astype(int)
    df = df.drop(columns=["readmitted"])

    return df, le_dict


def train_model(df: pd.DataFrame, random_seed: int, n_estimators: int):
    """Обучает модель и возвращает её."""
    target = "readmitted_binary"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_seed
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Accuracy: {accuracy:.4f}")
    return model, accuracy, report


def save_results(model, accuracy: float, report: dict, results_dir: str, le_dict: dict):
    """Сохраняет модель и результаты в папку results."""
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Сохраняем модель
    model_path = os.path.join(results_dir, f"model_{timestamp}.pkl")
    joblib.dump(model, model_path)
    print(f"Модель сохранена: {model_path}")

    # Сохраняем метрики
    metrics = {
        "timestamp": timestamp,
        "accuracy": accuracy,
        "classification_report": report,
    }
    metrics_path = os.path.join(results_dir, f"metrics_{timestamp}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Метрики сохранены: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Обучение модели предсказания повторной госпитализации")
    parser.add_argument("--data_path", type=str, required=True, help="Путь к CSV-файлу с данными")
    parser.add_argument("--random_seed", type=int, default=42, help="Случайное зерно")
    parser.add_argument("--n_estimators", type=int, default=100, help="Количество деревьев в лесу")
    parser.add_argument("--results_dir", type=str, default="results", help="Папка для результатов")
    args = parser.parse_args()

    print("=== Шаг 1: Подготовка данных ===")
    df, le_dict = prepare_data(args.data_path)

    print("=== Шаг 2: Обучение модели ===")
    model, accuracy, report = train_model(df, args.random_seed, args.n_estimators)

    print("=== Шаг 3: Сохранение результатов ===")
    save_results(model, accuracy, report, args.results_dir, le_dict)

    print("=== Готово! ===")


if __name__ == "__main__":
    main()