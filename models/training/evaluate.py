"""
Оценка сохраненной лучшей модели на тестовой выборке с отчётом по классам
и матрицей ошибок. Автоматически находит последний .weights.h5.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Корень проекта
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config
from models.training.train import build_model, compile_model
from models.data_loader import HAM10000DataLoader


def find_latest_weights(weights_dir: Path) -> Path:
    candidates = sorted(weights_dir.glob("*.weights.h5"))
    if not candidates:
        raise FileNotFoundError(f"Не найдено весов в {weights_dir}")
    return candidates[-1]


def tta_batch_predictions(model, x_batch):
    # Простая TTA: исходное, flip_lr, flip_ud
    preds = []
    preds.append(model.predict(x_batch, verbose=0))
    preds.append(model.predict(np.flip(x_batch, axis=2), verbose=0))  # горизонтальный
    preds.append(model.predict(np.flip(x_batch, axis=1), verbose=0))  # вертикальный
    return np.mean(preds, axis=0)


def evaluate():
    print("=" * 80)
    print("🔎 ОЦЕНКА ЛУЧШЕЙ МОДЕЛИ НА ТЕСТЕ")
    print("=" * 80)

    # Данные
    loader = HAM10000DataLoader()
    _, _, test_gen = loader.create_generators()

    # Модель
    model, _ = build_model(input_shape=(*config.IMG_SIZE, config.CHANNELS), num_classes=config.NUM_CLASSES)
    compile_model(model, learning_rate=config.MIN_LEARNING_RATE)

    # Загрузка весов
    weights_path = find_latest_weights(config.SAVED_MODELS_DIR)
    model.load_weights(str(weights_path))
    print(f"✅ Загружены веса: {weights_path}")

    # Оценка
    results = model.evaluate(test_gen, verbose=1)
    print("\nРЕЗУЛЬТАТЫ:")
    for k, v in zip(model.metrics_names, results):
        print(f"{k}: {v:.4f}")

    # Предсказания и отчёт
    # Предсказания с лёгкой TTA
    y_prob_list = []
    for i in range(len(test_gen)):
        x_batch, _ = test_gen[i]
        y_prob_list.append(tta_batch_predictions(model, x_batch))
    y_prob = np.vstack(y_prob_list)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes

    idx_to_class = {v: k for k, v in test_gen.class_indices.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("\nКЛАССИФИКАЦИОННЫЙ ОТЧЁТ:\n")
    print(report)

    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=target_names, yticklabels=target_names)
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
    ax.set_title('Матрица ошибок')
    plt.tight_layout()

    config.TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = config.TRAINING_DIR / f"confusion_matrix_{ts}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"📉 Матрица ошибок сохранена: {out_path}")


if __name__ == "__main__":
    evaluate()
