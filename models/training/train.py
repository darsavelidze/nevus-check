"""
Скрипт обучения модели EfficientNet-B4 для классификации кожных образований
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Отключение проверки SSL для загрузки весов (временное решение для macOS)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Добавление корневой директории проекта в путь
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4, EfficientNetB5
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)

import config
from models.data_loader import HAM10000DataLoader, MixupSequence

# Установка seed для воспроизводимости
np.random.seed(config.RANDOM_STATE)
tf.random.set_seed(config.RANDOM_STATE)


def build_model(input_shape=(380, 380, 3), num_classes=7):
    """
    Создание модели EfficientNet-B4 с transfer learning
    
    Args:
        input_shape: размер входного изображения
        num_classes: количество классов
    
    Returns:
        model: скомпилированная модель
    """
    print("🏗️  Создание модели EfficientNet-B4...")
    
    # Загрузка базовой модели с предобученными весами ImageNet
    if getattr(config, 'MODEL_VARIANT', 'b4') == 'b5':
        base_model = EfficientNetB5(
            include_top=False,
            weights=config.WEIGHTS,
            input_shape=input_shape
        )
    else:
        base_model = EfficientNetB4(
            include_top=False,
            weights=config.WEIGHTS,
            input_shape=input_shape
        )
    
    # Заморозка базовой модели для начального обучения
    if config.FREEZE_BASE_MODEL:
        base_model.trainable = False
        print("🔒 Базовая модель заморожена для transfer learning")
    
    # Создание кастомной головы
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(config.DROPOUT_RATE)(x)
    reg = tf.keras.regularizers.l2(getattr(config, 'WEIGHT_DECAY', 0.0))
    x = Dense(getattr(config, 'HEAD_UNITS', 512), activation='relu', kernel_regularizer=reg)(x)
    if getattr(config, 'USE_BN_HEAD', True):
        x = BatchNormalization()(x)
    x = Dropout(config.DROPOUT_RATE)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Полная модель
    model = Model(inputs=base_model.input, outputs=predictions)
    
    print(f"✅ Модель создана. Всего параметров: {model.count_params():,}")
    
    return model, base_model


def compile_model(model, learning_rate=0.001):
    """
    Компиляция модели
    
    Args:
        model: модель для компиляции
        learning_rate: начальная скорость обучения
    """
    label_smoothing = getattr(config, 'LABEL_SMOOTHING', 0.0)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc')
        ]
    )
    print(f"✅ Модель скомпилирована. Label smoothing={label_smoothing}")


def create_callbacks(model_name):
    """
    Создание callbacks для обучения
    
    Args:
        model_name: имя модели для сохранения
    
    Returns:
        list of callbacks
    """
    # Создание директорий
    config.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # For robustness, save weights-only to avoid JSON serialization issues
    model_path = config.SAVED_MODELS_DIR / f"{model_name}_{timestamp}.weights.h5"
    log_dir = config.TRAINING_DIR / f"logs_{timestamp}"
    csv_path = config.TRAINING_DIR / f"training_log_{timestamp}.csv"
    
    callbacks = [
        # Сохранение лучшей модели
        ModelCheckpoint(
            filepath=str(model_path),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        
        # Ранняя остановка
        EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Уменьшение learning rate при плато
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
            verbose=1
        ),
        
        # TensorBoard логи
        TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=0,
            write_graph=False,
            write_images=False,
            update_freq='epoch'
        ),
        
        # CSV логгер
        CSVLogger(
            filename=str(csv_path),
            separator=',',
            append=False
        )
    ]
    
    print(f"📁 Модель будет сохранена в: {model_path}")
    print(f"📊 TensorBoard логи: {log_dir}")
    
    return callbacks


class UnfreezeCallback(tf.keras.callbacks.Callback):
    """
    Разморозить базовую модель после заданной эпохи и уменьшить LR.
    """
    def __init__(self, base_model, unfreeze_epoch, new_lr):
        super().__init__()
        self.base_model = base_model
        self.unfreeze_epoch = unfreeze_epoch
        self.new_lr = new_lr
        self._done = False

    def on_epoch_begin(self, epoch, logs=None):
        if not self._done and epoch >= self.unfreeze_epoch:
            self.base_model.trainable = True
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.new_lr)
            self._done = True
            print(f"\n🔓 Разморозка базовой модели на эпохе {epoch}. LR → {self.new_lr}")


def plot_training_history(history, save_path=None):
    """
    Визуализация истории обучения
    
    Args:
        history: история обучения
        save_path: путь для сохранения графика
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Top-2 Accuracy
    if 'top2_acc' in history.history:
        axes[1, 0].plot(history.history['top2_acc'], label='Train')
        axes[1, 0].plot(history.history.get('val_top2_acc', []), label='Validation')
        axes[1, 0].set_title('Top-2 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    else:
        axes[1, 0].set_visible(False)
    
    # Дополнительная панель: оставим пустой, если нет метрик
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 График сохранен: {save_path}")
    
    plt.show()


def train_model():
    """
    Основная функция обучения
    """
    print("="*80)
    print("🚀 НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ EfficientNet-B4")
    print("="*80)
    
    # Загрузка данных
    print("\n📊 Загрузка данных...")
    data_loader = HAM10000DataLoader()
    train_gen, val_gen, test_gen = data_loader.create_generators()
    class_weights = data_loader.get_class_weights()
    
    print(f"✅ Train samples: {train_gen.samples}")
    print(f"✅ Val samples: {val_gen.samples}")
    print(f"✅ Test samples: {test_gen.samples}")
    
    # Создание модели
    model, base_model = build_model(
        input_shape=(*config.IMG_SIZE, config.CHANNELS),
        num_classes=config.NUM_CLASSES
    )
    
    # Компиляция (с возможным расписанием learning rate)
    lr = config.INITIAL_LEARNING_RATE
    if getattr(config, 'USE_COSINE_LR', False):
        steps_per_epoch = int(np.ceil(train_gen.samples / config.BATCH_SIZE))
        total_steps = steps_per_epoch * config.EPOCHS
        lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config.INITIAL_LEARNING_RATE,
            decay_steps=total_steps,
            alpha=max(config.MIN_LEARNING_RATE / config.INITIAL_LEARNING_RATE, 0.0)
        )
        print("🧭 Используем CosineDecay расписание LR")
    compile_model(model, learning_rate=lr)
    
    # Callbacks
    callbacks = create_callbacks(model_name=config.MODEL_NAME)
    # Добавим разморозку базовой модели после заданной эпохи
    callbacks.append(
        UnfreezeCallback(
            base_model=base_model,
            unfreeze_epoch=config.UNFREEZE_AFTER_EPOCH,
            new_lr=max(config.INITIAL_LEARNING_RATE * 0.1, config.MIN_LEARNING_RATE)
        )
    )
    
    # Обучение
    print("\n🎯 Начало обучения...")
    print(f"Эпох: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.INITIAL_LEARNING_RATE}")
    
    # Применим MixUp при необходимости
    if getattr(config, 'USE_MIXUP', False):
        print("🧪 Активирован MixUp")
        train_input = MixupSequence(train_gen, alpha=getattr(config, 'MIXUP_ALPHA', 0.2))
    else:
        train_input = train_gen

    history = model.fit(
        train_input,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Оценка на тестовой выборке
    print("\n📊 Оценка на тестовой выборке...")
    test_results = model.evaluate(test_gen, verbose=1)
    
    print("\n" + "="*80)
    print("📈 РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ:")
    print("="*80)
    for metric_name, value in zip(model.metrics_names, test_results):
        print(f"{metric_name}: {value:.4f}")
    
    # Визуализация
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = config.TRAINING_DIR / f"training_history_{timestamp}.png"
    plot_training_history(history, save_path=plot_path)
    
    print("\n✅ Обучение завершено!")
    
    return model, history


if __name__ == "__main__":
    # Mixed precision (опционально)
    if getattr(config, 'USE_MIXED_PRECISION', False):
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("⚗️  Mixed precision: mixed_float16")
    # Проверка доступности GPU
    print("🖥️  Проверка доступного оборудования:")
    print(f"GPU доступно: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print(f"TensorFlow версия: {tf.__version__}")
    
    # Запуск обучения
    model, history = train_model()
