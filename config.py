"""
Конфигурация для обучения модели EfficientNet-B4
"""

from pathlib import Path

# Пути
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
TRAINING_DIR = MODELS_DIR / "training"

# Параметры изображений
IMG_SIZE = (380, 380)  # Для B4: 380; B5: 456
BATCH_SIZE = 16  # Уменьшен для экономии памяти
CHANNELS = 3

# Параметры обучения
EPOCHS = 50
INITIAL_LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 1e-7
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# Расписание обучения
USE_COSINE_LR = False  # Включить CosineDecay вместо фиксированного LR
LABEL_SMOOTHING = 0.05

# Классы
NUM_CLASSES = 7
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Аугментация данных
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.15,
    'height_shift_range': 0.15,
    'shear_range': 0.1,
    'zoom_range': 0.15,
    'horizontal_flip': True,
    'vertical_flip': True,
    'fill_mode': 'nearest',
    'brightness_range': [0.8, 1.2]
}

# Transfer Learning
FREEZE_BASE_MODEL = True  # Заморозить базовую модель на первых эпохах
UNFREEZE_AFTER_EPOCH = 10  # После какой эпохи разморозить

# Модель
MODEL_NAME = 'EfficientNetB4'
MODEL_VARIANT = 'b4'  # 'b4' или 'b5'
WEIGHTS = 'imagenet'
DROPOUT_RATE = 0.3
HEAD_UNITS = 512
USE_BN_HEAD = True
WEIGHT_DECAY = 1e-5

# Сохранение модели
SAVE_BEST_ONLY = True
SAVE_WEIGHTS_ONLY = False

# Случайность
RANDOM_STATE = 42

# Обработка дисбаланса и аугментации
USE_OVERSAMPLING = True
OVERSAMPLING_STRATEGY = 'median'  # 'none' | 'median' | 'max'
OVERSAMPLING_MAX_MULT = 3  # ограничение на множитель апсемплинга
USE_MIXUP = True
MIXUP_ALPHA = 0.2
USE_CUTMIX = False
CUTMIX_ALPHA = 0.2

# Производительность
USE_MIXED_PRECISION = False  # при True включает mixed_float16
