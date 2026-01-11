"""
Data loader для HAM10000 с аугментацией
"""

import sys
from pathlib import Path

# Добавление корневой директории проекта в путь
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
import config

class HAM10000DataLoader:
    """
    Загрузчик данных для датасета HAM10000
    """
    
    def __init__(self, data_dir=None):
        """
        Args:
            data_dir: путь к директории с данными
        """
        self.data_dir = data_dir or config.RAW_DATA_DIR
        self.processed_dir = config.PROCESSED_DATA_DIR
        self.img_size = config.IMG_SIZE
        self.batch_size = config.BATCH_SIZE
        
        # Загрузка class weights
        class_weights_path = self.processed_dir / "class_weights.npy"
        if class_weights_path.exists():
            self.class_weights = np.load(class_weights_path, allow_pickle=True).item()
        else:
            self.class_weights = None
    
    def create_generators(self):
        """
        Создание генераторов для train/val/test
        
        Returns:
            train_gen, val_gen, test_gen
        """
        
        # Генератор с аугментацией для train
        # Используем preprocess_input для EfficientNet вместо простого rescale
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            **config.AUGMENTATION_CONFIG
        )
        
        # Генератор без аугментации для val/test
        val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        
        # Загрузка CSV
        train_df = pd.read_csv(self.processed_dir / "train.csv")
        val_df = pd.read_csv(self.processed_dir / "val.csv")
        test_df = pd.read_csv(self.processed_dir / "test.csv")
        
        # Добавление полного пути к изображениям
        train_df['full_path'] = train_df['image_path'].apply(
            lambda x: str(self.data_dir / x)
        )
        val_df['full_path'] = val_df['image_path'].apply(
            lambda x: str(self.data_dir / x)
        )
        test_df['full_path'] = test_df['image_path'].apply(
            lambda x: str(self.data_dir / x)
        )
        
        # Опциональный апсемплинг редких классов для train
        if getattr(config, 'USE_OVERSAMPLING', False):
            class_counts = train_df['dx'].value_counts()
            if getattr(config, 'OVERSAMPLING_STRATEGY', 'none') == 'median':
                target = int(class_counts.median())
            elif config.OVERSAMPLING_STRATEGY == 'max':
                target = int(class_counts.max())
            else:
                target = None
            if target:
                rows = []
                for cls, cnt in class_counts.items():
                    df_c = train_df[train_df['dx'] == cls]
                    max_allowed = min(target, int(cnt * getattr(config, 'OVERSAMPLING_MAX_MULT', 3)))
                    if cnt < max_allowed:
                        add_n = max_allowed - cnt
                        dup = df_c.sample(n=add_n, replace=True, random_state=config.RANDOM_STATE)
                        rows.append(dup)
                if rows:
                    train_df = pd.concat([train_df] + rows, axis=0).sample(frac=1.0, random_state=config.RANDOM_STATE).reset_index(drop=True)

        # Создание генераторов
        train_gen = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col='full_path',
            y_col='dx',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=config.RANDOM_STATE
        )
        
        val_gen = val_test_datagen.flow_from_dataframe(
            dataframe=val_df,
            x_col='full_path',
            y_col='dx',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_gen = val_test_datagen.flow_from_dataframe(
            dataframe=test_df,
            x_col='full_path',
            y_col='dx',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_gen, val_gen, test_gen

    def get_class_weights(self):
        """Получить веса классов для борьбы с дисбалансом"""
        if self.class_weights is None:
            return None
        # Преобразуем в обычные Python числа для сериализации в JSON
        # Используем .numpy() для TensorFlow тензоров и .item() для numpy scalar
        result = {}
        for k, v in self.class_weights.items():
            key = int(k)
            # Обработка различных типов значений
            if hasattr(v, 'numpy'):  # TensorFlow tensor
                value = float(v.numpy())
            elif hasattr(v, 'item'):  # NumPy scalar
                value = float(v.item())
            else:  # Обычный Python тип
                value = float(v)
            result[key] = value
        return result


class MixupSequence(tf.keras.utils.Sequence):
    def __init__(self, base_sequence, alpha=0.2):
        self.base = base_sequence
        self.alpha = alpha

    def __len__(self):
        return len(self.base)

    def on_epoch_end(self):
        if hasattr(self.base, 'on_epoch_end'):
            self.base.on_epoch_end()

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if x.shape[0] < 2:
            return x, y
        lam = np.random.beta(self.alpha, self.alpha)
        perm = np.random.permutation(x.shape[0])
        x2, y2 = x[perm], y[perm]
        x_mix = lam * x + (1 - lam) * x2
        y_mix = lam * y + (1 - lam) * y2
        return x_mix, y_mix
    
