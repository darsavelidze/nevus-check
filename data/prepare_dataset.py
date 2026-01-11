"""
Подготовка датасета HAM10000 для обучения
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

# Маппинг классов
CLASS_MAPPING = {
    'akiec': 'Actinic Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus',
    'vasc': 'Vascular Lesion'
}

CLASS_TO_IDX = {k: i for i, k in enumerate(CLASS_MAPPING.keys())}

def prepare_ham10000(test_size=0.2, val_size=0.1, random_state=42):
    """
    Подготовка датасета HAM10000
    
    Args:
        test_size: размер тестовой выборки
        val_size: размер валидационной выборки (от обучающей)
        random_state: seed для воспроизводимости
    """
    
    print("🔧 Подготовка датасета HAM10000...")
    
    # Пути
    raw_dir = Path(__file__).parent / "raw"
    processed_dir = Path(__file__).parent / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    # Загрузка метаданных
    metadata_path = raw_dir / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        print("❌ Метаданные не найдены. Сначала загрузите датасет: python data/download_dataset.py")
        return None, None, None
    
    df = pd.read_csv(metadata_path)
    print(f"✅ Загружено {len(df)} записей")
    
    # Добавление колонки с индексом класса
    df['class_idx'] = df['dx'].map(CLASS_TO_IDX)
    
    # Проверка наличия изображений
    img_dirs = [raw_dir / "HAM10000_images_part_1", raw_dir / "HAM10000_images_part_2"]
    
    # Добавление пути к изображению
    def find_image_path(image_id):
        for img_dir in img_dirs:
            img_path = img_dir / f"{image_id}.jpg"
            if img_path.exists():
                return str(img_path.relative_to(raw_dir))
        return None
    
    df['image_path'] = df['image_id'].apply(find_image_path)
    
    # Проверка на наличие всех изображений
    missing = df['image_path'].isna().sum()
    if missing > 0:
        print(f"⚠️  Предупреждение: {missing} изображений не найдено")
        df = df.dropna(subset=['image_path'])
    
    print(f"✅ Найдено {len(df)} изображений")
    
    # Разделение на train/test с учетом баланса классов
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['dx'],
        random_state=random_state
    )
    
    # Разделение train на train/val
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df['dx'],
        random_state=random_state
    )
    
    print(f"\n📊 Разделение данных:")
    print(f"Train: {len(train_df):5d} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val_df):5d} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test_df):5d} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Сохранение CSV файлов
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)
    
    print("\n✅ Датасет подготовлен и сохранен в data/processed/")
    
    # Статистика по классам
    print("\n📈 Распределение классов (Train):")
    class_dist = train_df['dx'].value_counts()
    for cls, count in class_dist.items():
        print(f"  {cls:6s}: {count:4d} ({count/len(train_df)*100:5.2f}%)")
    
    # Вычисление весов классов для борьбы с дисбалансом
    class_counts = train_df['dx'].value_counts()
    total = len(train_df)
    class_weights = {CLASS_TO_IDX[cls]: total / (len(class_counts) * count) 
                     for cls, count in class_counts.items()}
    
    print("\n⚖️  Веса классов для weighted loss:")
    for cls_name, idx in CLASS_TO_IDX.items():
        print(f"  {cls_name:6s}: {class_weights[idx]:.3f}")
    
    # Сохранение весов и маппинга
    np.save(processed_dir / "class_weights.npy", class_weights)
    
    with open(processed_dir / "class_mapping.json", 'w', encoding='utf-8') as f:
        json.dump(CLASS_MAPPING, f, ensure_ascii=False, indent=2)
    
    with open(processed_dir / "class_to_idx.json", 'w', encoding='utf-8') as f:
        json.dump(CLASS_TO_IDX, f, ensure_ascii=False, indent=2)
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    prepare_ham10000()
