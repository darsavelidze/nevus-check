"""
Скрипт для загрузки датасета HAM10000
Датасет доступен на Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def download_ham10000():
    """
    Инструкция по загрузке HAM10000:
    
    1. Зарегистрируйтесь на Kaggle: https://www.kaggle.com
    2. Установите Kaggle API (уже установлен в requirements.txt)
    3. Скачайте API token:
       - Зайдите в Account settings на Kaggle
       - Нажмите "Create New API Token"
       - Сохраните kaggle.json в ~/.kaggle/
       - На macOS: chmod 600 ~/.kaggle/kaggle.json
    4. Запустите этот скрипт
    """
    
    print("📥 Загрузка датасета HAM10000 с Kaggle...")
    
    # Проверка наличия Kaggle API credentials
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("❌ Kaggle API token не найден!")
        print("\n📝 Инструкция по настройке:")
        print("1. Зайдите на https://www.kaggle.com/settings/account")
        print("2. Нажмите 'Create New API Token'")
        print("3. Сохраните kaggle.json в ~/.kaggle/")
        print(f"4. Выполните: chmod 600 {kaggle_json}")
        return False
    
    # Проверка наличия Kaggle API
    try:
        import kaggle
    except ImportError:
        print("❌ Kaggle API не установлен. Установите: pip install kaggle")
        return False
    
    # Путь для сохранения
    data_dir = Path(__file__).parent / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Загрузка через Kaggle API
    print(f"📂 Сохранение в: {data_dir}")
    os.system(f"kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p {data_dir} --unzip")
    
    print("\n✅ Датасет HAM10000 загружен в data/raw/")
    
    # Проверка структуры
    metadata_path = data_dir / "HAM10000_metadata.csv"
    if metadata_path.exists():
        df = pd.read_csv(metadata_path)
        print(f"\n📊 Статистика датасета:")
        print(f"Всего изображений: {len(df)}")
        print(f"\nРаспределение по классам:")
        class_dist = df['dx'].value_counts()
        for cls, count in class_dist.items():
            print(f"  {cls:6s}: {count:5d} ({count/len(df)*100:5.2f}%)")
        return True
    else:
        print("⚠️ Метаданные не найдены. Проверьте загрузку.")
        return False

if __name__ == "__main__":
    download_ham10000()
