# 🚀 Быстрый старт

## Установка и настройка окружения

### 1. Клонирование репозитория
```bash
git clone <repository_url>
cd NevusAnalyzer
```

### 2. Создание виртуального окружения
```bash
python3 -m venv .venv
source .venv/bin/activate  # На macOS/Linux
# или
.venv\Scripts\activate  # На Windows
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
```

## Подготовка данных

### 1. Настройка Kaggle API

Для загрузки датасета HAM10000 необходимо настроить Kaggle API:

1. Зарегистрируйтесь на [Kaggle](https://www.kaggle.com)
2. Перейдите в [Account Settings](https://www.kaggle.com/settings/account)
3. Нажмите "Create New API Token"
4. Сохраните файл `kaggle.json` в директорию `~/.kaggle/`
5. На macOS/Linux выполните:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 2. Загрузка датасета HAM10000

```bash
python data/download_dataset.py
```

Датасет будет загружен в `data/raw/` (~1.5 GB).

### 3. Подготовка датасета

```bash
python data/prepare_dataset.py
```

Этот скрипт:
- Разделит данные на train/val/test (72%/8%/20%)
- Вычислит веса классов для борьбы с дисбалансом
- Сохранит метаданные в `data/processed/`

## Обучение модели

### Запуск обучения

```bash
python models/training/train.py
```

### Параметры обучения

Все параметры настраиваются в файле `config.py`:

- **IMG_SIZE**: (380, 380) - размер входа для EfficientNet-B4
- **BATCH_SIZE**: 16 - размер батча
- **EPOCHS**: 50 - максимальное количество эпох
- **INITIAL_LEARNING_RATE**: 0.001
- **EARLY_STOPPING_PATIENCE**: 10 - остановка при отсутствии улучшений

### Мониторинг обучения

#### TensorBoard

```bash
tensorboard --logdir=models/training/
```

Откройте браузер: http://localhost:6006

#### Просмотр логов

Логи обучения сохраняются в:
- `models/training/training_log_*.csv` - CSV формат
- `models/training/logs_*/` - TensorBoard логи
- `models/training/training_history_*.png` - графики метрик

### Результаты

После обучения:
- Лучшая модель сохраняется в `models/saved_models/EfficientNetB4_*.h5`
- Целевая точность: **>90%**
- Метрики: Accuracy, AUC, Precision, Recall

## Структура проекта

```
NevusAnalyzer/
├── config.py                 # Конфигурация обучения
├── requirements.txt          # Зависимости Python
├── README.md                # Основная документация
├── QUICKSTART.md            # Этот файл
│
├── data/
│   ├── download_dataset.py  # Загрузка HAM10000
│   ├── prepare_dataset.py   # Подготовка данных
│   ├── raw/                 # Исходные данные (игнорируется git)
│   └── processed/           # Обработанные данные
│
├── models/
│   ├── data_loader.py       # Загрузчик данных с аугментацией
│   ├── training/
│   │   └── train.py         # Скрипт обучения
│   └── saved_models/        # Сохраненные модели
│
├── mobile_app/              # Мобильное приложение (TODO)
├── hardware/                # 3D-модели крепления
├── docs/                    # Документация
└── tests/                   # Тесты
```

## Следующие шаги

1. ✅ Загрузить и подготовить датасет
2. ✅ Обучить модель EfficientNet-B4
3. 🔲 Оценить модель на тестовой выборке
4. 🔲 Конвертировать в TensorFlow Lite
5. 🔲 Разработать мобильное приложение
6. 🔲 Подготовить 3D-модели крепления

## Поддержка

При возникновении проблем:
1. Проверьте версии зависимостей в `requirements.txt`
2. Убедитесь, что Kaggle API настроен корректно
3. Проверьте доступное место на диске (минимум 5 GB)

## Системные требования

- **Python**: 3.9+
- **RAM**: минимум 8 GB (рекомендуется 16 GB)
- **Диск**: минимум 5 GB свободного места
- **GPU**: опционально (ускорит обучение в 5-10 раз)
