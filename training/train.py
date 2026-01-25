"""
Skin Lesion Classification Model Training Script
Based on: "Optimized CNN for Skin Lesion Classification" (Appl. Sci. 2023)

This script trains a CNN model on the HAM10000 dataset for skin lesion classification.
Target accuracy: 90%+

Classes:
- nv: Melanocytic nevus (6705 images)
- mel: Melanoma (1113 images)
- bkl: Benign keratosis (1099 images)
- bcc: Basal cell carcinoma (514 images)
- akiec: Actinic keratosis (327 images)
- vasc: Vascular lesions (142 images)
- df: Dermatofibroma (115 images)
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Enable mixed precision training for GPU memory efficiency
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"GPU Memory Optimization: Using {policy.name} precision")

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration
class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    METADATA_PATH = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
    IMAGE_DIRS = [
        os.path.join(DATA_DIR, 'HAM10000_images_part_1'),
        os.path.join(DATA_DIR, 'HAM10000_images_part_2')
    ]
    MODEL_DIR = os.path.join(BASE_DIR, 'models')

    # Image settings (optimized for GPU memory: RTX 4060 8GB)
    IMG_SIZE = 75  # OPTIMIZED: Компромисс между 50 (малый) и 100 (памяти много)
    IMG_CHANNELS = 3

    # Training hyperparameters (optimized for GPU memory)
    BATCH_SIZE = 16  # OPTIMIZED: Уменьшено для экономии памяти GPU
    EPOCHS = 100  # Достаточно для сходимости
    LEARNING_RATE = 0.0001
    DROPOUT_RATE = 0.3

    # GA search settings (optimized for GPU memory constraints)
    GA_POPULATION = 6  # OPTIMIZED: Уменьшено для экономии памяти
    GA_GENERATIONS = 4  # OPTIMIZED: Сбалансировано (был 5, теперь 4)
    GA_EPOCHS = 6  # OPTIMIZED: Уменьшено с 8 для скорости поиска
    GA_MUTATION_PROB = 0.4

    # Class labels
    CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    NUM_CLASSES = 7

    # Random seed for reproducibility
    SEED = 42


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_metadata():
    """Load and preprocess metadata"""
    print("Loading metadata...")
    df = pd.read_csv(Config.METADATA_PATH)
    
    # Create image path column
    def find_image_path(image_id):
        for img_dir in Config.IMAGE_DIRS:
            path = os.path.join(img_dir, f'{image_id}.jpg')
            if os.path.exists(path):
                return path
        return None
    
    df['image_path'] = df['image_id'].apply(find_image_path)
    
    # Remove rows with missing images
    df = df[df['image_path'].notna()].reset_index(drop=True)
    
    # Create numeric labels
    label_map = {label: idx for idx, label in enumerate(Config.CLASSES)}
    df['label'] = df['dx'].map(label_map)
    
    print(f"Total images found: {len(df)}")
    print("\nClass distribution:")
    print(df['dx'].value_counts())
    
    return df


def load_and_preprocess_image(image_path, target_size):
    """Load and preprocess a single image"""
    try:
        img = Image.open(image_path)
        img = img.resize((target_size, target_size), Image.LANCZOS)
        img = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1]
        img = img / 255.0
        
        return img
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def load_dataset(df, target_size=Config.IMG_SIZE):
    """Load all images into memory"""
    print(f"\nLoading images (resizing to {target_size}x{target_size})...")
    
    images = []
    labels = []
    valid_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading images"):
        img = load_and_preprocess_image(row['image_path'], target_size)
        if img is not None:
            images.append(img)
            labels.append(row['label'])
            valid_indices.append(idx)
    
    X = np.array(images)
    y = np.array(labels)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    return X, y


def augment_to_balance(X, y, target_count=None):
    """Oversample minority classes with augmentation to balance the dataset."""
    class_indices = {c: np.where(y == c)[0] for c in np.unique(y)}
    current_counts = {c: len(idxs) for c, idxs in class_indices.items()}
    max_count = max(current_counts.values()) if target_count is None else target_count

    if all(count >= max_count for count in current_counts.values()):
        return X, y

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )

    new_images, new_labels = [], []
    for cls, idxs in class_indices.items():
        needed = max_count - len(idxs)
        if needed <= 0:
            continue
        samples = X[idxs]
        gen = datagen.flow(samples, batch_size=1, shuffle=True)
        for _ in range(needed):
            aug_img = next(gen)[0]
            new_images.append(aug_img)
            new_labels.append(cls)

    if new_images:
        X_bal = np.concatenate([X, np.array(new_images)], axis=0)
        y_bal = np.concatenate([y, np.array(new_labels)], axis=0)
        return X_bal, y_bal
    return X, y


def create_augmented_generators(X_train, y_train, X_val, y_val, batch_size):
    """Create data generators with augmentation for training"""
    
    # Strong augmentation for training (based on the paper)
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator()
    
    # Fit the generator on training data
    train_datagen.fit(X_train)
    
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_generator, val_generator


def build_cnn_model(input_shape, num_classes, hparams):
    """Build the compact CNN described in the paper with tunable hyperparameters."""
    filters = hparams['filters']
    dense_units = hparams['dense_units']
    dropout_rate = hparams['dropout']

    layers = [Input(shape=input_shape)]
    for f in filters:
        layers.append(Conv2D(f, (3, 3), activation='relu', padding='same'))
        layers.append(BatchNormalization())
        layers.append(MaxPooling2D(pool_size=(2, 2)))

    layers.append(Flatten())

    for units in dense_units:
        layers.append(Dense(units, activation='relu'))
        layers.append(BatchNormalization())
        layers.append(Dropout(dropout_rate))

    # Output layer with float32 for mixed precision stability
    layers.append(Dense(num_classes, activation='softmax', dtype='float32'))

    model = Sequential(layers)
    return model


def compile_model(model, learning_rate):
    """Compile the model with optimizer and loss function"""
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def _safe_div(num, den):
    return num / den if den != 0 else 0.0


def compute_multiclass_metrics(y_true, y_pred, num_classes):
    """Compute macro metrics used in the paper (Dice, ACC, SN, SP, PREC, F-score)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    precision = [_safe_div(tp[i], tp[i] + fp[i]) for i in range(num_classes)]
    recall = [_safe_div(tp[i], tp[i] + fn[i]) for i in range(num_classes)]  # sensitivity
    specificity = [_safe_div(tn[i], tn[i] + fp[i]) for i in range(num_classes)]
    dice = [_safe_div(2 * tp[i], (2 * tp[i]) + fp[i] + fn[i]) for i in range(num_classes)]

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_specificity = float(np.mean(specificity))
    macro_dice = float(np.mean(dice))
    macro_f1 = _safe_div(2 * macro_precision * macro_recall, macro_precision + macro_recall)
    accuracy = float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0

    return {
        'dice': macro_dice,
        'accuracy': accuracy,
        'sensitivity': macro_recall,
        'specificity': macro_specificity,
        'precision': macro_precision,
        'f1': macro_f1
    }


def fitness_from_metrics(metrics):
    """Average the six metrics as a single fitness score."""
    keys = ['dice', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1']
    return float(np.mean([metrics[k] for k in keys]))


def sample_hparams():
    """Sample a hyperparameter set within the ranges of the paper."""
    filter_choices = [
        [32, 64, 128],
        [64, 128, 256],
        [32, 64, 128, 256]  # slightly deeper variant
    ]
    dense_choices = [
        [256, 128],
        [256, 128, 64]
    ]
    dropout_choices = [0.25, 0.3, 0.35, 0.4]
    lr_choices = [1e-5, 5e-5, 1e-4]  # Lower LR for stability
    batch_choices = [32, 64]

    return {
        'filters': filter_choices[np.random.randint(len(filter_choices))],
        'dense_units': dense_choices[np.random.randint(len(dense_choices))],
        'dropout': float(dropout_choices[np.random.randint(len(dropout_choices))]),
        'learning_rate': float(lr_choices[np.random.randint(len(lr_choices))]),
        'batch_size': int(batch_choices[np.random.randint(len(batch_choices))])
    }


def crossover(parent1, parent2):
    """Single-point crossover across hyperparameter dicts."""
    child = {}
    keys = list(parent1.keys())
    split = len(keys) // 2
    for i, key in enumerate(keys):
        child[key] = parent1[key] if i < split else parent2[key]
    return child


def mutate(hparams, prob=Config.GA_MUTATION_PROB):
    """Randomly mutate hyperparameters."""
    mutated = dict(hparams)
    for key in mutated.keys():
        if np.random.rand() < prob:
            mutated[key] = sample_hparams()[key]
    return mutated


def evaluate_candidate(hparams, X_train, y_train, X_val, y_val):
    """Train briefly and return fitness and metrics for a candidate set of hyperparameters."""
    input_shape = (Config.IMG_SIZE, Config.IMG_SIZE, Config.IMG_CHANNELS)
    num_classes = Config.NUM_CLASSES

    model = build_cnn_model(input_shape, num_classes, hparams)
    model = compile_model(model, hparams['learning_rate'])

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )
    val_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow(X_train, y_train, batch_size=hparams['batch_size'], shuffle=True)
    val_gen = val_datagen.flow(X_val, y_val, batch_size=hparams['batch_size'], shuffle=False)

    steps_per_epoch = max(1, len(X_train) // hparams['batch_size'])
    val_steps = max(1, len(X_val) // hparams['batch_size'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=0)
    ]

    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=Config.GA_EPOCHS,
        validation_data=val_gen,
        validation_steps=val_steps,
        verbose=0,
        callbacks=callbacks
    )

    # Evaluate on validation set
    y_val_true = np.argmax(y_val, axis=1)
    y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    metrics = compute_multiclass_metrics(y_val_true, y_val_pred, num_classes)
    fitness = fitness_from_metrics(metrics)
    return fitness, metrics


def genetic_algorithm_search(X_train, y_train, X_val, y_val):
    """Run a lightweight GA to approximate the paper's hyperparameter tuning."""
    population = [sample_hparams() for _ in range(Config.GA_POPULATION)]
    best = None

    for gen in range(Config.GA_GENERATIONS):
        scored = []
        for hparams in population:
            fitness, metrics = evaluate_candidate(hparams, X_train, y_train, X_val, y_val)
            scored.append((fitness, hparams, metrics))
            print(f"GA Gen {gen+1}: fitness={fitness:.4f}, hparams={hparams}, metrics={metrics}")

        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0] if best is None or scored[0][0] > best[0] else best

        # Selection: top 50%
        survivors = [item[1] for item in scored[: max(1, len(scored)//2)]]

        # Crossover + mutation to refill population
        new_population = survivors.copy()
        while len(new_population) < Config.GA_POPULATION:
            parents = np.random.choice(len(survivors), size=2, replace=True)
            child = crossover(survivors[parents[0]], survivors[parents[1]])
            child = mutate(child)
            new_population.append(child)
        population = new_population

    best_fitness, best_hparams, best_metrics = best
    print(f"Best GA fitness={best_fitness:.4f}, hparams={best_hparams}, metrics={best_metrics}")
    return best_hparams


def get_callbacks(model_path):
    """Get training callbacks"""
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-8,
            verbose=1
        )
    ]
    
    return callbacks


def train_with_hparams(hparams, X_train, y_train, X_val, y_val, class_weights=None):
    """Train the paper CNN with selected hyperparameters for full epochs."""
    input_shape = (Config.IMG_SIZE, Config.IMG_SIZE, Config.IMG_CHANNELS)
    model = build_cnn_model(input_shape, Config.NUM_CLASSES, hparams)
    model = compile_model(model, hparams['learning_rate'])

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )
    val_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow(X_train, y_train, batch_size=hparams['batch_size'], shuffle=True)
    val_gen = val_datagen.flow(X_val, y_val, batch_size=hparams['batch_size'], shuffle=False)

    callbacks = get_callbacks(None)
    steps_per_epoch = max(1, len(X_train) // hparams['batch_size'])
    val_steps = max(1, len(X_val) // hparams['batch_size'])

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=Config.EPOCHS,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        class_weight=class_weights,  # IMPROVED: Использование class weights для сложных классов
        verbose=1
    )
    return model, history


def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate model and print metrics"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    metrics = compute_multiclass_metrics(y_true, y_pred, len(class_names))
    print(f"\nTest Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Dice: {metrics['dice']:.4f} | Sensitivity: {metrics['sensitivity']:.4f} | Specificity: {metrics['specificity']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} | F1: {metrics['f1']:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return y_true, y_pred, metrics


def main():
    """Main training function"""
    print("="*60)
    print("SKIN LESION CLASSIFICATION - MODEL TRAINING")
    print("Based on: Optimized CNN with Genetic Algorithm (Appl. Sci. 2023)")
    print("="*60)
    
    # Set random seed
    set_seed(Config.SEED)
    
    # Create model directory
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Load metadata
    df = load_metadata()
    
    # Load dataset
    X, y = load_dataset(df, target_size=Config.IMG_SIZE)

    # Balance via augmentation (oversample minority classes)
    X_bal, y_bal = augment_to_balance(X, y)
    print(f"Dataset balanced from {len(y)} to {len(y_bal)} samples")
    
    # Print class distribution after balancing
    import collections
    print("\nClass distribution after balancing:")
    class_counts = collections.Counter(y_bal)
    for class_idx in range(Config.NUM_CLASSES):
        count = class_counts.get(class_idx, 0)
        class_name = Config.CLASSES[class_idx]
        print(f"  {class_name}: {count} samples")

    # Convert labels to categorical
    y_bal_cat = to_categorical(y_bal, num_classes=Config.NUM_CLASSES)

    # Split data: 80% train, 20% test (as per the paper)
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal_cat,
        test_size=0.2,
        stratify=y_bal,
        random_state=Config.SEED
    )

    # Further split training: 70% train, 30% validation for stability
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.3,
        stratify=np.argmax(y_train, axis=1),
        random_state=Config.SEED
    )

    print(f"\nDataset splits:")
    print(f"  Training:   {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test:       {X_test.shape[0]} samples")
    
    # Compute class weights for better handling of difficult classes (mel, bkl)
    from sklearn.utils.class_weight import compute_class_weight
    y_train_labels = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights (for difficult classes like mel, bkl):")
    for i, w in class_weight_dict.items():
        print(f"  {Config.CLASSES[i]}: {w:.3f}")

    # GA hyperparameter search
    print("\nRunning genetic algorithm search for hyperparameters...")
    best_hparams = genetic_algorithm_search(X_train, y_train, X_val, y_val)

    # Generate version name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save best hyperparameters
    hparams_path = os.path.join(Config.MODEL_DIR, "best_hparams.json")
    with open(hparams_path, 'w') as f:
        json.dump(best_hparams, f, indent=2)
    print(f"Saved best hyperparameters to {hparams_path}")

    # Train final model with best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    model, history = train_with_hparams(best_hparams, X_train, y_train, X_val, y_val, class_weights=class_weight_dict)

    # Evaluate on test set
    print("\nEvaluating model...")
    y_true, y_pred, test_metrics = evaluate_model(
        model, X_test, y_test, Config.CLASSES
    )

    # Create version-specific name with metrics
    accuracy_str = f"{test_metrics['accuracy']*100:.1f}"
    f1_str = f"{test_metrics['f1']*100:.1f}"
    version_name = f"model_v{timestamp}_acc{accuracy_str}_f1{f1_str}"
    
    # Create directories FIRST before any file operations
    versions_dir = os.path.join(Config.MODEL_DIR, "versions")
    plots_dir = os.path.join(Config.MODEL_DIR, "plots")
    os.makedirs(versions_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save model weights to versions folder FIRST (critical!)
    versioned_weights_path = os.path.join(versions_dir, f"{version_name}.h5")
    model.save_weights(versioned_weights_path)
    print(f"\nVersioned model saved to: {versioned_weights_path}")
    
    # Also save as current best model (for app to use)
    best_weights_path = os.path.join(Config.MODEL_DIR, "skin_lesion_cnn_paper_final_weights.h5")
    model.save_weights(best_weights_path)
    print(f"Current best model: {best_weights_path}")
    
    # Save training plots (after weights, so if this fails, we still have the model)
    try:
        plot_training_history(
            history,
            os.path.join(plots_dir, f"{version_name}_training_history.png")
        )

        plot_confusion_matrix(
            y_true, y_pred, Config.CLASSES,
            os.path.join(plots_dir, f"{version_name}_confusion_matrix.png")
        )
        print("Plots saved successfully")
    except Exception as e:
        print(f"Warning: Could not save plots: {e}")
    
    
    # Save training metadata
    metadata = {
        "version": version_name,
        "timestamp": timestamp,
        "hyperparameters": best_hparams,
        "metrics": {
            "accuracy": float(test_metrics['accuracy']),
            "dice": float(test_metrics['dice']),
            "sensitivity": float(test_metrics['sensitivity']),
            "specificity": float(test_metrics['specificity']),
            "precision": float(test_metrics['precision']),
            "f1": float(test_metrics['f1'])
        },
        "config": {
            "image_size": Config.IMG_SIZE,
            "epochs": Config.EPOCHS,
            "batch_size": best_hparams['batch_size']
        }
    }
    
    metadata_path = os.path.join(versions_dir, f"{version_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    # Print final results
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nVersion: {version_name}")
    print(f"Final Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Dice: {test_metrics['dice']:.4f} | Sensitivity: {test_metrics['sensitivity']:.4f} | Specificity: {test_metrics['specificity']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f} | F1: {test_metrics['f1']:.4f}")

    return model, test_metrics


def load_trained_model(weights_path=None, hparams_path=None):
    """Rebuild the paper CNN from saved hyperparameters and weights."""
    if weights_path is None:
        weights_path = os.path.join(Config.MODEL_DIR, "skin_lesion_cnn_paper_final_weights.h5")
    if hparams_path is None:
        hparams_path = os.path.join(Config.MODEL_DIR, "best_hparams.json")

    with open(hparams_path, 'r') as f:
        hparams = json.load(f)

    model = build_cnn_model(
        input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, Config.IMG_CHANNELS),
        num_classes=Config.NUM_CLASSES,
        hparams=hparams
    )
    model = compile_model(model, hparams.get('learning_rate', Config.LEARNING_RATE))
    model.load_weights(weights_path)
    print(f"Model loaded with weights from {weights_path} and hparams from {hparams_path}")
    return model


if __name__ == "__main__":
    model, metrics = main()
