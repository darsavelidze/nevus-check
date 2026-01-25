
"""
train_vit.py
Обучение ViT на PyTorch с балансировкой, аугментацией и timm.
"""

import os
import random
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import timm

# Конфигурация
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
SEED = 42
PATIENCE = 7

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw'
METADATA_PATH = DATA_DIR / 'HAM10000_metadata.csv'
IMAGE_DIRS = [
    DATA_DIR / 'HAM10000_images_part_1',
    DATA_DIR / 'HAM10000_images_part_2'
]
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
NUM_CLASSES = len(CLASSES)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(METADATA_PATH)
    def resolve_path(image_id: str) -> str:
        for img_dir in IMAGE_DIRS:
            candidate = img_dir / f"{image_id}.jpg"
            if candidate.exists():
                return str(candidate)
        return None

    df['image_path'] = df['image_id'].apply(resolve_path)
    df = df[df['image_path'].notna()].reset_index(drop=True)
    label_map = {label: idx for idx, label in enumerate(CLASSES)}
    df['label'] = df['dx'].map(label_map)
    return df


def balance_oversample(df: pd.DataFrame) -> pd.DataFrame:
    max_count = df['label'].value_counts().max()
    balanced = []
    for cls in range(NUM_CLASSES):
        subset = df[df['label'] == cls]
        if len(subset) == 0:
            continue
        if len(subset) < max_count:
            sampled = subset.sample(max_count, replace=True, random_state=SEED)
        else:
            sampled = subset
        balanced.append(sampled)
    return pd.concat(balanced).sample(frac=1, random_state=SEED).reset_index(drop=True)


class HAMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        image = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = int(row['label'])
        return image, label


def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE + 32),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    return transforms.Compose([
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


def create_dataloaders(df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['label'],
        random_state=SEED
    )

    train_dataset = HAMDataset(train_df, transform=get_transforms(train=True))
    val_dataset = HAMDataset(val_df, transform=get_transforms(train=False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, val_loader


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    _, pred_classes = torch.max(preds, dim=1)
    correct = (pred_classes == targets).sum().item()
    return correct / targets.size(0)


def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += accuracy(outputs, labels) * images.size(0)

    num_samples = len(loader.dataset)
    return running_loss / num_samples, running_acc / num_samples


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            running_acc += accuracy(outputs, labels) * images.size(0)

    num_samples = len(loader.dataset)
    return running_loss / num_samples, running_acc / num_samples


def main():
    set_seed(SEED)
    df = load_metadata()
    print('Class counts before balancing:')
    print(df['label'].value_counts().sort_index())

    balanced_df = balance_oversample(df)
    print('Class counts after oversampling:')
    print(balanced_df['label'].value_counts().sort_index())

    train_loader, val_loader = create_dataloaders(balanced_df)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_CLASSES)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_acc = 0.0
    steps_without_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            steps_without_improve = 0
            torch.save(model.state_dict(), MODEL_DIR / 'vit_state.pt')
            print(f"Best model saved (acc {best_acc:.4f}).")
        else:
            steps_without_improve += 1

        if steps_without_improve >= PATIENCE:
            print(f"Stopping early (no improvement for {PATIENCE} epochs)")
            break

    print('Training complete. Best accuracy:', best_acc)


if __name__ == '__main__':
    main()
