"""
Dataset loading and augmentation for IP102 insect pest classification.

Mirrors the dataset setup used in the project notebook
(notebooks/pest_classifier.ipynb), organized as reusable functions.

Author: Selin Bildirici
CS 372 Final Project

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


# ImageNet normalization stats (used because we fine-tune ImageNet-pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================
# Basic transforms (used in Runs 1-2, no augmentation)
# ============================================

basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ============================================
# Albumentations transforms (used in Runs 3-8 with augmentation)
# ============================================

# Training augmentation pipeline: 5 techniques per project blueprint
train_aug = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.7),
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(8, 32),
                    hole_width_range=(8, 32), p=0.3),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# Validation/test pipeline: deterministic resize + normalize, no augmentation
val_aug = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


class AlbumentationsDataset(Dataset):
    """Wraps a torchvision ImageFolder to apply Albumentations transforms.

    ImageFolder yields PIL images; Albumentations operates on numpy arrays.
    This wrapper bridges the two while preserving the class_to_idx mapping.
    """
    def __init__(self, image_folder_dataset, transform):
        self.dataset = image_folder_dataset
        self.transform = transform
        self.samples = image_folder_dataset.samples
        self.classes = image_folder_dataset.classes
        self.class_to_idx = image_folder_dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        image = self.transform(image=image)['image']
        return image, label


def build_basic_datasets(dataset_root):
    """Build train/val/test datasets with basic transform (no augmentation).

    Used in Runs 1 (frozen ViT) and Run 2 (unfrozen ViT, no aug).

    Args:
        dataset_root: path to dataset root containing train/val/test subdirs

    Returns:
        tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = datasets.ImageFolder(f'{dataset_root}/train', transform=basic_transform)
    val_dataset = datasets.ImageFolder(f'{dataset_root}/val', transform=basic_transform)
    test_dataset = datasets.ImageFolder(f'{dataset_root}/test', transform=basic_transform)

    # Sanity check: class indices must match across splits
    assert train_dataset.class_to_idx == val_dataset.class_to_idx == test_dataset.class_to_idx, \
        "Class index mismatch between splits"

    return train_dataset, val_dataset, test_dataset


def build_augmented_datasets(fast_root):
    """Build train/val/test datasets with Albumentations augmentation pipeline.

    Used in Runs 3-8 (with augmentation). The train set uses the 5-technique
    augmentation pipeline; val and test use only resize + normalize.

    Args:
        fast_root: path to dataset root (typically /content/ip102_fast/classification)

    Returns:
        tuple of (train_dataset_aug, val_dataset_aug, test_dataset_aug)
    """
    train_folder = datasets.ImageFolder(f'{fast_root}/train')
    val_folder = datasets.ImageFolder(f'{fast_root}/val')
    test_folder = datasets.ImageFolder(f'{fast_root}/test')

    # Sanity check: class indices must match across splits
    assert train_folder.class_to_idx == val_folder.class_to_idx == test_folder.class_to_idx, \
        "Class index mismatch between splits"

    train_dataset_aug = AlbumentationsDataset(train_folder, train_aug)
    val_dataset_aug = AlbumentationsDataset(val_folder, val_aug)
    test_dataset_aug = AlbumentationsDataset(test_folder, val_aug)

    return train_dataset_aug, val_dataset_aug, test_dataset_aug


def compute_class_weights(train_dataset):
    """Inverse-square-root class weights for handling long-tail imbalance.

    Used in Run 4 (class-weighted CrossEntropyLoss).

    Args:
        train_dataset: torchvision ImageFolder dataset

    Returns:
        numpy array of class weights, normalized so mean weight = 1
    """
    class_counts = np.zeros(len(train_dataset.classes), dtype=int)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    class_weights = 1.0 / np.sqrt(class_counts)
    class_weights = class_weights / class_weights.mean()
    return class_weights
