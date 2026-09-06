import os
import random
from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataset import random_split


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_default_cifar100_transforms() -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """Return standard image transforms for train, validation, and test."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        ]
    )

    return train_transform, val_transform, test_transform


def build_cifar100_datasets(
    data_root: str = "./data",
    val_fraction: float = 0.1,
    seed: int = 42,
    download: bool = False,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Build train, validation, and test datasets for CIFAR-100.

    The original training set is split into train and validation subsets.
    The test set is left unchanged and used for final evaluation.
    """
    set_seed(seed)

    train_transform, val_transform, test_transform = get_default_cifar100_transforms()

    train_dataset = torchvision.datasets.CIFAR100(
        root=data_root,
        train=True,
        download=download,
        transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_root,
        train=False,
        download=download,
        transform=test_transform,
    )

    total_train_size = len(train_dataset)
    val_size = int(total_train_size * val_fraction)
    train_size = total_train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform

    return train_subset, val_subset, test_dataset


def build_cifar100_dataloaders(
    data_root: str = "./data",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    seed: int = 42,
    download: bool = False,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders for CIFAR-100."""
    train_dataset, val_dataset, test_dataset = build_cifar100_datasets(
        data_root=data_root,
        val_fraction=val_fraction,
        seed=seed,
        download=download,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
