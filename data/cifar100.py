
"""CIFAR-100 transforms, deterministic splitting, and DataLoader builders."""

import os

import random

from typing import Tuple

# PyTorch and torchvision provide datasets, tensor operations, and transforms.

import torch

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset, Subset

from torch.utils.data.dataset import random_split


def set_seed(seed: int = 42) -> None:

    """Seed dataset-related Python and PyTorch randomness."""

    # Seeding makes dataset splitting and other random operations repeatable.

    random.seed(seed)

    # Set the random seed for PyTorch operations on the CPU.
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        # CUDA may use a separate random-number generator from the CPU.

        torch.cuda.manual_seed_all(seed)


def get_default_cifar100_transforms() -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:

    """Return training augmentation and deterministic evaluation transforms."""

    # Training uses random augmentation so the model sees varied versions of images.

    train_transform = transforms.Compose(

        [

            # Resize CIFAR-100 images to 224x224 for the DINO ViT-S/16 model.
            transforms.Resize((224, 224)),

            # Randomly crop the image after adding padding.
            # This is a data-augmentation technique used during training.
            transforms.RandomCrop(224, padding=4),

            # Randomly flip images horizontally to increase training-data diversity.
            transforms.RandomHorizontalFlip(),

            # Convert the image into a PyTorch tensor.
            transforms.ToTensor(),

            # Normalize the RGB channels using CIFAR-100 mean and standard deviation.
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),

        ]

    )

    # Validation and test data use only deterministic preprocessing for fair comparison.

    val_transform = transforms.Compose(

        [

            # Resize validation images to the required input size.
            transforms.Resize((224, 224)),

            # Convert validation images into PyTorch tensors.
            transforms.ToTensor(),

            # Normalize validation images using the same CIFAR-100 statistics.
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),

        ]

    )

    # Test preprocessing matches validation preprocessing and does not alter the image randomly.

    test_transform = transforms.Compose(

        [

            # Resize test images to the required input size.
            transforms.Resize((224, 224)),

            # Convert test images into PyTorch tensors.
            transforms.ToTensor(),

            # Normalize test images using the same CIFAR-100 statistics.
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),

        ]

    )

    # Return the three transformation pipelines.
    return train_transform, val_transform, test_transform


def build_cifar100_datasets(

    data_root: str = "./data",

    val_fraction: float = 0.1,

    seed: int = 42,

    download: bool = False,

) -> Tuple[Dataset, Dataset, Dataset]:

    """Build train, validation, and test datasets for CIFAR-100.

    The original training set is split into train and validation subsets.

    The test set is left unchanged and used for final evaluation. A seeded

    ``random_split`` makes train/validation membership reproducible.

    """

    # Set randomness before creating the split so the same seed gives the same samples.

    set_seed(seed)

    # Each subset needs its own transform policy: augmentation for training, stable transforms otherwise.

    train_transform, val_transform, test_transform = get_default_cifar100_transforms()

    # Load CIFAR-100 training images and labels.

    train_dataset = torchvision.datasets.CIFAR100(

        # Directory where CIFAR-100 is stored.
        root=data_root,

        # train=True selects the official CIFAR-100 training set.
        train=True,

        # Download the dataset if requested and it is not already available.
        download=download,

        # Apply the training transformations when an image is loaded.
        transform=train_transform,

    )

    # Load the official test set; it is not split or used for training.

    test_dataset = torchvision.datasets.CIFAR100(

        # Use the same directory as the training dataset.
        root=data_root,

        # train=False selects the official CIFAR-100 test set.
        train=False,

        # Download the dataset if requested.
        download=download,

        # Apply deterministic test preprocessing.
        transform=test_transform,

    )

    # Get the total number of samples in the original training set.
    total_train_size = len(train_dataset)

    # The validation set is carved out of the original training set.

    # Calculate the number of samples assigned to validation.
    val_size = int(total_train_size * val_fraction)

    # All remaining samples are used for training.
    train_size = total_train_size - val_size

    # Use a seeded generator so train/validation membership is reproducible.

    generator = torch.Generator().manual_seed(seed)

    # Randomly split the original training set into train and validation subsets.
    train_subset, val_subset = random_split(

        train_dataset,

        # Define the required sizes of the two subsets.
        [train_size, val_size],

        # Use the seeded generator to obtain the same split across runs.
        generator=generator,

    )

    # random_split creates Subsets that share the original dataset object.

    # Assigning transforms through that shared object gives each subset its intended policy.

    train_subset.dataset.transform = train_transform

    val_subset.dataset.transform = val_transform

    # Return the training subset, validation subset, and official test dataset.
    return train_subset, val_subset, test_dataset


def build_cifar100_dataloaders(

    data_root: str = "./data",

    batch_size: int = 128,

    val_fraction: float = 0.1,

    seed: int = 42,

    download: bool = False,

    num_workers: int = 2,

) -> Tuple[DataLoader, DataLoader, DataLoader]:

    """Wrap the three CIFAR-100 subsets with train/evaluation loader policies."""

    # Build the datasets before wrapping them in loaders.

    train_dataset, val_dataset, test_dataset = build_cifar100_datasets(

        # Location of the CIFAR-100 dataset.
        data_root=data_root,

        # Fraction of the original training set used for validation.
        val_fraction=val_fraction,

        # Seed used to make the split reproducible.
        seed=seed,

        # Control whether CIFAR-100 should be downloaded.
        download=download,

    )

    # Shuffle training batches so optimization does not follow dataset order.

    train_loader = DataLoader(

        # Training subset that provides the samples.
        train_dataset,

        # Number of samples processed together in one batch.
        batch_size=batch_size,

        # Randomize training-sample order during training.
        shuffle=True,

        # Number of worker processes used to load data.
        num_workers=num_workers,

        # Pinned memory can improve CPU-to-GPU data transfer when CUDA is available.
        pin_memory=torch.cuda.is_available(),

    )

    # Keep validation order stable because it is used for consistent monitoring.

    val_loader = DataLoader(

        # Validation subset used to evaluate the model during development.
        val_dataset,

        # Number of validation samples processed in one batch.
        batch_size=batch_size,

        # Validation samples do not need to be shuffled.
        shuffle=False,

        # Number of worker processes used to load validation data.
        num_workers=num_workers,

        # Enable pinned memory when CUDA is available.
        pin_memory=torch.cuda.is_available(),

    )

    # Keep test order stable for reproducible final evaluation.

    test_loader = DataLoader(

        # Official CIFAR-100 test dataset.
        test_dataset,

        # Number of test samples processed in one batch.
        batch_size=batch_size,

        # Test samples are evaluated in a fixed order.
        shuffle=False,

        # Number of worker processes used to load test data.
        num_workers=num_workers,

        # Enable pinned memory when CUDA is available.
        pin_memory=torch.cuda.is_available(),

    )

    # Return DataLoaders for training, validation, and final testing.
    return train_loader, val_loader, test_loader
