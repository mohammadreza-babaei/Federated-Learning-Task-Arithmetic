"""Reusable mini-batch training and evaluation loops."""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def calculate_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> int:
    """Return the number of correctly classified samples."""

    predictions = logits.argmax(dim=1)
    return int((predictions == targets).sum().item())


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: Optional[int] = None,
    log_interval: int = 10,
) -> Dict[str, float]:
    """Run updates over one epoch and return sample-weighted metrics.

    ``max_batches`` supports short smoke tests without changing full-epoch
    behavior.
    """

    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be greater than zero or None.")

    if log_interval <= 0:
        raise ValueError("log_interval must be greater than zero.")

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (images, targets) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_correct += calculate_accuracy(logits, targets)
        total_samples += batch_size

        if batch_index == 1 or batch_index % log_interval == 0:
            running_loss = total_loss / total_samples
            running_accuracy = total_correct / total_samples

            print(
                f"Train batch {batch_index}/{len(dataloader)} | "
                f"Loss: {running_loss:.4f} | "
                f"Accuracy: {running_accuracy:.4f}"
            )

        if max_batches is not None and batch_index >= max_batches:
            print(f"Stopped training after {max_batches} batches for smoke testing.")
            break

    if total_samples == 0:
        raise RuntimeError("The training DataLoader is empty.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
    log_interval: int = 10,
) -> Dict[str, float]:
    """Evaluate without gradients and return sample-weighted metrics."""

    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be greater than zero or None.")

    if log_interval <= 0:
        raise ValueError("log_interval must be greater than zero.")

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (images, targets) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_correct += calculate_accuracy(logits, targets)
        total_samples += batch_size

        if batch_index == 1 or batch_index % log_interval == 0:
            running_loss = total_loss / total_samples
            running_accuracy = total_correct / total_samples

            print(
                f"Evaluation batch {batch_index}/{len(dataloader)} | "
                f"Loss: {running_loss:.4f} | "
                f"Accuracy: {running_accuracy:.4f}"
            )

        if max_batches is not None and batch_index >= max_batches:
            print(f"Stopped evaluation after {max_batches} batches for smoke testing.")
            break

    if total_samples == 0:
        raise RuntimeError("The evaluation DataLoader is empty.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }