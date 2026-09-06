from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.trainer import evaluate, train_one_epoch


def save_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_validation_accuracy: float,
    history: List[Dict[str, float]],
    checkpoint_path: str,
) -> None:
    """Save a complete checkpoint for resuming training."""

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_validation_accuracy": best_validation_accuracy,
            "history": history,
        },
        path,
    )


def train_centralized(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    checkpoint_path: str = (
        "experiments/checkpoints/best_model.pt"
    ),
    last_checkpoint_path: str = (
        "experiments/checkpoints/last_centralized.pt"
    ),
    resume_path: Optional[str] = None,
    max_train_batches: Optional[int] = None,
    max_validation_batches: Optional[int] = None,
    max_test_batches: Optional[int] = None,
    log_interval: int = 10,
) -> Dict[str, object]:
    """
    Train and evaluate the centralized CIFAR-100 baseline.

    If resume_path is provided, training continues from the saved epoch.
    """

    if epochs <= 0:
        raise ValueError(
            "epochs must be greater than zero."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than zero."
        )

    if momentum < 0:
        raise ValueError(
            "momentum cannot be negative."
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay cannot be negative."
        )

    if log_interval <= 0:
        raise ValueError(
            "log_interval must be greater than zero."
        )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "The model has no trainable parameters."
        )

    optimizer = torch.optim.SGD(
        trainable_parameters,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=epochs,
        eta_min=0.0,
    )

    history: List[Dict[str, float]] = []
    best_validation_accuracy = -1.0
    start_epoch = 1

    # ---------------------------------------------------------
    # Resume training
    # ---------------------------------------------------------

    if resume_path is not None:
        resume_file = Path(resume_path)

        if not resume_file.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )

        checkpoint = torch.load(
            resume_file,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        start_epoch = int(
            checkpoint["epoch"]
        ) + 1

        best_validation_accuracy = float(
            checkpoint.get(
                "best_validation_accuracy",
                -1.0,
            )
        )

        history = checkpoint.get(
            "history",
            [],
        )

        print(
            f"\nResumed training from: {resume_path}"
        )

        print(
            f"Last completed epoch: "
            f"{start_epoch - 1}"
        )

        print(
            f"Training will continue from epoch "
            f"{start_epoch}/{epochs}"
        )

    if start_epoch > epochs:
        print(
            "\nThe checkpoint has already completed "
            f"epoch {start_epoch - 1}."
        )

        print(
            f"Requested total epochs: {epochs}"
        )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    for epoch in range(start_epoch, epochs + 1):
        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"\nStarting epoch {epoch}/{epochs}"
        )

        print(
            "Current learning rate: "
            f"{current_learning_rate:.8f}"
        )

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
            log_interval=log_interval,
        )

        validation_metrics = evaluate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
            max_batches=max_validation_batches,
            log_interval=log_interval,
        )

        epoch_result = {
            "epoch": float(epoch),
            "learning_rate": float(
                current_learning_rate
            ),
            "train_loss": float(
                train_metrics["loss"]
            ),
            "train_accuracy": float(
                train_metrics["accuracy"]
            ),
            "validation_loss": float(
                validation_metrics["loss"]
            ),
            "validation_accuracy": float(
                validation_metrics["accuracy"]
            ),
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch}/{epochs} completed | "
            f"LR: {current_learning_rate:.8f} | "
            f"Train loss: "
            f"{train_metrics['loss']:.4f} | "
            f"Train accuracy: "
            f"{train_metrics['accuracy']:.4f} | "
            f"Validation loss: "
            f"{validation_metrics['loss']:.4f} | "
            f"Validation accuracy: "
            f"{validation_metrics['accuracy']:.4f}"
        )

        if (
            validation_metrics["accuracy"]
            > best_validation_accuracy
        ):
            best_validation_accuracy = float(
                validation_metrics["accuracy"]
            )

            save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_validation_accuracy=(
                    best_validation_accuracy
                ),
                history=history,
                checkpoint_path=checkpoint_path,
            )

            print(
                f"Best checkpoint saved to: "
                f"{checkpoint_path}"
            )

        scheduler.step()

        # Save after every epoch for resume.
        save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_validation_accuracy=(
                best_validation_accuracy
            ),
            history=history,
            checkpoint_path=last_checkpoint_path,
        )

        print(
            f"Last checkpoint saved to: "
            f"{last_checkpoint_path}"
        )

    # ---------------------------------------------------------
    # Load best model and run final test
    # ---------------------------------------------------------

    best_checkpoint_file = Path(
        checkpoint_path
    )

    if not best_checkpoint_file.exists():
        raise FileNotFoundError(
            "Best checkpoint was not found. "
            f"Expected path: {checkpoint_path}"
        )

    best_checkpoint = torch.load(
        best_checkpoint_file,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    print(
        "\nLoaded best checkpoint."
    )

    print(
        "Starting final test evaluation"
    )

    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        max_batches=max_test_batches,
        log_interval=log_interval,
    )

    print(
        f"Test loss: {test_metrics['loss']:.4f} | "
        f"Test accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    return {
        "history": history,
        "best_epoch": int(
            best_checkpoint["epoch"]
        ),
        "best_validation_accuracy": float(
            best_validation_accuracy
        ),
        "test_loss": float(
            test_metrics["loss"]
        ),
        "test_accuracy": float(
            test_metrics["accuracy"]
        ),
        "checkpoint_path": checkpoint_path,
        "last_checkpoint_path": (
            last_checkpoint_path
        ),
    }