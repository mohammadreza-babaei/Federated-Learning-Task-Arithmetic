from pathlib import Path
from typing import Dict, List, Optional
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.trainer import evaluate, train_one_epoch


HISTORY_FIELDNAMES = [
    "epoch",
    "learning_rate",
    "train_loss",
    "train_accuracy",
    "validation_loss",
    "validation_accuracy",
    "test_loss",
    "test_accuracy",
]


def save_history_csv(
    history: List[Dict[str, float]],
    history_path: str,
) -> None:
    """Write the complete training history to a CSV file."""

    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=HISTORY_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(history)

    print(f"Training history saved to: {path}")


def append_history_row(
    epoch_result: Dict[str, float],
    history_path: str,
) -> None:
    """Append one epoch result to the history CSV."""

    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists() and path.stat().st_size > 0

    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=HISTORY_FIELDNAMES,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(epoch_result)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    scheduler_step_size: int,
    scheduler_gamma: float,
):
    """Create the requested learning-rate scheduler."""

    normalized_name = scheduler_name.lower()

    if normalized_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=epochs,
            eta_min=0.0,
        )

    if normalized_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma,
        )

    if normalized_name == "none":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda _: 1.0,
        )

    raise ValueError(
        "scheduler_name must be one of: cosine, step, none."
    )


def save_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_validation_accuracy: float,
    history: List[Dict[str, float]],
    checkpoint_path: str,
    training_config: Dict[str, object],
) -> None:
    """Save a complete checkpoint for evaluation and resume."""

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    latest_metrics = history[-1] if history else {}

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_validation_accuracy": best_validation_accuracy,
            "validation_loss": latest_metrics.get("validation_loss"),
            "validation_accuracy": latest_metrics.get(
                "validation_accuracy"
            ),
            "test_loss": latest_metrics.get("test_loss"),
            "test_accuracy": latest_metrics.get("test_accuracy"),
            "history": history,
            "training_config": training_config,
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
    scheduler_name: str = "cosine",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    checkpoint_path: str = (
        "experiments/checkpoints/best_model.pt"
    ),
    last_checkpoint_path: str = (
        "experiments/checkpoints/last_centralized.pt"
    ),
    history_path: str = (
        "experiments/centralized_history.csv"
    ),
    resume_path: Optional[str] = None,
    max_train_batches: Optional[int] = None,
    max_validation_batches: Optional[int] = None,
    max_test_batches: Optional[int] = None,
    log_interval: int = 10,
) -> Dict[str, object]:
    """
    Train and evaluate the centralized CIFAR-100 baseline.

    Every epoch is written to a CSV file. The best checkpoint is
    selected only by validation accuracy. Test metrics are recorded
    for final plots but must not be used for hyperparameter selection.
    """

    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than zero."
        )

    if momentum < 0:
        raise ValueError("momentum cannot be negative.")

    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")

    if scheduler_step_size <= 0:
        raise ValueError(
            "scheduler_step_size must be greater than zero."
        )

    if not 0.0 < scheduler_gamma <= 1.0:
        raise ValueError(
            "scheduler_gamma must be in the interval (0, 1]."
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

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        epochs=epochs,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
    )

    training_config: Dict[str, object] = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "scheduler_name": scheduler_name,
        "scheduler_step_size": scheduler_step_size,
        "scheduler_gamma": scheduler_gamma,
    }

    history: List[Dict[str, float]] = []
    best_validation_accuracy = -1.0
    start_epoch = 1

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

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_accuracy = float(
            checkpoint.get(
                "best_validation_accuracy",
                -1.0,
            )
        )
        history = checkpoint.get("history", [])

        print(f"\nResumed training from: {resume_path}")
        print(f"Last completed epoch: {start_epoch - 1}")
        print(
            f"Training will continue from epoch "
            f"{start_epoch}/{epochs}"
        )

        save_history_csv(
            history=history,
            history_path=history_path,
        )
    else:
        # A new run must not append to an old CSV accidentally.
        history_file = Path(history_path)
        if history_file.exists():
            history_file.unlink()

    if start_epoch > epochs:
        print(
            "\nThe checkpoint has already completed "
            f"epoch {start_epoch - 1}."
        )
        print(f"Requested total epochs: {epochs}")

    for epoch in range(start_epoch, epochs + 1):
        current_learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        print(f"\nStarting epoch {epoch}/{epochs}")
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

        test_metrics = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            max_batches=max_test_batches,
            log_interval=log_interval,
        )

        epoch_result = {
            "epoch": float(epoch),
            "learning_rate": current_learning_rate,
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(
                train_metrics["accuracy"]
            ),
            "validation_loss": float(
                validation_metrics["loss"]
            ),
            "validation_accuracy": float(
                validation_metrics["accuracy"]
            ),
            "test_loss": float(test_metrics["loss"]),
            "test_accuracy": float(
                test_metrics["accuracy"]
            ),
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch}/{epochs} completed | "
            f"LR: {current_learning_rate:.8f} | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Train accuracy: "
            f"{train_metrics['accuracy']:.4f} | "
            f"Validation loss: "
            f"{validation_metrics['loss']:.4f} | "
            f"Validation accuracy: "
            f"{validation_metrics['accuracy']:.4f} | "
            f"Test loss: {test_metrics['loss']:.4f} | "
            f"Test accuracy: "
            f"{test_metrics['accuracy']:.4f}"
        )

        append_history_row(
            epoch_result=epoch_result,
            history_path=history_path,
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
                training_config=training_config,
            )

            print(
                f"Best checkpoint saved to: "
                f"{checkpoint_path}"
            )

        scheduler.step()

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
            training_config=training_config,
        )

        print(
            f"Last checkpoint saved to: "
            f"{last_checkpoint_path}"
        )

    best_checkpoint_file = Path(checkpoint_path)

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

    print("\nLoaded best checkpoint.")
    print("Starting final test evaluation")

    final_test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        max_batches=max_test_batches,
        log_interval=log_interval,
    )

    print(
        f"Final test loss: "
        f"{final_test_metrics['loss']:.4f} | "
        f"Final test accuracy: "
        f"{final_test_metrics['accuracy']:.4f}"
    )

    save_history_csv(
        history=history,
        history_path=history_path,
    )

    return {
        "history": history,
        "history_path": history_path,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_accuracy": float(
            best_checkpoint["best_validation_accuracy"]
        ),
        "test_loss": float(final_test_metrics["loss"]),
        "test_accuracy": float(
            final_test_metrics["accuracy"]
        ),
        "checkpoint_path": checkpoint_path,
        "last_checkpoint_path": (
            last_checkpoint_path
        ),
        "training_config": training_config,
    }