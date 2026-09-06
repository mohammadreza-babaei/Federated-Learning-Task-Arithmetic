"""Centralized CIFAR-100 training, evaluation, logging, and checkpointing."""

from pathlib import Path
# Path provides convenient and platform-independent file and directory handling.

from typing import Dict, List, Optional
# Type hints used for dictionaries, lists, and optional values.

import csv
# Used to save training metrics into CSV files.

import torch
# Main PyTorch library used for tensors, optimization, schedulers, and checkpoints.

import torch.nn as nn
# Provides neural-network modules and loss functions.

from torch.utils.data import DataLoader
# DataLoader provides batched access to train, validation, and test datasets.

from src.trainer import evaluate, train_one_epoch
# Import the project's reusable functions for one training epoch and evaluation.


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
# Define the exact column order used when saving training history to CSV.


def save_history_csv(
    history: List[Dict[str, float]],
    history_path: str,
) -> None:
    """Overwrite a CSV with the complete sequence of epoch metrics."""

    # Convert the provided path string into a Path object.
    path = Path(history_path)

    # Create the parent directory if it does not already exist.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Open the CSV file in write mode, replacing any previous content.
    with path.open("w", newline="", encoding="utf-8") as csv_file:

        # Create a dictionary-based CSV writer using the predefined columns.
        writer = csv.DictWriter(
            csv_file,
            fieldnames=HISTORY_FIELDNAMES,
        )

        # Write the column names as the first row.
        writer.writeheader()

        # Write the complete sequence of epoch metric dictionaries.
        writer.writerows(history)

    # Print the output path for easier experiment tracking.
    print(f"Training history saved to: {path}")


def append_history_row(
    epoch_result: Dict[str, float],
    history_path: str,
) -> None:
    """Append one epoch's metrics, creating the CSV header when necessary."""

    # Convert the history path into a Path object.
    path = Path(history_path)

    # Ensure the destination directory exists.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Check whether a non-empty CSV file already exists.
    file_exists = path.exists() and path.stat().st_size > 0

    # Open the CSV file in append mode so previous epochs are preserved.
    with path.open("a", newline="", encoding="utf-8") as csv_file:

        # Create a dictionary-based CSV writer.
        writer = csv.DictWriter(
            csv_file,
            fieldnames=HISTORY_FIELDNAMES,
        )

        # If this is a new or empty file, write the column names first.
        if not file_exists:
            writer.writeheader()

        # Append the metrics of the current epoch.
        writer.writerow(epoch_result)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    scheduler_step_size: int,
    scheduler_gamma: float,
):
    """Construct the configured learning-rate scheduler."""

    # Convert the scheduler name to lowercase so input is case-insensitive.
    normalized_name = scheduler_name.lower()

    # Use cosine annealing to gradually reduce the learning rate.
    if normalized_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,

            # T_max defines the length of one cosine annealing cycle.
            T_max=epochs,

            # eta_min is the minimum learning rate reached by the scheduler.
            eta_min=0.0,
        )

    # Use StepLR to reduce the learning rate at fixed epoch intervals.
    if normalized_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,

            # Number of epochs between learning-rate reductions.
            step_size=scheduler_step_size,

            # Multiplicative factor applied to the learning rate.
            gamma=scheduler_gamma,
        )

    # Keep the learning rate constant when no scheduler is requested.
    if normalized_name == "none":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,

            # Returning 1.0 leaves the optimizer learning rate unchanged.
            lr_lambda=lambda _: 1.0,
        )

    # Reject unsupported scheduler names.
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
    """Persist model, optimizer, scheduler, metrics, history, and configuration.

    The file supports later evaluation and continuation from the next
    centralized epoch.
    """

    # Convert the checkpoint path into a Path object.
    path = Path(checkpoint_path)

    # Create the checkpoint directory if necessary.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Retrieve the most recent recorded metrics if history is available.
    latest_metrics = history[-1] if history else {}

    # Save all state required to reproduce or resume the training run.
    torch.save(
        {
            # Store the most recently completed epoch.
            "epoch": epoch,

            # Store all model parameters and buffers.
            "model_state_dict": model.state_dict(),

            # Store optimizer state such as momentum buffers.
            "optimizer_state_dict": optimizer.state_dict(),

            # Store scheduler state so learning-rate progression can resume correctly.
            "scheduler_state_dict": scheduler.state_dict(),

            # Keep track of the best validation accuracy reached so far.
            "best_validation_accuracy": best_validation_accuracy,

            # Store the most recent validation loss.
            "validation_loss": latest_metrics.get("validation_loss"),

            # Store the most recent validation accuracy.
            "validation_accuracy": latest_metrics.get(
                "validation_accuracy"
            ),

            # Store the most recent test loss.
            "test_loss": latest_metrics.get("test_loss"),

            # Store the most recent test accuracy.
            "test_accuracy": latest_metrics.get("test_accuracy"),

            # Store the complete training history.
            "history": history,

            # Store hyperparameters and scheduler configuration.
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

    Args:
        model: Classification model to optimize.
        train_loader, validation_loader, test_loader: DataLoaders for the
            three evaluation roles.
        device: CPU or CUDA device used for computation.
        epochs: Total number of epochs, including any resumed epochs.
        resume_path: Optional checkpoint from which to continue.

    Returns the history, final test metrics, best epoch, and checkpoint paths.
    """

    # The total number of training epochs must be positive.
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")

    # The learning rate must be strictly positive.
    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than zero."
        )

    # Negative momentum is not allowed.
    if momentum < 0:
        raise ValueError("momentum cannot be negative.")

    # Negative weight decay is not valid.
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")

    # StepLR requires a positive interval between learning-rate updates.
    if scheduler_step_size <= 0:
        raise ValueError(
            "scheduler_step_size must be greater than zero."
        )

    # Gamma must reduce or preserve the learning rate, but cannot be zero.
    if not 0.0 < scheduler_gamma <= 1.0:
        raise ValueError(
            "scheduler_gamma must be in the interval (0, 1]."
        )

    # Logging frequency must be positive.
    if log_interval <= 0:
        raise ValueError(
            "log_interval must be greater than zero."
        )

    # Move the model to the selected CPU or CUDA device.
    model = model.to(device)

    # CrossEntropyLoss is used for the CIFAR-100 multiclass classification task.
    criterion = nn.CrossEntropyLoss()

    # Keep only model parameters that are allowed to be updated.
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    # Stop if the model contains no trainable parameters.
    if not trainable_parameters:
        raise RuntimeError(
            "The model has no trainable parameters."
        )

    # Create the SGDM optimizer required for centralized training.
    optimizer = torch.optim.SGD(
        trainable_parameters,

        # Base learning rate controlling the update size.
        lr=learning_rate,

        # Momentum accelerates updates using previous gradients.
        momentum=momentum,

        # Weight decay applies L2-style regularization.
        weight_decay=weight_decay,
    )

    # Build the selected learning-rate scheduler.
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        epochs=epochs,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
    )

    # Store the main hyperparameters so they are saved inside checkpoints.
    training_config: Dict[str, object] = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "scheduler_name": scheduler_name,
        "scheduler_step_size": scheduler_step_size,
        "scheduler_gamma": scheduler_gamma,
    }

    # Initialize an empty list that will contain one metrics dictionary per epoch.
    history: List[Dict[str, float]] = []

    # Start below any possible accuracy so the first result can become the best.
    best_validation_accuracy = -1.0

    # Training normally starts from epoch 1.
    start_epoch = 1

    # Resume restores all state needed for epoch and scheduler continuity.
    if resume_path is not None:

        # Convert the resume checkpoint path into a Path object.
        resume_file = Path(resume_path)

        # The requested checkpoint must exist before loading.
        if not resume_file.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )

        # Load the complete checkpoint onto the selected device.
        checkpoint = torch.load(
            resume_file,
            map_location=device,
            weights_only=False,
        )

        # Restore model parameters and buffers.
        model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer state, including momentum information.
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        # Restore the scheduler so the learning rate continues correctly.
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        # Continue from the epoch after the last completed one.
        start_epoch = int(checkpoint["epoch"]) + 1

        # Restore the best validation accuracy recorded so far.
        best_validation_accuracy = float(
            checkpoint.get(
                "best_validation_accuracy",
                -1.0,
            )
        )

        # Restore all previously recorded epoch metrics.
        history = checkpoint.get("history", [])

        print(f"\nResumed training from: {resume_path}")
        print(f"Last completed epoch: {start_epoch - 1}")
        print(
            f"Training will continue from epoch "
            f"{start_epoch}/{epochs}"
        )

        # Rewrite the CSV using the history stored in the checkpoint.
        save_history_csv(
            history=history,
            history_path=history_path,
        )
    else:
        # A new run must not append to an old CSV accidentally.

        # Locate the history file of a possible previous run.
        history_file = Path(history_path)

        # Remove the previous CSV so this run starts with a clean history.
        if history_file.exists():
            history_file.unlink()

    # Warn when the loaded checkpoint already completed the requested number of epochs.
    if start_epoch > epochs:
        print(
            "\nThe checkpoint has already completed "
            f"epoch {start_epoch - 1}."
        )
        print(f"Requested total epochs: {epochs}")

    # Run every remaining epoch from start_epoch up to the requested total.
    for epoch in range(start_epoch, epochs + 1):

        # Read the optimizer's current learning rate before training this epoch.
        current_learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        print(f"\nStarting epoch {epoch}/{epochs}")
        print(
            "Current learning rate: "
            f"{current_learning_rate:.8f}"
        )

        # Train the model for one complete epoch.
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
            log_interval=log_interval,
        )

        # Measure performance on validation data.
        # These metrics are used to select the best checkpoint.
        validation_metrics = evaluate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
            max_batches=max_validation_batches,
            log_interval=log_interval,
        )

        # Measure performance on test data for reporting and final plots.
        test_metrics = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            max_batches=max_test_batches,
            log_interval=log_interval,
        )

        # Collect all metrics from the current epoch into one dictionary.
        epoch_result = {
            "epoch": float(epoch),

            # Store the learning rate used during this epoch.
            "learning_rate": current_learning_rate,

            # Store training loss.
            "train_loss": float(train_metrics["loss"]),

            # Store training accuracy.
            "train_accuracy": float(
                train_metrics["accuracy"]
            ),

            # Store validation loss.
            "validation_loss": float(
                validation_metrics["loss"]
            ),

            # Store validation accuracy.
            "validation_accuracy": float(
                validation_metrics["accuracy"]
            ),

            # Store test loss.
            "test_loss": float(test_metrics["loss"]),

            # Store test accuracy.
            "test_accuracy": float(
                test_metrics["accuracy"]
            ),
        }

        # Add the current epoch results to the in-memory history.
        history.append(epoch_result)

        # Print a compact summary of all important metrics for this epoch.
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

        # Immediately append the current epoch to the CSV history file.
        append_history_row(
            epoch_result=epoch_result,
            history_path=history_path,
        )

        # Validation selects the best checkpoint; test data remains evaluation-only.
        if (
            validation_metrics["accuracy"]
            > best_validation_accuracy
        ):

            # Update the best validation accuracy reached so far.
            best_validation_accuracy = float(
                validation_metrics["accuracy"]
            )

            # Save a special checkpoint whenever validation accuracy improves.
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

        # Advance the learning-rate scheduler after the current epoch.
        scheduler.step()

        # Save the most recent training state every epoch.
        # This checkpoint is used for recovery if training is interrupted.
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

    # Locate the checkpoint corresponding to the best validation accuracy.
    best_checkpoint_file = Path(checkpoint_path)

    # Final evaluation cannot continue if the best checkpoint is missing.
    if not best_checkpoint_file.exists():
        raise FileNotFoundError(
            "Best checkpoint was not found. "
            f"Expected path: {checkpoint_path}"
        )

    # Load the best checkpoint selected using validation accuracy.
    best_checkpoint = torch.load(
        best_checkpoint_file,
        map_location=device,
        weights_only=False,
    )

    # Replace the current model state with the best saved model state.
    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    print("\nLoaded best checkpoint.")
    print("Starting final test evaluation")

    # Perform the final test evaluation using the best validation-selected model.
    final_test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        max_batches=max_test_batches,
        log_interval=log_interval,
    )

    # Print the final test performance.
    print(
        f"Final test loss: "
        f"{final_test_metrics['loss']:.4f} | "
        f"Final test accuracy: "
        f"{final_test_metrics['accuracy']:.4f}"
    )

    # Rewrite the final CSV with the complete training history.
    save_history_csv(
        history=history,
        history_path=history_path,
    )

    # Return all important results and paths for later reporting or analysis.
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