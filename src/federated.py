from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.client import FederatedClient
from src.masks import (
    build_highest_magnitude_mask,
    build_least_sensitive_mask,
    build_lowest_magnitude_mask,
    build_most_sensitive_mask,
    build_random_mask,
    calculate_mask_statistics,
)
from src.sensitivity import (
    average_sensitivity_scores,
    compute_fisher_sensitivity,
)
from src.server import FederatedServer
from src.trainer import evaluate


SUPPORTED_MASK_STRATEGIES = {
    "least_sensitive",
    "most_sensitive",
    "lowest_magnitude",
    "highest_magnitude",
    "random",
}


def save_training_checkpoint(
    global_model: nn.Module,
    current_round: int,
    best_validation_accuracy: float,
    history: List[Dict[str, object]],
    checkpoint_path: str | Path,
    generator: torch.Generator,
    gradient_mask: Optional[Dict[str, torch.Tensor]] = None,
    mask_statistics: Optional[Dict[str, object]] = None,
) -> None:
    """
    Save the latest federated training state.

    This checkpoint is used to resume training from the next
    communication round.
    """

    path = Path(checkpoint_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_data = {
        "round": current_round,
        "model_state_dict": global_model.state_dict(),
        "best_validation_accuracy": (
            best_validation_accuracy
        ),
        "history": history,
        "generator_state": generator.get_state(),
        "mask_statistics": mask_statistics,
    }

    if gradient_mask is not None:
        checkpoint_data["gradient_mask"] = {
            name: mask.detach().cpu()
            for name, mask in gradient_mask.items()
        }

    torch.save(
        checkpoint_data,
        path,
    )


def build_gradient_mask(
    model: nn.Module,
    client_loaders: List[DataLoader],
    device: torch.device,
    mask_strategy: str,
    sparsity_ratio: float,
    calibration_rounds: int,
    calibration_batches: Optional[int],
    seed: int,
) -> Dict[str, torch.Tensor]:
    """
    Build a gradient mask for sparse fine-tuning.

    Sensitivity-based strategies use the diagonal Fisher approximation.
    Magnitude-based and random strategies do not require Fisher scores.
    """

    if mask_strategy not in SUPPORTED_MASK_STRATEGIES:
        raise ValueError(
            f"Unsupported mask strategy: {mask_strategy}. "
            f"Available strategies: "
            f"{sorted(SUPPORTED_MASK_STRATEGIES)}"
        )

    if not 0.0 < sparsity_ratio <= 1.0:
        raise ValueError(
            "sparsity_ratio must be greater than 0 "
            "and at most 1."
        )

    if calibration_rounds <= 0:
        raise ValueError(
            "calibration_rounds must be greater than zero."
        )

    if (
        calibration_batches is not None
        and calibration_batches <= 0
    ):
        raise ValueError(
            "calibration_batches must be greater than zero "
            "or None."
        )

    print("\n========== Gradient-mask calibration ==========")
    print(f"Mask strategy: {mask_strategy}")
    print(f"Active parameter ratio: {sparsity_ratio}")

    if mask_strategy == "lowest_magnitude":
        return build_lowest_magnitude_mask(
            model=model,
            sparsity_ratio=sparsity_ratio,
        )

    if mask_strategy == "highest_magnitude":
        return build_highest_magnitude_mask(
            model=model,
            sparsity_ratio=sparsity_ratio,
        )

    if mask_strategy == "random":
        return build_random_mask(
            model=model,
            sparsity_ratio=sparsity_ratio,
            seed=seed,
        )

    criterion = nn.CrossEntropyLoss()

    sensitivity_rounds = []

    for calibration_round in range(calibration_rounds):
        calibration_client_index = (
            calibration_round % len(client_loaders)
        )

        calibration_loader = client_loaders[
            calibration_client_index
        ]

        print(
            f"Calibration round "
            f"{calibration_round + 1}/{calibration_rounds} | "
            f"Client: {calibration_client_index}"
        )

        round_scores = compute_fisher_sensitivity(
            model=model,
            dataloader=calibration_loader,
            criterion=criterion,
            device=device,
            max_batches=calibration_batches,
        )

        sensitivity_rounds.append(round_scores)

    averaged_sensitivity = average_sensitivity_scores(
        sensitivity_rounds
    )

    model.zero_grad(set_to_none=True)

    if mask_strategy == "least_sensitive":
        return build_least_sensitive_mask(
            sensitivity_scores=averaged_sensitivity,
            sparsity_ratio=sparsity_ratio,
        )

    return build_most_sensitive_mask(
        sensitivity_scores=averaged_sensitivity,
        sparsity_ratio=sparsity_ratio,
    )


def train_federated(
    global_model: nn.Module,
    client_loaders: List[DataLoader],
    validation_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    rounds: int = 5,
    local_steps: int = 4,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    client_fraction: float = 0.1,
    seed: int = 42,
    checkpoint_path: str = (
        "experiments/checkpoints/best_federated.pt"
    ),
    use_sparse_sgd: bool = False,
    mask_strategy: str = "least_sensitive",
    sparsity_ratio: float = 0.1,
    calibration_rounds: int = 1,
    calibration_batches: Optional[int] = 1,
    resume_path: Optional[str] = None,
    last_checkpoint_path: str = (
        "experiments/checkpoints/last_federated.pt"
    ),
) -> Dict[str, object]:
    """
    Train a model using FedAvg with random client sampling.

    When use_sparse_sgd is True, a gradient mask is calibrated before
    federated training and every client uses SparseSGDM.

    local_steps corresponds to J in the project specification.
    Each selected client performs exactly J mini-batch updates.

    When resume_path is provided, the global model, history,
    best validation accuracy and client-sampling generator state
    are restored. Training continues from the next round.
    """

    if rounds <= 0:
        raise ValueError(
            "rounds must be greater than zero."
        )

    if local_steps <= 0:
        raise ValueError(
            "local_steps must be greater than zero."
        )

    if not 0.0 < client_fraction <= 1.0:
        raise ValueError(
            "client_fraction must be greater than 0 "
            "and at most 1."
        )

    if not client_loaders:
        raise ValueError(
            "client_loaders cannot be empty."
        )

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    global_model = global_model.to(device)

    checkpoint_path_object = Path(checkpoint_path)
    last_checkpoint_path_object = Path(
        last_checkpoint_path
    )

    checkpoint_path_object.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    last_checkpoint_path_object.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history: List[Dict[str, object]] = []
    best_validation_accuracy = -1.0
    start_round = 0

    gradient_mask = None
    mask_statistics = None

    generator = torch.Generator()
    generator.manual_seed(seed)

    # =========================================================
    # Load resume checkpoint
    # =========================================================

    if resume_path is not None:
        resume_checkpoint_path = Path(resume_path)

        if not resume_checkpoint_path.exists():
            raise FileNotFoundError(
                "Federated resume checkpoint does not exist: "
                f"{resume_checkpoint_path}"
            )

        resume_checkpoint = torch.load(
            resume_checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        required_keys = {
            "round",
            "model_state_dict",
            "best_validation_accuracy",
            "history",
        }

        missing_keys = required_keys.difference(
            resume_checkpoint.keys()
        )

        if missing_keys:
            raise KeyError(
                "The federated resume checkpoint is missing "
                f"required keys: {sorted(missing_keys)}"
            )

        global_model.load_state_dict(
            resume_checkpoint["model_state_dict"]
        )

        completed_round = int(
            resume_checkpoint["round"]
        )

        start_round = completed_round

        best_validation_accuracy = float(
            resume_checkpoint[
                "best_validation_accuracy"
            ]
        )

        history = list(
            resume_checkpoint["history"]
        )

        if "generator_state" in resume_checkpoint:
            generator_state = resume_checkpoint[
                "generator_state"
            ]

            if not isinstance(
                generator_state,
                torch.ByteTensor,
            ):
                generator_state = torch.tensor(
                    generator_state,
                    dtype=torch.uint8,
                )

            generator.set_state(
                generator_state.cpu()
            )

        else:
            for _ in range(completed_round):
                torch.randperm(
                    len(client_loaders),
                    generator=generator,
                )

        if use_sparse_sgd:
            if "gradient_mask" in resume_checkpoint:
                gradient_mask = {
                    name: mask.to(device)
                    for name, mask
                    in resume_checkpoint[
                        "gradient_mask"
                    ].items()
                }

                mask_statistics = resume_checkpoint.get(
                    "mask_statistics"
                )

                if mask_statistics is None:
                    mask_statistics = (
                        calculate_mask_statistics(
                            gradient_mask
                        )
                    )

            else:
                print(
                    "\nThe resume checkpoint does not contain "
                    "a gradient mask."
                )

                print(
                    "Rebuilding the gradient mask for "
                    "SparseSGDM compatibility..."
                )

                gradient_mask = build_gradient_mask(
                    model=global_model,
                    client_loaders=client_loaders,
                    device=device,
                    mask_strategy=mask_strategy,
                    sparsity_ratio=sparsity_ratio,
                    calibration_rounds=calibration_rounds,
                    calibration_batches=calibration_batches,
                    seed=seed,
                )

                mask_statistics = (
                    calculate_mask_statistics(
                        gradient_mask
                    )
                )

        print(
            "\nResumed federated training from: "
            f"{resume_checkpoint_path}"
        )

        print(
            f"Last completed round: "
            f"{completed_round}"
        )

        if start_round < rounds:
            print(
                "Training will continue from round "
                f"{start_round + 1}/{rounds}"
            )
        else:
            print(
                "The checkpoint has already completed "
                f"{completed_round} rounds."
            )

    # =========================================================
    # Build sparse gradient mask for new runs
    # =========================================================

    if use_sparse_sgd and gradient_mask is None:
        gradient_mask = build_gradient_mask(
            model=global_model,
            client_loaders=client_loaders,
            device=device,
            mask_strategy=mask_strategy,
            sparsity_ratio=sparsity_ratio,
            calibration_rounds=calibration_rounds,
            calibration_batches=calibration_batches,
            seed=seed,
        )

        mask_statistics = calculate_mask_statistics(
            gradient_mask
        )

    if use_sparse_sgd:
        print(
            "Gradient mask ready | "
            f"Active entries: "
            f"{mask_statistics['active_entries']}/"
            f"{mask_statistics['total_entries']} | "
            f"Active ratio: "
            f"{mask_statistics['active_ratio']:.6f} | "
            f"Zero ratio: "
            f"{mask_statistics['zero_ratio']:.6f}"
        )

    server = FederatedServer(
        global_model=global_model,
        device=device,
    )

    clients = [
        FederatedClient(
            client_id=client_id,
            model=global_model,
            train_loader=loader,
            device=device,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            gradient_mask=gradient_mask,
        )
        for client_id, loader in enumerate(client_loaders)
    ]

    criterion = nn.CrossEntropyLoss()

    num_clients = len(clients)

    clients_per_round = max(
        1,
        int(client_fraction * num_clients),
    )

    optimizer_name = (
        "SparseSGDM"
        if use_sparse_sgd
        else "SGD"
    )

    print("\n========== Federated configuration ==========")
    print(f"Total clients: {num_clients}")
    print(f"Client fraction: {client_fraction}")

    print(
        f"Clients selected per round: "
        f"{clients_per_round}"
    )

    print(f"Local steps per client: {local_steps}")
    print(f"Client optimizer: {optimizer_name}")

    print(
        f"Best checkpoint path: "
        f"{checkpoint_path_object}"
    )

    print(
        f"Last checkpoint path: "
        f"{last_checkpoint_path_object}"
    )

    if use_sparse_sgd:
        print(f"Mask strategy: {mask_strategy}")
        print(f"Active parameter ratio: {sparsity_ratio}")
        print(f"Calibration rounds: {calibration_rounds}")

        print(
            f"Calibration batches: "
            f"{calibration_batches}"
        )

    # =========================================================
    # Federated training rounds
    # =========================================================

    for round_idx in range(start_round, rounds):
        print(
            f"\n========== Round "
            f"{round_idx + 1}/{rounds} =========="
        )

        selected_indices = torch.randperm(
            num_clients,
            generator=generator,
        )[:clients_per_round].tolist()

        selected_clients = [
            clients[index]
            for index in selected_indices
        ]

        selected_client_ids = [
            client.client_id
            for client in selected_clients
        ]

        print(
            "Selected clients: "
            + ", ".join(
                str(client_id)
                for client_id in selected_client_ids
            )
        )

        server.distribute(selected_clients)

        client_models = []
        client_metrics = []

        for client in selected_clients:
            metrics = client.train(
                local_steps=local_steps,
            )

            print(
                f"Client {client.client_id} | "
                f"Optimizer: {metrics['optimizer']} | "
                f"Steps: {metrics['steps']} | "
                f"Samples: {metrics['samples']} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Accuracy: {metrics['accuracy']:.4f}"
            )

            client_models.append(
                client.get_model()
            )

            client_metrics.append(
                {
                    "client_id": client.client_id,
                    "optimizer": metrics["optimizer"],
                    "loss": metrics["loss"],
                    "accuracy": metrics["accuracy"],
                    "steps": metrics["steps"],
                    "samples": metrics["samples"],
                }
            )

        server.aggregate(client_models)

        validation_metrics = evaluate(
            model=server.get_global_model(),
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Validation loss: "
            f"{validation_metrics['loss']:.4f} | "
            f"Validation accuracy: "
            f"{validation_metrics['accuracy']:.4f}"
        )

        if (
            validation_metrics["accuracy"]
            > best_validation_accuracy
        ):
            best_validation_accuracy = (
                validation_metrics["accuracy"]
            )

            checkpoint_data = {
                "round": round_idx + 1,
                "model_state_dict": (
                    server
                    .get_global_model()
                    .state_dict()
                ),
                "validation_loss": (
                    validation_metrics["loss"]
                ),
                "validation_accuracy": (
                    best_validation_accuracy
                ),
                "selected_client_ids": (
                    selected_client_ids
                ),
                "client_fraction": (
                    client_fraction
                ),
                "num_clients": (
                    num_clients
                ),
                "local_steps": (
                    local_steps
                ),
                "seed": (
                    seed
                ),
                "optimizer": (
                    optimizer_name
                ),
                "use_sparse_sgd": (
                    use_sparse_sgd
                ),
                "mask_strategy": (
                    mask_strategy
                    if use_sparse_sgd
                    else None
                ),
                "sparsity_ratio": (
                    sparsity_ratio
                    if use_sparse_sgd
                    else None
                ),
                "calibration_rounds": (
                    calibration_rounds
                    if use_sparse_sgd
                    else None
                ),
                "calibration_batches": (
                    calibration_batches
                    if use_sparse_sgd
                    else None
                ),
                "mask_statistics": (
                    mask_statistics
                ),
            }

            if gradient_mask is not None:
                checkpoint_data["gradient_mask"] = {
                    name: mask.detach().cpu()
                    for name, mask
                    in gradient_mask.items()
                }

            torch.save(
                checkpoint_data,
                checkpoint_path_object,
            )

            print(
                f"Best checkpoint saved to: "
                f"{checkpoint_path_object}"
            )

        round_result = {
            "round": round_idx + 1,
            "validation_loss": (
                validation_metrics["loss"]
            ),
            "validation_accuracy": (
                validation_metrics["accuracy"]
            ),
            "selected_client_ids": (
                selected_client_ids
            ),
            "client_metrics": (
                client_metrics
            ),
        }

        history.append(round_result)

        save_training_checkpoint(
            global_model=server.get_global_model(),
            current_round=round_idx + 1,
            best_validation_accuracy=(
                best_validation_accuracy
            ),
            history=history,
            checkpoint_path=(
                last_checkpoint_path_object
            ),
            generator=generator,
            gradient_mask=gradient_mask,
            mask_statistics=mask_statistics,
        )

        print(
            f"Last checkpoint saved to: "
            f"{last_checkpoint_path_object}"
        )

    # =========================================================
    # Load best checkpoint for final test evaluation
    # =========================================================

    if not checkpoint_path_object.exists():
        raise FileNotFoundError(
            "The best federated checkpoint was not found: "
            f"{checkpoint_path_object}"
        )

    checkpoint = torch.load(
        checkpoint_path_object,
        map_location=device,
        weights_only=False,
    )

    server.get_global_model().load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "\nLoaded best federated checkpoint "
        "for final test evaluation."
    )

    test_metrics = evaluate(
        model=server.get_global_model(),
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(
        f"Test loss: {test_metrics['loss']:.4f} | "
        f"Test accuracy: {test_metrics['accuracy']:.4f}"
    )

    return {
        "global_model": server.get_global_model(),
        "history": history,
        "best_validation_accuracy": (
            best_validation_accuracy
        ),
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "checkpoint_path": str(
            checkpoint_path_object
        ),
        "last_checkpoint_path": str(
            last_checkpoint_path_object
        ),
        "optimizer": optimizer_name,
        "gradient_mask": gradient_mask,
        "mask_statistics": mask_statistics,
    }