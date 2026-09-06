"""Command-line entry point for CIFAR-100 centralized and federated runs."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from data import (
    build_cifar100_dataloaders,
    build_client_dataloaders,
)
from models import build_dino_vits16_cifar100
from src import train_centralized
from src.federated import train_federated


def parse_args() -> argparse.Namespace:
    """Parse dataset, model, optimization, and experiment CLI settings."""

    parser = argparse.ArgumentParser(
        description=(
            "DINO ViT-S/16 on CIFAR-100 "
            "(Centralized / Federated / Sparse Federated)"
        )
    )

    # =========================================================
    # General arguments
    # =========================================================

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Directory used to store CIFAR-100.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="centralized",
        choices=["centralized", "federated"],
        help="Training mode.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help=(
            "Device used for training. "
            "'auto' selects CUDA when available."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    # =========================================================
    # Dataset arguments
    # =========================================================

    parser.add_argument(
        "--partition",
        type=str,
        default="iid",
        choices=["iid", "non_iid"],
        help="Client data partitioning method.",
    )

    parser.add_argument(
        "--classes-per-client",
        type=int,
        default=None,
        choices=[1, 5, 10, 50],
        help=(
            "Number of classes assigned to each client "
            "in non-IID mode."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training and evaluation batch size.",
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help=(
            "Fraction of the original training set "
            "used for validation."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes.",
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download CIFAR-100 if it is "
            "not available locally."
        ),
    )

    # =========================================================
    # Model arguments
    # =========================================================

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use official pretrained DINO weights.",
    )

    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the CIFAR-100 classification head.",
    )

    # =========================================================
    # Centralized training arguments
    # =========================================================

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Total number of centralized training epochs.",
    )

    # =========================================================
    # Federated Learning arguments
    # =========================================================

    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="Total number of federated communication rounds.",
    )

    parser.add_argument(
        "--local-steps",
        type=int,
        default=4,
        choices=[4, 8, 16],
        help=(
            "Number of local optimization steps "
            "per selected client."
        ),
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=100,
        help="Total number of federated clients.",
    )

    parser.add_argument(
        "--client-fraction",
        type=float,
        default=0.1,
        help=(
            "Fraction of clients selected in each round. "
            "For example, 0.1 with 100 clients selects "
            "10 clients."
        ),
    )

    # =========================================================
    # Optimizer arguments
    # =========================================================

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="SGDM or SparseSGDM learning rate.",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGDM or SparseSGDM momentum.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Optimizer weight decay.",
    )

    parser.add_argument(
        "--scheduler-name",
        type=str,
        default="cosine",
        choices=["cosine", "step", "none"],
        help="Learning-rate scheduler name.",
    )

    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=10,
        help="Step size used by the step scheduler.",
    )

    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.1,
        help="Decay factor used by the step scheduler.",
    )

    # =========================================================
    # Centralized checkpoint arguments
    # =========================================================

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=(
            "experiments/checkpoints/"
            "best_centralized.pt"
        ),
        help="Path for the best centralized checkpoint.",
    )

    parser.add_argument(
        "--last-checkpoint-path",
        type=str,
        default=(
            "experiments/checkpoints/"
            "last_centralized.pt"
        ),
        help=(
            "Path for the latest centralized checkpoint. "
            "This checkpoint is saved after every epoch."
        ),
    )

    parser.add_argument(
        "--history-path",
        type=str,
        default="experiments/centralized_history.csv",
        help="Path for centralized epoch history CSV.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume centralized training from a checkpoint. "
            "For example: "
            "experiments/checkpoints/last_centralized.pt"
        ),
    )

    # =========================================================
    # Federated checkpoint arguments
    # =========================================================

    parser.add_argument(
        "--federated-checkpoint-path",
        type=str,
        default=(
            "experiments/checkpoints/"
            "best_federated.pt"
        ),
        help="Path for the best federated checkpoint.",
    )

    parser.add_argument(
        "--last-federated-checkpoint-path",
        type=str,
        default=(
            "experiments/checkpoints/"
            "last_federated.pt"
        ),
        help=(
            "Path for the latest federated checkpoint. "
            "This checkpoint is saved after every round."
        ),
    )

    parser.add_argument(
        "--federated-resume",
        type=str,
        default=None,
        help=(
            "Resume federated training from a checkpoint. "
            "For example: "
            "experiments/checkpoints/last_federated.pt"
        ),
    )

    # =========================================================
    # SparseSGDM and gradient-mask arguments
    # =========================================================

    parser.add_argument(
        "--use-sparse-sgd",
        action="store_true",
        help=(
            "Use gradient masks and SparseSGDM "
            "during federated training."
        ),
    )

    parser.add_argument(
        "--mask-strategy",
        type=str,
        default="least_sensitive",
        choices=[
            "least_sensitive",
            "most_sensitive",
            "lowest_magnitude",
            "highest_magnitude",
            "random",
        ],
        help="Strategy used to select active parameters.",
    )

    parser.add_argument(
        "--sparsity-ratio",
        type=float,
        default=0.1,
        help=(
            "Fraction of trainable parameters that remain active. "
            "For example, 0.1 means 10 percent active."
        ),
    )

    parser.add_argument(
        "--calibration-rounds",
        type=int,
        default=1,
        help=(
            "Number of sensitivity-calibration rounds "
            "used before federated training."
        ),
    )

    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=1,
        help=(
            "Number of mini-batches used in each "
            "Fisher calibration round."
        ),
    )

    # =========================================================
    # Smoke-test arguments
    # =========================================================

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help=(
            "Limit centralized training batches "
            "for a smoke test."
        ),
    )

    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=None,
        help=(
            "Limit validation batches "
            "for a smoke test."
        ),
    )

    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
        help=(
            "Limit test batches "
            "for a smoke test."
        ),
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Print centralized progress every N batches.",
    )

    return parser.parse_args()


def select_device(
    requested_device: str,
) -> torch.device:
    """Resolve the requested execution device, validating explicit CUDA use."""

    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but CUDA is not available."
            )

        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch sources of experiment randomness."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_checkpoint_directories(
    args: argparse.Namespace,
) -> None:
    """Create parent directories for histories and checkpoint files."""

    output_paths = [
        Path(args.checkpoint_path),
        Path(args.last_checkpoint_path),
        Path(args.history_path),
        Path(args.federated_checkpoint_path),
        Path(args.last_federated_checkpoint_path),
    ]

    for output_path in output_paths:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


def validate_optional_positive_integer(
    value: int | None,
    argument_name: str,
) -> None:
    """
    Validate an optional positive integer argument.

    None means that no limit was requested.
    """

    if value is not None and value <= 0:
        raise ValueError(
            f"{argument_name} must be greater than zero "
            "or omitted."
        )


def validate_resume_path(
    resume_path: str | None,
    argument_name: str,
) -> None:
    """Validate an optional resume-checkpoint path."""

    if resume_path is None:
        return

    path = Path(resume_path)

    if not path.exists():
        raise FileNotFoundError(
            f"{argument_name} checkpoint not found: "
            f"{resume_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{argument_name} must point to a file: "
            f"{resume_path}"
        )


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    """Validate argument ranges and mode-specific combinations."""

    if args.num_clients <= 0:
        raise ValueError(
            "--num-clients must be greater than zero."
        )

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be greater than zero."
        )

    if args.rounds <= 0:
        raise ValueError(
            "--rounds must be greater than zero."
        )

    if args.local_steps <= 0:
        raise ValueError(
            "--local-steps must be greater than zero."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than zero."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be greater than zero."
        )

    if args.momentum < 0:
        raise ValueError(
            "--momentum cannot be negative."
        )

    if args.weight_decay < 0:
        raise ValueError(
            "--weight-decay cannot be negative."
        )

    if args.scheduler_step_size <= 0:
        raise ValueError(
            "--scheduler-step-size must be greater than zero."
        )

    if not 0 < args.scheduler_gamma <= 1:
        raise ValueError(
            "--scheduler-gamma must be between 0 and 1."
        )

    if args.num_workers < 0:
        raise ValueError(
            "--num-workers cannot be negative."
        )

    if args.log_interval <= 0:
        raise ValueError(
            "--log-interval must be greater than zero."
        )

    if not 0.0 < args.client_fraction <= 1.0:
        raise ValueError(
            "--client-fraction must be greater than 0 "
            "and at most 1."
        )

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError(
            "--val-fraction must be between 0 and 1."
        )

    if not 0.0 < args.sparsity_ratio <= 1.0:
        raise ValueError(
            "--sparsity-ratio must be greater than 0 "
            "and at most 1."
        )

    if args.calibration_rounds <= 0:
        raise ValueError(
            "--calibration-rounds must be "
            "greater than zero."
        )

    if args.calibration_batches <= 0:
        raise ValueError(
            "--calibration-batches must be "
            "greater than zero."
        )

    selected_clients_per_round = max(
        1,
        int(
            args.num_clients
            * args.client_fraction
        ),
    )

    if selected_clients_per_round > args.num_clients:
        raise ValueError(
            "The number of selected clients cannot exceed "
            "the total number of clients."
        )

    validate_optional_positive_integer(
        args.max_train_batches,
        "--max-train-batches",
    )

    validate_optional_positive_integer(
        args.max_validation_batches,
        "--max-validation-batches",
    )

    validate_optional_positive_integer(
        args.max_test_batches,
        "--max-test-batches",
    )

    if (
        args.mode == "federated"
        and args.partition == "non_iid"
        and args.classes_per_client is None
    ):
        raise ValueError(
            "--classes-per-client is required when "
            "--partition non_iid is selected."
        )

    if (
        args.mode == "federated"
        and args.partition == "iid"
        and args.classes_per_client is not None
    ):
        raise ValueError(
            "--classes-per-client must be omitted "
            "when --partition iid is selected."
        )

    if (
        args.mode == "centralized"
        and args.use_sparse_sgd
    ):
        raise ValueError(
            "--use-sparse-sgd is supported only "
            "in federated mode."
        )

    if (
        args.mode == "federated"
        and args.resume is not None
    ):
        raise ValueError(
            "--resume is supported only "
            "in centralized mode."
        )

    if (
        args.mode == "centralized"
        and args.federated_resume is not None
    ):
        raise ValueError(
            "--federated-resume is supported only "
            "in federated mode."
        )

    validate_resume_path(
        args.resume,
        "--resume",
    )

    validate_resume_path(
        args.federated_resume,
        "--federated-resume",
    )


def print_general_configuration(
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Print reproducibility, model, and optimization settings."""

    print("=" * 60)
    print("DINO ViT-S/16 on CIFAR-100")
    print("=" * 60)

    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Momentum: {args.momentum}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Scheduler: {args.scheduler_name}")
    print(f"Scheduler step size: {args.scheduler_step_size}")
    print(f"Scheduler gamma: {args.scheduler_gamma}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Freeze backbone: {args.freeze_backbone}")

    if args.resume is not None:
        print(
            f"Centralized resume checkpoint: "
            f"{args.resume}"
        )

    if args.federated_resume is not None:
        print(
            f"Federated resume checkpoint: "
            f"{args.federated_resume}"
        )


def main() -> None:
    """Build the experiment and dispatch centralized or federated training.

    Both modes share dataset creation and model construction; federated mode
    additionally partitions the training subset into client DataLoaders.
    """

    args = parse_args()

    validate_arguments(args)
    set_random_seed(args.seed)
    create_checkpoint_directories(args)

    device = select_device(args.device)

    print_general_configuration(
        args=args,
        device=device,
    )

    train_loader, validation_loader, test_loader = (
        build_cifar100_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
            download=args.download,
            num_workers=args.num_workers,
        )
    )

    print(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_loader.dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    model = build_dino_vits16_cifar100(
        num_classes=100,
        freeze_backbone=args.freeze_backbone,
        pretrained=args.pretrained,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # Centralized mode uses one optimizer over the complete training subset.

    if args.mode == "centralized":
        print("\nStarting centralized training...")
        print(f"Total epochs: {args.epochs}")

        print(
            "Best checkpoint path: "
            f"{args.checkpoint_path}"
        )

        print(
            "Last checkpoint path: "
            f"{args.last_checkpoint_path}"
        )

        print(
            "History path: "
            f"{args.history_path}"
        )

        results = train_centralized(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            scheduler_name=args.scheduler_name,
            scheduler_step_size=args.scheduler_step_size,
            scheduler_gamma=args.scheduler_gamma,
            checkpoint_path=args.checkpoint_path,
            last_checkpoint_path=(
                args.last_checkpoint_path
            ),
            history_path=args.history_path,
            resume_path=args.resume,
            max_train_batches=(
                args.max_train_batches
            ),
            max_validation_batches=(
                args.max_validation_batches
            ),
            max_test_batches=(
                args.max_test_batches
            ),
            log_interval=args.log_interval,
        )

        print("\nCentralized training completed.")

        print(
            "Best validation accuracy: "
            f"{results['best_validation_accuracy']:.4f}"
        )

        print(
            "Test accuracy: "
            f"{results['test_accuracy']:.4f}"
        )

        print(
            "Best checkpoint: "
            f"{results['checkpoint_path']}"
        )

        print(
            "Last checkpoint: "
            f"{results['last_checkpoint_path']}"
        )

        print(
            "History file: "
            f"{results['history_path']}"
        )

        return

    # Federated mode partitions the training subset before creating clients.

    selected_clients_per_round = max(
        1,
        int(
            args.num_clients
            * args.client_fraction
        ),
    )

    print("\nStarting federated training...")
    print(f"Partition: {args.partition}")
    print(f"Total clients: {args.num_clients}")
    print(f"Client fraction: {args.client_fraction}")

    print(
        "Clients selected per round: "
        f"{selected_clients_per_round}"
    )

    print(f"Communication rounds: {args.rounds}")

    print(
        "Local steps per client: "
        f"{args.local_steps}"
    )

    print(
        "Best federated checkpoint path: "
        f"{args.federated_checkpoint_path}"
    )

    print(
        "Last federated checkpoint path: "
        f"{args.last_federated_checkpoint_path}"
    )

    if args.federated_resume is not None:
        print(
            "Federated resume checkpoint: "
            f"{args.federated_resume}"
        )

    if args.partition == "non_iid":
        print(
            "Classes per client: "
            f"{args.classes_per_client}"
        )

    if args.use_sparse_sgd:
        print("Client optimizer: SparseSGDM")
        print(f"Mask strategy: {args.mask_strategy}")

        print(
            "Active parameter ratio: "
            f"{args.sparsity_ratio}"
        )

        print(
            "Calibration rounds: "
            f"{args.calibration_rounds}"
        )

        print(
            "Calibration batches: "
            f"{args.calibration_batches}"
        )

    else:
        print("Client optimizer: SGD")

    client_loaders = build_client_dataloaders(
        dataset=train_loader.dataset,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        partition=args.partition,
        classes_per_client=args.classes_per_client,
        seed=args.seed,
    )

    results = train_federated(
        global_model=model,
        client_loaders=client_loaders,
        validation_loader=validation_loader,
        test_loader=test_loader,
        device=device,
        rounds=args.rounds,
        local_steps=args.local_steps,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        client_fraction=args.client_fraction,
        seed=args.seed,
        checkpoint_path=(
            args.federated_checkpoint_path
        ),
        use_sparse_sgd=args.use_sparse_sgd,
        mask_strategy=args.mask_strategy,
        sparsity_ratio=args.sparsity_ratio,
        calibration_rounds=(
            args.calibration_rounds
        ),
        calibration_batches=(
            args.calibration_batches
        ),
        resume_path=args.federated_resume,
        last_checkpoint_path=(
            args.last_federated_checkpoint_path
        ),
    )

    print("\nFederated training completed.")

    print(
        f"Optimizer: "
        f"{results['optimizer']}"
    )

    print(
        "Best validation accuracy: "
        f"{results['best_validation_accuracy']:.4f}"
    )

    print(
        "Test accuracy: "
        f"{results['test_accuracy']:.4f}"
    )

    print(
        "Best checkpoint: "
        f"{results['checkpoint_path']}"
    )

    print(
        "Last checkpoint: "
        f"{results['last_checkpoint_path']}"
    )

    mask_statistics = results.get(
        "mask_statistics"
    )

    if mask_statistics is not None:
        print(
            "Mask active entries: "
            f"{mask_statistics['active_entries']}/"
            f"{mask_statistics['total_entries']}"
        )

        print(
            "Mask active ratio: "
            f"{mask_statistics['active_ratio']:.6f}"
        )


if __name__ == "__main__":
    main()