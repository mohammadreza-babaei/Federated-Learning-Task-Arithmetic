from pathlib import Path
import csv

import torch
import torch.nn as nn

from data import build_cifar100_dataloaders
from models import build_dino_vits16_cifar100
from src.trainer import evaluate


CHECKPOINT_DIR = Path("experiments/checkpoints")
OUTPUT_PATH = Path("experiments/checkpoint_metrics.csv")


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # Same dataset configuration used in our experiments.
    _, _, test_loader = build_cifar100_dataloaders(
        data_root="./data",
        batch_size=32,
        val_fraction=0.1,
        seed=42,
        download=False,
        num_workers=2,
    )

    checkpoint_paths = sorted(
        CHECKPOINT_DIR.glob("best_*.pt")
    )

    if not checkpoint_paths:
        raise FileNotFoundError(
            "No best_*.pt checkpoints were found."
        )

    criterion = nn.CrossEntropyLoss()

    results = []

    for checkpoint_path in checkpoint_paths:
        print("\n" + "=" * 60)
        print(f"Evaluating: {checkpoint_path.name}")
        print("=" * 60)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if "model_state_dict" not in checkpoint:
            print(
                f"Skipping {checkpoint_path.name}: "
                "model_state_dict not found."
            )
            continue

        model = build_dino_vits16_cifar100(
            num_classes=100,
            freeze_backbone=False,
            pretrained=True,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model = model.to(device)

        test_metrics = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

        validation_accuracy = checkpoint.get(
            "validation_accuracy",
            checkpoint.get(
                "best_validation_accuracy",
                None,
            ),
        )

        validation_loss = checkpoint.get(
            "validation_loss",
            None,
        )

        round_or_epoch = checkpoint.get(
            "round",
            checkpoint.get(
                "epoch",
                None,
            ),
        )

        row = {
            "checkpoint": checkpoint_path.name,
            "round_or_epoch": round_or_epoch,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
        }

        results.append(row)

        print(
            f"Validation accuracy: "
            f"{validation_accuracy}"
        )

        print(
            f"Test loss: "
            f"{test_metrics['loss']:.6f}"
        )

        print(
            f"Test accuracy: "
            f"{test_metrics['accuracy']:.6f}"
        )

        # Free GPU memory before loading next checkpoint.
        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        fieldnames = [
            "checkpoint",
            "round_or_epoch",
            "validation_loss",
            "validation_accuracy",
            "test_loss",
            "test_accuracy",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("Evaluation completed.")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()