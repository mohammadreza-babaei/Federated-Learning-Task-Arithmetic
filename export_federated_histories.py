"""Export round histories stored inside latest federated checkpoints."""

from pathlib import Path
import csv
import torch


CHECKPOINT_DIR = Path("experiments/checkpoints")
OUTPUT_DIR = Path("experiments/federated_histories")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_files = sorted(
    CHECKPOINT_DIR.glob("last_federated*.pt")
)

if not checkpoint_files:
    raise FileNotFoundError(
        "No last_federated checkpoint files were found."
    )

# Each checkpoint contains structured round records; export only plotting fields.
for checkpoint_path in checkpoint_files:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    history = checkpoint.get("history", [])

    if not history:
        print(f"No history found: {checkpoint_path.name}")
        continue

    output_name = (
        checkpoint_path.stem.replace("last_", "") + "_history.csv"
    )
    output_path = OUTPUT_DIR / output_name

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "round",
                "validation_loss",
                "validation_accuracy",
            ]
        )

        for row in history:
            writer.writerow(
                [
                    row.get("round"),
                    row.get("validation_loss"),
                    row.get("validation_accuracy"),
                ]
            )

    print(f"Saved: {output_path}")

print("\nDone.")