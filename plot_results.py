from pathlib import Path

import matplotlib

# Use a non-GUI backend to avoid Tkinter/Tcl errors.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


STRATEGY_ORDER = [
    "least_sensitive",
    "most_sensitive",
    "lowest_magnitude",
    "highest_magnitude",
    "random",
]


def extract_mask_strategy(checkpoint_name: str) -> str | None:
    checkpoint_name = checkpoint_name.lower()

    for strategy in STRATEGY_ORDER:
        if strategy in checkpoint_name:
            return strategy

    return None


def plot_accuracy_comparison(
    metric_name: str,
    output_name: str,
    values: pd.Series,
    labels: list[str],
    output_dir: Path,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        labels,
        values,
        color=[
            "#4C72B0",
            "#55A868",
            "#C44E52",
            "#8172B3",
            "#CCB974",
        ],
    )

    ax.set_xlabel("Mask Strategy")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs Mask Strategy")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()

    output_path = output_dir / output_name
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved: {output_path}")

    return fig


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    experiments_dir = base_dir / "experiments"
    csv_path = (
        experiments_dir
        / "mask_strategy_comparison.csv"
    )

    experiments_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = {
        "checkpoint",
        "validation_accuracy",
        "test_accuracy",
    }

    missing_columns = (
        required_columns.difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df = df.copy()

    df["mask_strategy"] = df[
        "checkpoint"
    ].map(extract_mask_strategy)

    df = df.dropna(
        subset=["mask_strategy"]
    ).copy()

    if df.empty:
        raise ValueError(
            "No valid mask strategies were found "
            "in the checkpoint names."
        )

    df["validation_accuracy"] = pd.to_numeric(
        df["validation_accuracy"],
        errors="raise",
    )

    df["test_accuracy"] = pd.to_numeric(
        df["test_accuracy"],
        errors="raise",
    )

    df["mask_strategy"] = pd.Categorical(
        df["mask_strategy"],
        categories=STRATEGY_ORDER,
        ordered=True,
    )

    df = df.sort_values(
        "mask_strategy"
    ).reset_index(drop=True)

    labels = (
        df["mask_strategy"]
        .astype(str)
        .tolist()
    )

    validation_values = df[
        "validation_accuracy"
    ]

    test_values = df[
        "test_accuracy"
    ]

    validation_fig = plot_accuracy_comparison(
        metric_name="Validation Accuracy",
        output_name=(
            "validation_accuracy_comparison.png"
        ),
        values=validation_values,
        labels=labels,
        output_dir=experiments_dir,
    )

    test_fig = plot_accuracy_comparison(
        metric_name="Test Accuracy",
        output_name=(
            "test_accuracy_comparison.png"
        ),
        values=test_values,
        labels=labels,
        output_dir=experiments_dir,
    )

    plt.close(validation_fig)
    plt.close(test_fig)

    print("All plots were created successfully.")


if __name__ == "__main__":
    main()