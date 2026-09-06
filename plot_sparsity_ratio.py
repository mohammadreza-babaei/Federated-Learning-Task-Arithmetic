"""Plot the recorded effect of active-parameter ratio on test accuracy."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

sparsity_ratios = [0.05, 0.10, 0.20]
test_accuracy = [0.2163, 0.7161, 0.1190]

plt.figure(figsize=(7, 5))

bars = plt.bar(
    [str(x) for x in sparsity_ratios],
    test_accuracy
)

plt.xlabel("Sparsity Ratio")
plt.ylabel("Test Accuracy")
plt.title("Effect of Sparsity Ratio on Test Accuracy")

for bar, value in zip(bars, test_accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{value:.4f}",
        ha="center"
    )

plt.ylim(0, 0.8)
plt.tight_layout()

plt.savefig(
    "experiments/sparsity_ratio_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: experiments/sparsity_ratio_comparison.png")