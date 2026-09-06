"""Plot the recorded test-accuracy comparison for mask strategies."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

strategies = [
    "Least-sensitive",
    "Most-sensitive",
    "Lowest-magnitude",
    "Highest-magnitude",
    "Random"
]

test_accuracy = [0.7161, 0.0916, 0.1326, 0.1180, 0.1240]

plt.figure(figsize=(8, 5))
bars = plt.bar(strategies, test_accuracy)

plt.ylabel("Test Accuracy")
plt.xlabel("Mask Strategy")
plt.title("Test Accuracy across Mask Selection Strategies")
plt.xticks(rotation=15)
plt.ylim(0, 0.8)

for bar, value in zip(bars, test_accuracy):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        value + 0.01,
        f"{value:.4f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig("experiments/test_accuracy_comparison.png", dpi=300)
plt.close()

print("Saved: experiments/test_accuracy_comparison.png")