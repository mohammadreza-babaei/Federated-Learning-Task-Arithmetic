import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# -----------------------------
# Figure 4: Mask strategies
# -----------------------------
strategies = [
    "Least-sensitive",
    "Most-sensitive",
    "Lowest-magnitude",
    "Highest-magnitude",
    "Random",
]

strategy_runs = [
    [0.7161, 0.1863, 0.0106],
    [0.0916, 0.1457, 0.1399],
    [0.1326, 0.1151, 0.1105],
    [0.1180, 0.0696, 0.1247],
    [0.1240, 0.0950, 0.1201],
]

strategy_means = [np.mean(x) for x in strategy_runs]
strategy_stds = [np.std(x, ddof=1) for x in strategy_runs]

plt.figure(figsize=(8, 5))
plt.bar(
    strategies,
    strategy_means,
    yerr=strategy_stds,
    capsize=5,
)
plt.ylabel("Test Accuracy")
plt.xlabel("Mask Strategy")
plt.title("Test Accuracy across Mask Selection Strategies")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

plt.savefig(
    "experiments/test_accuracy_comparison.png",
    dpi=300,
)

plt.close()


# -----------------------------
# Figure 5: Sparsity ratios
# -----------------------------
ratios = [
    "0.05",
    "0.10",
    "0.20",
]

sparsity_runs = [
    [0.2163, 0.0135, 0.1655],
    [0.7161, 0.1863, 0.0106],
    [0.1190, 0.0740, 0.0735],
]

sparsity_means = [np.mean(x) for x in sparsity_runs]
sparsity_stds = [np.std(x, ddof=1) for x in sparsity_runs]

plt.figure(figsize=(7, 5))
plt.bar(
    ratios,
    sparsity_means,
    yerr=sparsity_stds,
    capsize=5,
)
plt.ylabel("Test Accuracy")
plt.xlabel("Sparsity Ratio")
plt.title("Effect of Sparsity Ratio on Test Accuracy")
plt.tight_layout()

plt.savefig(
    "experiments/sparsity_ratio_comparison.png",
    dpi=300,
)

plt.close()


# -----------------------------
# Figure 3: IID vs Non-IID
# -----------------------------
settings = [
    "IID (3 seeds)",
    "Non-IID nc=1",
    "Non-IID nc=5",
    "Non-IID nc=10",
    "Non-IID nc=50",
]

test_means = [
    0.0320,
    0.0151,
    0.0107,
    0.0137,
    0.0178,
]

test_stds = [
    0.0157,
    0.0031,
    0.0012,
    0.0022,
    0.0035,
]

plt.figure(figsize=(8, 5))
plt.bar(
    settings,
    test_means,
    yerr=test_stds,
    capsize=5,
)
plt.ylabel("Test Accuracy")
plt.xlabel("Data Distribution")
plt.title("IID vs. Non-IID Test Accuracy")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

plt.savefig(
    "experiments/iid_vs_non_iid_test_accuracy.png",
    dpi=300,
)

plt.close()


# -----------------------------
# Output summary
# -----------------------------
print("Done.")

print(
    "Figure 4:",
    list(zip(strategies, strategy_means, strategy_stds))
)

print(
    "Figure 5:",
    list(zip(ratios, sparsity_means, sparsity_stds))
)

print(
    "Figure 3:",
    list(zip(settings, test_means, test_stds))
)