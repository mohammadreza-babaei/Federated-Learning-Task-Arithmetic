"""Plot test-accuracy means and standard deviations for data partitions."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

df = pd.read_csv("experiments/iid_vs_non_iid_summary.csv")

labels = df["Setting"].tolist()

# The summary stores each result as ``mean +/- standard deviation``.
test_stats = df["Test Accuracy"].str.split(" +/- ", expand=True, regex=False).astype(float)
test_values = test_stats[0].tolist()
test_errors = test_stats[1].tolist()

plt.figure(figsize=(8, 5))
plt.bar(labels, test_values, yerr=test_errors, capsize=4)

plt.ylabel("Test Accuracy")
plt.xlabel("Data Distribution")
plt.title("IID vs Non-IID Federated Learning Performance")

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("experiments/iid_vs_non_iid_test_accuracy.png", dpi=300)
