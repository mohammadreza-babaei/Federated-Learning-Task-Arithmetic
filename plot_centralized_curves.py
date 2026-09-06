"""Plot mean centralized test curves across the recorded random seeds."""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = [
    "experiments/final_seed1_history.csv",
    "experiments/final_seed2_history.csv",
    "experiments/final_seed3_history.csv",
]

dfs = [pd.read_csv(f) for f in files]

epochs = dfs[0]["epoch"]

mean_test_acc = sum(df["test_accuracy"] for df in dfs) / len(dfs)
mean_test_loss = sum(df["test_loss"] for df in dfs) / len(dfs)

# Aggregate and plot test accuracy across epochs.
plt.figure(figsize=(6, 4))
plt.plot(epochs, mean_test_acc)
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Centralized Test Accuracy Across Epochs")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("experiments/centralized_test_accuracy_curve.png", dpi=300)
plt.close()

# Aggregate and plot test loss across epochs.
plt.figure(figsize=(6, 4))
plt.plot(epochs, mean_test_loss)
plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.title("Centralized Test Loss Across Epochs")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("experiments/centralized_test_loss_curve.png", dpi=300)
plt.close()

print("Saved:")
print("experiments/centralized_test_accuracy_curve.png")
print("experiments/centralized_test_loss_curve.png")