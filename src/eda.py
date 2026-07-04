import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset generated in Phase 4
df = pd.read_csv("../data/heat_sink_dataset.csv")

print("=== Dataset shape ===")
print(df.shape)

print("\n=== Summary statistics ===")
print(df.describe())

print("\n=== Missing values check ===")
print(df.isnull().sum())

# --- Distribution plots for the two ML targets ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["R_total"], kde=True, ax=axes[0])
axes[0].set_title("Distribution of R_total (deg C/W)")
sns.histplot(df["T_j"], kde=True, ax=axes[1])
axes[1].set_title("Distribution of T_j (deg C)")
plt.tight_layout()
plt.savefig("../reports/target_distributions.png", dpi=150)
plt.close()

# --- Correlation heatmap (numeric preview of Phase 8, useful now too) ---
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("../reports/correlation_heatmap.png", dpi=150)
plt.close()

# --- Scatter: each input vs each target ---
inputs = ["TDP", "V_air", "k_tim"]
targets = ["R_total", "T_j"]

fig, axes = plt.subplots(len(targets), len(inputs), figsize=(15, 8))
for i, target in enumerate(targets):
    for j, inp in enumerate(inputs):
        axes[i, j].scatter(df[inp], df[target], s=3, alpha=0.3)
        axes[i, j].set_xlabel(inp)
        axes[i, j].set_ylabel(target)
plt.tight_layout()
plt.savefig("../reports/input_vs_target_scatter.png", dpi=150)
plt.close()

print("\nEDA plots saved to ../reports/")