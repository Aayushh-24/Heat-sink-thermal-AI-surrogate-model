import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

FEATURES = ["TDP", "V_air", "k_tim"]
TARGETS = ["R_total", "T_j"]

df = pd.read_csv("../data/heat_sink_dataset.csv")
X = df[FEATURES]

scaler = joblib.load("../reports/feature_scaler.pkl")
trained_models = joblib.load("../reports/trained_models.pkl")
X_scaled = scaler.transform(X)

# --- Permutation importance for each Random Forest model ---
for target in TARGETS:
    rf = trained_models[f"rf_{target}"]
    y = df[target]

    perm_result = permutation_importance(
        rf, X_scaled, y, n_repeats=20, random_state=42, n_jobs=-1
    )

    print(f"\n=== Permutation importance for {target} ===")
    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std
    }).sort_values("importance_mean", ascending=False)
    print(importance_df)

    importance_df.to_csv(f"../reports/permutation_importance_{target}.csv", index=False)

# --- Partial dependence plots (shape of each input's effect) ---
for target in TARGETS:
    rf = trained_models[f"rf_{target}"]
    fig, ax = plt.subplots(figsize=(12, 4))
    PartialDependenceDisplay.from_estimator(
        rf, X_scaled, features=[0, 1, 2], feature_names=FEATURES, ax=ax
    )
    plt.suptitle(f"Partial Dependence — {target}")
    plt.tight_layout()
    plt.savefig(f"../reports/partial_dependence_{target}.png", dpi=150)
    plt.close()

print("\nSensitivity analysis complete. Results saved to ../reports/")