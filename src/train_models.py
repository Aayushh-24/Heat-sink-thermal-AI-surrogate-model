import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# --- Load data ---
df = pd.read_csv("../data/heat_sink_dataset.csv")

# --- Define features (inputs) and targets ---
FEATURES = ["TDP", "V_air", "k_tim"]
TARGETS = ["R_total", "T_j"]

X = df[FEATURES]

# --- Train/test split (80/20), fixed random_state for reproducibility ---
X_train, X_test, df_train_idx, df_test_idx = train_test_split(
    X, df.index, test_size=0.2, random_state=42
)

# --- Feature scaling ---
# Linear Regression benefits from scaled features when inputs have very
# different ranges (TDP: 30-250, V_air: 0.5-15, k_tim: 1-12).
# Random Forest is scale-invariant, but we scale consistently anyway so
# the same pipeline works for both models.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def evaluate_model(y_true, y_pred, model_name, target_name):
    """Compute MAE, RMSE, R2 and print a clean summary line."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{target_name}] {model_name:20s} | MAE: {mae:.5f} | RMSE: {rmse:.5f} | R2: {r2:.5f}")
    return {"model": model_name, "target": target_name, "MAE": mae, "RMSE": rmse, "R2": r2}


results = []
trained_models = {}

for target in TARGETS:
    y_train = df.loc[df_train_idx, target]
    y_test = df.loc[df_test_idx, target]

    # --- Model 1: Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    results.append(evaluate_model(y_test, y_pred_lr, "Linear Regression", target))
    trained_models[f"lr_{target}"] = lr

    # --- Model 2: Random Forest Regressor ---
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    results.append(evaluate_model(y_test, y_pred_rf, "Random Forest", target))
    trained_models[f"rf_{target}"] = rf

    # --- Feature importance from Random Forest ---
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(f"\nFeature importance for {target} (Random Forest):")
    print(importances)
    print()

# ============================================================
# RIGOROUS GENERALIZATION TEST
# ------------------------------------------------------------
# The random 80/20 split above is misleading for R_total: since
# TDP has ~zero effect on R_total, and our sweep is a full
# factorial grid, most "held-out" (V_air, k_tim) combinations
# already appear in the training set under a different TDP value.
# This lets the model "cheat" by memorizing rather than truly
# generalizing.
#
# Because our simulator is deterministic and cheap, we generate a
# SEPARATE test set of randomly sampled points that do NOT lie on
# the training grid at all. This is a true test of interpolation
# accuracy on genuinely unseen parameter combinations.
# ============================================================
from heat_sink_model import heat_sink_design

np.random.seed(123)  # different seed from the grid, for independence
N_RANDOM_TEST = 1000

random_TDP = np.random.uniform(30, 250, N_RANDOM_TEST)
random_V_air = np.random.uniform(0.5, 15, N_RANDOM_TEST)
random_k_tim = np.random.uniform(1, 12, N_RANDOM_TEST)

random_test_rows = []
for tdp, v_air, k_tim in zip(random_TDP, random_V_air, random_k_tim):
    result = heat_sink_design(TDP=tdp, V_air=v_air, k_tim=k_tim)
    random_test_rows.append({"TDP": tdp, "V_air": v_air, "k_tim": k_tim, **result})

random_test_df = pd.DataFrame(random_test_rows)
X_random_test_scaled = scaler.transform(random_test_df[FEATURES])

print("\n=== Rigorous generalization test (off-grid random points) ===")
for target in TARGETS:
    y_true = random_test_df[target]

    y_pred_lr = trained_models[f"lr_{target}"].predict(X_random_test_scaled)
    evaluate_model(y_true, y_pred_lr, "Linear Regression (off-grid)", target)

    y_pred_rf = trained_models[f"rf_{target}"].predict(X_random_test_scaled)
    evaluate_model(y_true, y_pred_rf, "Random Forest (off-grid)", target)

# --- Save results table ---
results_df = pd.DataFrame(results)
results_df.to_csv("../reports/model_evaluation_results.csv", index=False)
print("\n=== Full results table ===")
print(results_df)

# --- Save trained models + scaler for reuse ---
joblib.dump(trained_models, "../reports/trained_models.pkl")
joblib.dump(scaler, "../reports/feature_scaler.pkl")
print("\nModels and results saved to ../reports/")