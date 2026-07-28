"""
================================================================================
 MAIZE YIELD PREDICTION  —  MODEL BUILDING & COMPARISON  (Chapter 3.9 - 3.11)
================================================================================
 This script trains and compares THREE models on the cleaned maize data:

     1. Linear Regression  -> the BASELINE (simple, interpretable)
     2. Random Forest      -> main model 1 (ensemble of independent trees)
     3. XGBoost (tuned)    -> main model 2 (trees built to correct each other)

 Each model is scored on the SAME held-out test set using four metrics
 (RMSE, MAE, R-squared, and accuracy within +/-10%). The best model, the
 scaler, and the feature list are then saved so the interactive dashboard
 can load them and make live predictions.

 Run:  python outputs/train_models.py
================================================================================
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # write figures to file, no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns
import joblib                                # saves trained models to disk

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --- reproducibility: fixing the seed means we get the SAME result every run ---
RNG = 42

# --- folder layout ---------------------------------------------------------
BASE = Path(__file__).resolve().parent          # .../outputs
PROJ = BASE.parent                              # project root
FIG = BASE / "figures"; FIG.mkdir(exist_ok=True)
MODELS = BASE / "models"; MODELS.mkdir(exist_ok=True)
RAW_CSV = PROJ / "crop_yield_dataset.csv"


def savefig(name):
    """Save the current matplotlib figure into outputs/figures and close it."""
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=120, bbox_inches="tight")
    plt.close()


# ==========================================================================
# STEP 1.  LOAD & PREPARE THE DATA
#          (same cleaning logic proven in Stage 1 — kept here so this script
#           runs stand-alone and always starts from the raw file)
# ==========================================================================
# IMPORTANT: pandas treats the text "None" as missing by default. Here "None"
# is a REAL category (rainfed / no previous crop), so we switch that off and
# only treat truly blank fields as missing.
NA_TOKENS = ["", "NA", "N/A", "NaN", "nan", "null", "NULL"]      # 'None' excluded on purpose
df = pd.read_csv(RAW_CSV, keep_default_na=False, na_values=NA_TOKENS)

# give the sentinel a clear, self-explanatory label
df["Irrigation"] = df["Irrigation"].replace({"None": "Rainfed"})
df["Previous_Crop"] = df["Previous_Crop"].replace({"None": "No_Previous_Crop"})

# the study is about maize only -> keep just those rows
maize = df[df["Crop"] == "Maize"].copy().reset_index(drop=True)

# split the columns into the three roles they play
NUMERIC = ["Soil_pH", "Rainfall_mm", "Temperature_C", "Humidity_pct",
           "Fertilizer_Used_kg", "Pesticides_Used_kg", "Planting_Density"]
CATEG = ["Region", "Soil_Type", "Irrigation", "Previous_Crop"]
TARGET = "Yield_ton_per_ha"

print(f"[data] maize records: {len(maize)}")

# --- one-hot encode the categorical columns into 0/1 indicator columns -----
# (drop_first=True removes one redundant column per category -> avoids the
#  "dummy variable trap" where columns are perfectly predictable from others)
model_df = maize.drop(columns=["Crop"])
model_df = pd.get_dummies(model_df, columns=CATEG, drop_first=True)

X = model_df.drop(columns=[TARGET])     # the inputs (features)
y = model_df[TARGET]                     # the thing we predict (yield)
FEATURE_COLUMNS = list(X.columns)        # remember exact column order for the dashboard
print(f"[data] feature count after encoding: {len(FEATURE_COLUMNS)}")


# ==========================================================================
# STEP 2.  TRAIN / TEST SPLIT  (Chapter 3.8)
#          80% to learn from, 20% held back for a fair final exam.
#          Stratified by Region so every region appears in both parts.
# ==========================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RNG, stratify=maize["Region"]
)
print(f"[split] train={len(X_train)}  test={len(X_test)}")


# ==========================================================================
# STEP 3.  NORMALISATION / STANDARDISATION  (Chapter 3.7)
#          Rescale each numeric feature to mean 0, std 1:  z = (x - mu) / sigma
#          mu and sigma are learned from the TRAIN set only, then applied to
#          test -> prevents "data leakage" from test into training.
# ==========================================================================
scaler = StandardScaler()
X_train_s, X_test_s = X_train.copy(), X_test.copy()
X_train_s[NUMERIC] = scaler.fit_transform(X_train[NUMERIC])   # learn + apply on train
X_test_s[NUMERIC] = scaler.transform(X_test[NUMERIC])         # apply same rule to test
# Note: tree models don't need scaling, but we standardise for ALL models so the
# comparison against Linear Regression is fair and consistent (as Chapter 3 states).


# ==========================================================================
# STEP 4.  A REUSABLE SCORING HELPER
#          Computes the four evaluation metrics from Chapter 3.10.
# ==========================================================================
# "accuracy" = % of predictions within +/-10% of the mean yield (a margin t)
TOLERANCE = 0.10 * y_test.mean()

def evaluate(name, model, X_te, y_te):
    """Return a dict of the four metrics for one trained model."""
    pred = model.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))     # penalises big misses
    mae = float(mean_absolute_error(y_te, pred))              # average miss size
    r2 = float(r2_score(y_te, pred))                          # % of variation explained
    acc = float((np.abs(y_te.values - pred) <= TOLERANCE).mean() * 100)  # within margin
    print(f"[eval] {name:18s} RMSE={rmse:6.2f}  MAE={mae:6.2f}  R2={r2:.3f}  Acc={acc:5.1f}%")
    return {"model": name, "rmse": round(rmse, 3), "mae": round(mae, 3),
            "r2": round(r2, 4), "accuracy_pct": round(acc, 2)}

results = {}          # metrics per model
fitted = {}           # the trained model objects


# ==========================================================================
# STEP 5.  MODEL 1 — LINEAR REGRESSION  (BASELINE, Chapter 3.9.1)
#          Fits a straight-line relationship: yield = b0 + b1*x1 + b2*x2 + ...
#          Fast and interpretable; sets the bar the ensembles must beat.
# ==========================================================================
lr = LinearRegression()
lr.fit(X_train_s, y_train)                         # learn coefficients from training data
results["Linear Regression"] = evaluate("Linear Regression", lr, X_test_s, y_test)
fitted["Linear Regression"] = lr


# ==========================================================================
# STEP 6.  MODEL 2 — RANDOM FOREST  (Chapter 3.9.2)
#          Builds many decision trees on random samples and AVERAGES them.
#          We lightly tune it with GridSearchCV (tries each combination using
#          5-fold cross-validation and keeps the best).
# ==========================================================================
rf_grid = {
    "n_estimators": [200, 400],          # how many trees in the forest
    "max_depth": [None, 15, 25],         # how deep each tree can grow
    "min_samples_leaf": [1, 2],          # min samples required at a leaf (higher = smoother)
}
rf_search = GridSearchCV(
    RandomForestRegressor(random_state=RNG, n_jobs=-1),
    rf_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
)
rf_search.fit(X_train_s, y_train)
rf = rf_search.best_estimator_                     # the best forest found
print(f"[tune] Random Forest best params: {rf_search.best_params_}")
results["Random Forest"] = evaluate("Random Forest", rf, X_test_s, y_test)
fitted["Random Forest"] = rf


# ==========================================================================
# STEP 7.  MODEL 3 — XGBOOST  (TUNED, Chapter 3.9.3)
#          Builds trees SEQUENTIALLY, each new tree correcting the previous
#          errors, plus a regularisation penalty that fights overfitting.
#          We tune the most important knobs with GridSearchCV.
# ==========================================================================
xgb_grid = {
    "n_estimators": [300, 600],          # number of boosting rounds (trees)
    "max_depth": [3, 5, 7],              # complexity of each tree
    "learning_rate": [0.03, 0.05, 0.1],  # how much each tree contributes (smaller = safer)
    "subsample": [0.8],                  # fraction of rows sampled per tree (adds robustness)
    "colsample_bytree": [0.8],           # fraction of features sampled per tree
}
xgb_search = GridSearchCV(
    XGBRegressor(random_state=RNG, n_jobs=-1, objective="reg:squarederror",
                 reg_lambda=1.0),        # reg_lambda = L2 regularisation strength
    xgb_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
)
xgb_search.fit(X_train_s, y_train)
xgb = xgb_search.best_estimator_
print(f"[tune] XGBoost best params: {xgb_search.best_params_}")
results["XGBoost"] = evaluate("XGBoost", xgb, X_test_s, y_test)
fitted["XGBoost"] = xgb


# ==========================================================================
# STEP 8.  CROSS-VALIDATION CHECK
#          Re-score each model with 5-fold CV on the training data to confirm
#          performance is stable and not a lucky test split.
# ==========================================================================
print("\n[cv] 5-fold cross-validated R^2 (mean +/- std):")
for name, model in fitted.items():
    scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="r2")
    results[name]["cv_r2_mean"] = round(float(scores.mean()), 4)
    results[name]["cv_r2_std"] = round(float(scores.std()), 4)
    print(f"      {name:18s} {scores.mean():.3f} +/- {scores.std():.3f}")


# ==========================================================================
# STEP 9.  PICK THE WINNER  (lowest RMSE = best)
# ==========================================================================
best_name = min(results, key=lambda k: results[k]["rmse"])
best_model = fitted[best_name]
print(f"\n[select] BEST MODEL = {best_name} (lowest RMSE)")


# ==========================================================================
# STEP 10.  COMPARISON FIGURES
# ==========================================================================
comp = pd.DataFrame(results).T[["rmse", "mae", "r2", "accuracy_pct"]].astype(float)

# 10a) RMSE & MAE bars (lower is better)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
comp[["rmse", "mae"]].plot(kind="bar", ax=ax[0], color=["#e76f51", "#f4a261"])
ax[0].set_title("Error (lower is better)"); ax[0].set_ylabel("ton/ha")
ax[0].tick_params(axis="x", rotation=0)
comp[["r2", "accuracy_pct"]].plot(kind="bar", ax=ax[1], color=["#2a9d8f", "#264653"], secondary_y="accuracy_pct")
ax[1].set_title("R-squared & Accuracy (higher is better)")
ax[1].tick_params(axis="x", rotation=0)
savefig("fig8_model_comparison.png")

# 10b) Actual vs Predicted for the best model (points near the diagonal = good)
pred_best = best_model.predict(X_test_s)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, pred_best, alpha=0.35, color="#2a9d8f", edgecolor="none")
lims = [y_test.min(), y_test.max()]
ax.plot(lims, lims, "--", color="#333", linewidth=1)          # the perfect-prediction line
ax.set_xlabel("Actual yield (ton/ha)"); ax.set_ylabel("Predicted yield (ton/ha)")
ax.set_title(f"Actual vs Predicted — {best_name}")
savefig("fig9_actual_vs_predicted.png")

# 10c) Residual analysis for the best model (Chapter 3.11 error analysis)
residuals = y_test.values - pred_best                          # error per record
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].scatter(pred_best, residuals, alpha=0.35, color="#457b9d", edgecolor="none")
ax[0].axhline(0, color="#333", linestyle="--")
ax[0].set_xlabel("Predicted yield"); ax[0].set_ylabel("Residual (actual - predicted)")
ax[0].set_title("Residuals vs Predicted")
sns.histplot(residuals, kde=True, ax=ax[1], color="#e9c46a")
ax[1].set_title("Distribution of residuals"); ax[1].set_xlabel("Residual (ton/ha)")
savefig("fig10_residual_analysis.png")

# 10d) Error broken down by region (find where the model struggles)
err_df = pd.DataFrame({"Region": maize.loc[y_test.index, "Region"].values,
                       "abs_error": np.abs(residuals)})
region_err = err_df.groupby("Region")["abs_error"].mean().sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
region_err.plot(kind="bar", ax=ax, color="#2a9d8f")
ax.set_ylabel("Mean absolute error (ton/ha)")
ax.set_title(f"Average error by region — {best_name}")
ax.tick_params(axis="x", rotation=0)
savefig("fig11_error_by_region.png")

# 10e) Feature importance of the best tree model (if available)
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=FEATURE_COLUMNS)
    imp = imp.sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=imp.values, y=imp.index, ax=ax, color="#2a9d8f")
    ax.set_title(f"Feature importance — {best_name}"); ax.set_xlabel("Importance")
    savefig("fig12_best_feature_importance.png")


# ==========================================================================
# STEP 11.  SAVE EVERYTHING THE DASHBOARD NEEDS
# ==========================================================================
# the three models (so the dashboard can let users switch between them)
joblib.dump(lr, MODELS / "linear_regression.joblib")
joblib.dump(rf, MODELS / "random_forest.joblib")
joblib.dump(xgb, MODELS / "xgboost.joblib")
# the scaler + feature order (must match exactly at prediction time)
joblib.dump(scaler, MODELS / "scaler.joblib")

# reference lists the dashboard uses to build & validate the input form
CATEGORY_OPTIONS = {c: sorted(maize[c].unique().tolist()) for c in CATEG}
NUMERIC_RANGES = {c: [float(maize[c].min()), float(maize[c].max())] for c in NUMERIC}

metadata = {
    "feature_columns": FEATURE_COLUMNS,
    "numeric_features": NUMERIC,
    "categorical_features": CATEG,
    "category_options": CATEGORY_OPTIONS,
    "numeric_ranges": NUMERIC_RANGES,
    "target": TARGET,
    "tolerance_ton_per_ha": round(float(TOLERANCE), 3),
    "best_model": best_name,
    "results": results,
    "best_params": {"random_forest": rf_search.best_params_,
                    "xgboost": xgb_search.best_params_},
}
with open(MODELS / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# also save the plain comparison table as CSV for the report
comp.round(3).to_csv(BASE / "model_comparison.csv")

print(f"\n[save] models + scaler + metadata -> {MODELS}")
print("[done] model building complete.")
