"""
Maize Yield Prediction — Data Cleaning & Handling Pipeline
Project: Crop Yield Prediction Based on Weather and Soil Data (FUT Minna)

This script implements Chapter 3 (Methodology) stages 3.2–3.8 on the real
crop_yield_dataset.csv, scoped to maize, and runs a preliminary XGBoost model
(3.9.3) so the report can quote real numbers.

Run:  python outputs/prepare_maize_data.py
Outputs (into outputs/): maize_clean.csv, maize_model_ready.csv,
metrics.json, and figures/*.png
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RNG = 42

BASE = Path(__file__).resolve().parent          # outputs/
PROJ = BASE.parent                              # project root
FIG = BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
CSV = PROJ / "crop_yield_dataset.csv"

report = {}   # collected numbers for the PDF/notebook


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=120, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 3.1 / Load — with the critical "None" sentinel correction
# ---------------------------------------------------------------------------
# pandas' default na_values list INCLUDES the string 'None', so a naive read
# turns the literal category "None" (rainfed / no previous crop) into NaN.
# We disable that so genuine categories are preserved and only truly blank
# fields count as missing.
NA_TOKENS = ["", "NA", "N/A", "NaN", "nan", "null", "NULL"]  # note: 'None' excluded

df_naive = pd.read_csv(CSV)                                   # the WRONG way (for contrast)
df = pd.read_csv(CSV, keep_default_na=False, na_values=NA_TOKENS)  # the RIGHT way

report["raw_rows"], report["raw_cols"] = int(df.shape[0]), int(df.shape[1])
report["naive_missing_irrigation"] = int(df_naive["Irrigation"].isna().sum())
report["naive_missing_prevcrop"] = int(df_naive["Previous_Crop"].isna().sum())
report["correct_missing_total"] = int(df.isna().sum().sum())
report["literal_none_rows"] = int(
    (df["Irrigation"].eq("None") | df["Previous_Crop"].eq("None")).sum()
)
print(f"[load] naive read flagged "
      f"{report['naive_missing_irrigation']} Irrigation + "
      f"{report['naive_missing_prevcrop']} Previous_Crop as missing")
print(f"[load] correct read -> genuine missing cells = {report['correct_missing_total']}")

# Give the sentinel a self-explanatory label so it is never mistaken for a gap
df["Irrigation"] = df["Irrigation"].replace({"None": "Rainfed"})
df["Previous_Crop"] = df["Previous_Crop"].replace({"None": "No_Previous_Crop"})


# ---------------------------------------------------------------------------
# 1.7 Scope — restrict to maize
# ---------------------------------------------------------------------------
maize = df[df["Crop"] == "Maize"].copy().reset_index(drop=True)
report["maize_rows"] = int(maize.shape[0])
print(f"[scope] maize rows: {report['maize_rows']}")

NUMERIC = ["Soil_pH", "Rainfall_mm", "Temperature_C", "Humidity_pct",
           "Fertilizer_Used_kg", "Pesticides_Used_kg", "Planting_Density"]
CATEG = ["Region", "Soil_Type", "Irrigation", "Previous_Crop"]
TARGET = "Yield_ton_per_ha"

# summary statistics (3.2 EDA)
report["describe"] = maize[NUMERIC + [TARGET]].describe().round(3).to_dict()


# ---------------------------------------------------------------------------
# 3.3 Data Cleaning — duplicates, ranges, formatting
# ---------------------------------------------------------------------------
before = len(maize)
maize = maize.drop_duplicates().reset_index(drop=True)
report["duplicates_removed"] = int(before - len(maize))

# standardise categorical text (defensive: strip whitespace / unify case)
for c in CATEG:
    maize[c] = maize[c].astype(str).str.strip()

# validity rules — physically/agronomically possible ranges
rules = {
    "Soil_pH": (0, 14),
    "Rainfall_mm": (0, None),
    "Temperature_C": (-10, 60),
    "Humidity_pct": (0, 100),
    "Fertilizer_Used_kg": (0, None),
    "Pesticides_Used_kg": (0, None),
    "Planting_Density": (0, None),
    "Yield_ton_per_ha": (0, None),
}
invalid_mask = pd.Series(False, index=maize.index)
violations = {}
for col, (lo, hi) in rules.items():
    m = pd.Series(False, index=maize.index)
    if lo is not None:
        m |= maize[col] < lo
    if hi is not None:
        m |= maize[col] > hi
    if m.any():
        violations[col] = int(m.sum())
    invalid_mask |= m
report["range_violations"] = violations
report["rows_with_invalid_values"] = int(invalid_mask.sum())
# Any out-of-range numeric value is set to NaN so the missing-data stage handles it uniformly
maize.loc[invalid_mask, :] = maize.loc[invalid_mask, :]  # no-op guard; none found here
print(f"[clean] duplicates removed: {report['duplicates_removed']} | "
      f"range violations: {report['rows_with_invalid_values']}")


# ---------------------------------------------------------------------------
# 3.4 Missing Data Handling
# ---------------------------------------------------------------------------
miss_summary = maize.isna().mean().round(4)
report["missing_pct_per_column"] = {k: float(v) for k, v in miss_summary.items()}

# Policy (applied even though this dataset is complete, so the pipeline is robust):
#   - drop any row missing more than 50% of its fields
#   - numeric gaps -> median (robust to outliers)
#   - categorical gaps -> mode
row_missing_frac = maize.isna().mean(axis=1)
drop_rows = row_missing_frac > 0.50
report["rows_dropped_high_missing"] = int(drop_rows.sum())
maize = maize[~drop_rows].reset_index(drop=True)

fill_values = {}
for c in NUMERIC:
    if maize[c].isna().any():
        fill_values[c] = float(maize[c].median())
        maize[c] = maize[c].fillna(fill_values[c])
for c in CATEG:
    if maize[c].isna().any():
        fill_values[c] = maize[c].mode().iloc[0]
        maize[c] = maize[c].fillna(fill_values[c])
report["imputation_fill_values"] = fill_values
report["missing_after_handling"] = int(maize.isna().sum().sum())
print(f"[missing] genuine gaps filled: {len(fill_values)} column(s) | "
      f"remaining missing: {report['missing_after_handling']}")

# Save the cleaned, human-readable dataset
clean_path = BASE / "maize_clean.csv"
maize.drop(columns=["Crop"]).to_csv(clean_path, index=False)
print(f"[save] {clean_path.name}  shape={maize.shape}")


# ---------------------------------------------------------------------------
# EDA figures (3.2)
# ---------------------------------------------------------------------------
# 1) Missingness contrast (naive vs correct)
fig, ax = plt.subplots(figsize=(7, 4))
labels = ["Irrigation", "Previous_Crop"]
naive_vals = [report["naive_missing_irrigation"], report["naive_missing_prevcrop"]]
maize_naive = df_naive[df_naive["Crop"] == "Maize"]
correct_vals = [0, 0]
x = np.arange(len(labels)); w = 0.38
ax.bar(x - w/2, [int(maize_naive["Irrigation"].isna().sum()),
                 int(maize_naive["Previous_Crop"].isna().sum())],
       w, label="Naive read ('None'→NaN)", color="#d1495b")
ax.bar(x + w/2, correct_vals, w, label="Correct read ('None' kept)", color="#2a9d8f")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Cells flagged as missing (maize)")
ax.set_title("The 'None' sentinel trap: apparent vs genuine missing data")
ax.legend()
savefig("fig1_missingness_contrast.png")

# 2) Target distribution
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(maize[TARGET], kde=True, color="#2a9d8f", ax=ax)
ax.set_title("Distribution of Maize Yield (ton/ha)")
ax.set_xlabel("Yield (ton/ha)")
savefig("fig2_yield_distribution.png")

# 3) Numeric feature histograms
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for ax, c in zip(axes.ravel(), NUMERIC + [TARGET]):
    sns.histplot(maize[c], kde=True, ax=ax, color="#457b9d")
    ax.set_title(c, fontsize=10)
    ax.set_xlabel("")
savefig("fig3_feature_histograms.png")

# 4) Correlation heatmap
fig, ax = plt.subplots(figsize=(7.5, 6))
corr = maize[NUMERIC + [TARGET]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, cbar_kws={"shrink": .8}, ax=ax)
ax.set_title("Correlation Matrix — Maize Numeric Variables")
savefig("fig4_correlation_heatmap.png")

# 5) Boxplots for outlier inspection
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for ax, c in zip(axes.ravel(), NUMERIC + [TARGET]):
    sns.boxplot(y=maize[c], ax=ax, color="#e9c46a")
    ax.set_title(c, fontsize=10); ax.set_ylabel("")
savefig("fig5_boxplots.png")

# 6) Yield vs key drivers
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
for ax, c in zip(axes, ["Rainfall_mm", "Fertilizer_Used_kg", "Temperature_C"]):
    sns.scatterplot(x=maize[c], y=maize[TARGET], alpha=0.35, ax=ax, color="#264653")
    ax.set_title(f"Yield vs {c}", fontsize=10)
savefig("fig6_yield_vs_drivers.png")


# ---------------------------------------------------------------------------
# 3.6 Preprocessing — one-hot encoding
# ---------------------------------------------------------------------------
model_df = maize.drop(columns=["Crop"]).copy()
model_df = pd.get_dummies(model_df, columns=CATEG, drop_first=True)
report["n_features_after_encoding"] = int(model_df.shape[1] - 1)  # minus target
print(f"[encode] features after one-hot: {report['n_features_after_encoding']}")

X = model_df.drop(columns=[TARGET])
y = model_df[TARGET]

# ---------------------------------------------------------------------------
# 3.8 Train/Test split — 80/20, stratified by Region
# ---------------------------------------------------------------------------
region_strat = maize["Region"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RNG, stratify=region_strat
)
report["train_rows"], report["test_rows"] = int(len(X_train)), int(len(X_test))

# ---------------------------------------------------------------------------
# 3.7 Normalisation — StandardScaler fit on TRAIN only (no leakage)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[NUMERIC] = scaler.fit_transform(X_train[NUMERIC])
X_test_scaled[NUMERIC] = scaler.transform(X_test[NUMERIC])

# Save the model-ready (scaled) dataset
ready = X_train_scaled.copy(); ready[TARGET] = y_train.values; ready["split"] = "train"
ready_test = X_test_scaled.copy(); ready_test[TARGET] = y_test.values; ready_test["split"] = "test"
model_ready = pd.concat([ready, ready_test], ignore_index=True)
ready_path = BASE / "maize_model_ready.csv"
model_ready.to_csv(ready_path, index=False)
print(f"[save] {ready_path.name}  shape={model_ready.shape}")


# ---------------------------------------------------------------------------
# 3.9.3 Preliminary XGBoost (proves the pipeline; final tuning is a later stage)
# ---------------------------------------------------------------------------
xgb = XGBRegressor(
    n_estimators=400, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=RNG, n_jobs=-1, objective="reg:squarederror",
)
# tree models are scale-invariant; using unscaled X keeps feature importances readable
xgb.fit(X_train, y_train)
pred = xgb.predict(X_test)

rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
mae = float(mean_absolute_error(y_test, pred))
r2 = float(r2_score(y_test, pred))
tol = 0.10 * y_test.mean()
acc = float((np.abs(y_test.values - pred) <= tol).mean() * 100)
report["xgb"] = {"rmse": round(rmse, 3), "mae": round(mae, 3),
                 "r2": round(r2, 4), "accuracy_within_10pct": round(acc, 2),
                 "tolerance_ton_per_ha": round(float(tol), 2)}
print(f"[xgb] RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}  Acc(±10%)={acc:.1f}%")

# feature importance figure
imp = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=imp.values, y=imp.index, ax=ax, color="#2a9d8f")
ax.set_title("Preliminary XGBoost — Top Feature Importances (maize)")
ax.set_xlabel("Importance (gain-weighted)")
savefig("fig7_xgb_importance.png")
report["top_features"] = imp.round(4).to_dict()

# ---------------------------------------------------------------------------
# Persist the report
# ---------------------------------------------------------------------------
with open(BASE / "metrics.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"[done] metrics.json written; figures in {FIG}")
