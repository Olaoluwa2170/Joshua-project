"""
================================================================================
 MAIZE YIELD PREDICTION — INTERACTIVE DASHBOARD  (Streamlit)
================================================================================
 Chapter 3.12 (System Design) delivered as a working interactive tool.

 What it does:
   * lets a user enter weather, soil and management values for a maize plot
   * runs the trained model of their choice (Linear Regression / Random Forest / XGBoost)
   * shows the predicted yield, a typical-error band, and the main driving factors
   * stores every prediction in a small SQLite database (the "history")
   * shows a Model Performance page comparing the three models

 The models, scaler and metadata are produced by outputs/train_models.py.

 Run:  streamlit run dashboard/app.py
================================================================================
"""

import json
import sqlite3
from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# reportlab builds the downloadable PDF prediction report. It is optional:
# if it is not installed, the app still runs and offers a plain-text report instead.
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

# --------------------------------------------------------------------------
# PATHS — everything is located relative to this file, so it runs anywhere.
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # .../dashboard
PROJ = HERE.parent                              # project root
MODELS_DIR = PROJ / "outputs" / "models"        # where train_models.py saved everything
DB_PATH = HERE / "predictions.db"               # local SQLite file for prediction history

# Streamlit page configuration (title + wide layout + icon in the browser tab).
st.set_page_config(page_title="Maize Yield Prediction", page_icon="🌽", layout="wide")


# --------------------------------------------------------------------------
# Small helpers for building sliders from a [min, max] range.
#   _r   -> (min, max) as floats, unpacked into the slider's bounds
#   _mid -> the midpoint, used as the slider's default starting value
# --------------------------------------------------------------------------
def _r(rng):
    return float(rng[0]), float(rng[1])


def _mid(rng):
    return float((rng[0] + rng[1]) / 2.0)


# ==========================================================================
# LOADING THE TRAINED ARTEFACTS
#   @st.cache_resource means this runs ONCE and is reused on every interaction,
#   so the app stays fast (models are not reloaded on each click).
# ==========================================================================
@st.cache_resource
def load_artifacts():
    """Load the three models, the scaler, and the metadata file from disk."""
    models = {
        "Linear Regression": joblib.load(MODELS_DIR / "linear_regression.joblib"),
        "Random Forest":     joblib.load(MODELS_DIR / "random_forest.joblib"),
        "XGBoost":           joblib.load(MODELS_DIR / "xgboost.joblib"),
    }
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    meta = json.loads((MODELS_DIR / "model_metadata.json").read_text())
    return models, scaler, meta


models, scaler, META = load_artifacts()
FEATURES   = META["feature_columns"]        # exact column order the models expect
NUMERIC    = META["numeric_features"]
CATEG      = META["categorical_features"]
CAT_OPTS   = META["category_options"]        # dropdown choices per category
NUM_RANGES = META["numeric_ranges"]          # min/max per numeric field (for sliders)
RESULTS    = META["results"]                 # each model's test metrics


# ==========================================================================
# DATABASE — a tiny SQLite store for prediction history.
#   This mirrors the PredictionRecord table + DatabaseManager class from
#   Chapter 3.12: every prediction is saved with its inputs and the model used.
# ==========================================================================
def db_init():
    """Create the predictions table on first run (if it does not exist yet)."""
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at      TEXT,       -- when the prediction was made
            model_used      TEXT,       -- which algorithm produced it
            predicted_yield REAL,       -- the result, in ton/ha
            inputs_json     TEXT        -- the raw user inputs, stored as JSON
        )""")
    con.commit(); con.close()


def db_save(model_name, predicted, inputs):
    """Insert one prediction row into the history table."""
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO predictions (created_at, model_used, predicted_yield, inputs_json) VALUES (?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_name,
         float(predicted), json.dumps(inputs)),
    )
    con.commit(); con.close()


def db_history():
    """Return the full prediction history as a DataFrame (newest first)."""
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", con)
    con.close()
    return df


def db_clear():
    """Delete all saved predictions (used by the 'Clear history' button)."""
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM predictions"); con.commit(); con.close()


db_init()


# ==========================================================================
# THE CORE PREDICTION LOGIC
#   Turns a dictionary of user inputs into a single prediction, applying the
#   EXACT same preprocessing (encoding + scaling) used during training.
# ==========================================================================
def build_feature_row(user_inputs: dict) -> pd.DataFrame:
    """
    Convert raw user inputs into the one-row feature table the model expects.

    Steps:
      1. start every model feature at 0
      2. fill in the numeric values
      3. switch on the correct one-hot column for each chosen category
         (if the chosen category was the 'dropped' baseline during training,
          no column is switched on — exactly matching drop_first=True)
    """
    row = {col: 0.0 for col in FEATURES}          # 1) all features start at zero

    for f in NUMERIC:                             # 2) numeric values straight in
        row[f] = float(user_inputs[f])

    for cat in CATEG:                             # 3) one-hot the categories
        chosen = user_inputs[cat]
        col_name = f"{cat}_{chosen}"              # e.g. "Soil_Type_Loam"
        if col_name in row:                       # baseline category has no column -> stays 0
            row[col_name] = 1.0

    # build a one-row DataFrame in the exact training column order
    return pd.DataFrame([row])[FEATURES]


def predict(model_name: str, user_inputs: dict):
    """Preprocess the inputs, run the chosen model, and return the predicted yield."""
    X = build_feature_row(user_inputs)
    # scale ONLY the numeric columns, using the scaler fitted on the training data
    X[NUMERIC] = scaler.transform(X[NUMERIC])
    yhat = float(models[model_name].predict(X)[0])
    return yhat, X


def driver_ranking(model_name: str, X_scaled: pd.DataFrame) -> pd.Series:
    """
    Return the factors that most influenced THIS prediction (top drivers).

    * Linear Regression: contribution = coefficient x scaled input value
      (this is specific to the current prediction and shows direction +/-).
    * Random Forest / XGBoost: use the model's overall feature importance
      (which inputs the model relies on most in general).
    """
    model = models[model_name]
    if hasattr(model, "coef_"):                                   # Linear Regression
        contrib = pd.Series(model.coef_ * X_scaled.iloc[0].values, index=FEATURES)
        return contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)
    else:                                                         # tree-based models
        imp = pd.Series(model.feature_importances_, index=FEATURES)
        return imp.sort_values(ascending=False).head(8)


# ==========================================================================
# INPUT VALIDATION  (mirrors the InputData.validate() method in Chapter 3.12)
#   Returns two lists:
#     errors   -> values that are physically impossible; prediction is BLOCKED
#     warnings -> values outside the training range; prediction is ALLOWED but
#                 flagged, because the model is then extrapolating (less reliable)
# ==========================================================================
# Hard physical limits — a value outside these is simply not possible.
PHYSICAL_LIMITS = {
    "Soil_pH": (0, 14), "Rainfall_mm": (0, None), "Temperature_C": (-10, 60),
    "Humidity_pct": (0, 100), "Fertilizer_Used_kg": (0, None),
    "Pesticides_Used_kg": (0, None), "Planting_Density": (0, None),
}
NICE = {"Soil_pH": "Soil pH", "Rainfall_mm": "Rainfall", "Temperature_C": "Temperature",
        "Humidity_pct": "Humidity", "Fertilizer_Used_kg": "Fertilizer",
        "Pesticides_Used_kg": "Pesticides", "Planting_Density": "Planting density"}


def validate_inputs(user_inputs: dict):
    """Check every numeric input; return (errors, warnings) as lists of messages."""
    errors, warnings = [], []
    for f in NUMERIC:
        val = user_inputs[f]
        lo, hi = PHYSICAL_LIMITS[f]
        # 1) physically impossible -> hard error
        if (lo is not None and val < lo) or (hi is not None and val > hi):
            limit_txt = f"{lo} to {hi}" if hi is not None else f"{lo} or above"
            errors.append(f"**{NICE[f]}** = {val} is impossible (must be {limit_txt}).")
            continue
        # 2) outside the range the model was trained on -> extrapolation warning
        tlo, thi = NUM_RANGES[f]
        if val < tlo or val > thi:
            warnings.append(f"**{NICE[f]}** = {val} is outside the training range "
                            f"({tlo:g}–{thi:g}); the prediction is an extrapolation.")
    return errors, warnings


# ==========================================================================
# DOWNLOADABLE PREDICTION REPORT
#   Builds a one-page PDF (or plain text if reportlab is unavailable) that the
#   user can download and keep as a record of the prediction.
# ==========================================================================
def build_pdf_report(model_name, predicted, rmse, inputs, drivers) -> bytes:
    """Return a one-page PDF prediction report as raw bytes."""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=18*mm, bottomMargin=18*mm,
                            leftMargin=18*mm, rightMargin=18*mm)
    styles = getSampleStyleSheet()
    navy = colors.HexColor("#1F3A5F"); teal = colors.HexColor("#2A9D8F")
    title = ParagraphStyle("t", parent=styles["Title"], textColor=navy, fontSize=18)
    h = ParagraphStyle("h", parent=styles["Heading2"], textColor=teal, fontSize=12)
    body = styles["BodyText"]
    story = []

    story.append(Paragraph("Maize Yield Prediction Report", title))
    story.append(Paragraph("Crop Yield Prediction Based on Weather and Soil Data", body))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body))
    story.append(Spacer(1, 8*mm))

    # headline result
    story.append(Paragraph("Result", h))
    story.append(Paragraph(
        f"<b>Predicted yield: {predicted:.1f} ton/ha</b> "
        f"(expected range {predicted-rmse:.1f} – {predicted+rmse:.1f} ton/ha)", body))
    story.append(Paragraph(f"Model used: {model_name}", body))
    story.append(Spacer(1, 6*mm))

    # inputs table
    story.append(Paragraph("Inputs provided", h))
    rows = [["Variable", "Value"]] + [[k, str(v)] for k, v in inputs.items()]
    tbl = Table(rows, colWidths=[70*mm, 90*mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF3F5")]),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 6*mm))

    # top drivers
    story.append(Paragraph("Main influencing factors", h))
    for name, val in drivers.head(5).items():
        story.append(Paragraph(f"• {name}: {val:+.3f}", body))

    story.append(Spacer(1, 8*mm))
    story.append(Paragraph(
        "<i>This prediction is produced by a machine-learning model trained on historical "
        "maize data and is intended as decision support, not a guarantee.</i>",
        ParagraphStyle("d", parent=body, fontSize=8, textColor=colors.HexColor("#777777"))))

    doc.build(story)
    return buf.getvalue()


def build_text_report(model_name, predicted, rmse, inputs, drivers) -> bytes:
    """Fallback plain-text report if reportlab is not installed."""
    lines = ["MAIZE YIELD PREDICTION REPORT",
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
             f"Predicted yield: {predicted:.1f} ton/ha",
             f"Expected range : {predicted-rmse:.1f} - {predicted+rmse:.1f} ton/ha",
             f"Model used     : {model_name}", "", "INPUTS:"]
    lines += [f"  {k}: {v}" for k, v in inputs.items()]
    lines += ["", "TOP FACTORS:"] + [f"  {n}: {v:+.3f}" for n, v in drivers.head(5).items()]
    return "\n".join(lines).encode("utf-8")


# ==========================================================================
# SIDEBAR — navigation between the three pages + the model selector.
# ==========================================================================
st.sidebar.title("🌽 Maize Yield Predictor")
st.sidebar.caption("Crop Yield Prediction Based on Weather and Soil Data")
page = st.sidebar.radio("Navigate", ["Predict Yield", "Model Performance", "Prediction History"])

# The user chooses which trained model to use; the best one is pre-selected.
model_names = list(models.keys())
default_idx = model_names.index(META["best_model"]) if META["best_model"] in model_names else 0
model_choice = st.sidebar.selectbox("Prediction model", model_names, index=default_idx)
st.sidebar.metric(f"{model_choice} test R²", f"{RESULTS[model_choice]['r2']:.3f}")
st.sidebar.caption(f"Typical error ± {RESULTS[model_choice]['rmse']:.2f} ton/ha (test RMSE)")


# ==========================================================================
# PAGE 1 — PREDICT YIELD
# ==========================================================================
if page == "Predict Yield":
    st.title("Predict Maize Yield")
    st.write("Enter the plot's weather, soil and management values, then click **Predict**. "
             "Each field shows the range seen in the training data as a guide.")

    # Inputs are grouped into three columns for a clean, form-like layout.
    # We use number_input (typed values) so the validation step is meaningful:
    # a user can enter anything, and the app checks it before predicting.
    c1, c2, c3 = st.columns(3)
    inputs = {}

    def num(label, key):
        """A numeric input pre-filled with the mid-range value, with the valid range shown as help."""
        lo, hi = NUM_RANGES[key]
        return st.number_input(label, value=_mid(NUM_RANGES[key]), step=1.0,
                               help=f"Typical training range: {lo:g} – {hi:g}")

    with c1:
        st.subheader("🌦️ Weather")
        inputs["Rainfall_mm"]   = num("Rainfall (mm)", "Rainfall_mm")
        inputs["Temperature_C"] = num("Temperature (°C)", "Temperature_C")
        inputs["Humidity_pct"]  = num("Humidity (%)", "Humidity_pct")

    with c2:
        st.subheader("🪨 Soil")
        inputs["Soil_pH"]   = st.number_input("Soil pH", value=_mid(NUM_RANGES["Soil_pH"]),
                                              step=0.1, help="Typical training range: 5.5 – 7.5")
        inputs["Soil_Type"] = st.selectbox("Soil type", CAT_OPTS["Soil_Type"])
        inputs["Region"]    = st.selectbox("Region", CAT_OPTS["Region"])

    with c3:
        st.subheader("🚜 Management")
        inputs["Fertilizer_Used_kg"] = num("Fertilizer used (kg)", "Fertilizer_Used_kg")
        inputs["Pesticides_Used_kg"] = num("Pesticides used (kg)", "Pesticides_Used_kg")
        inputs["Planting_Density"]   = num("Planting density", "Planting_Density")
        inputs["Irrigation"]    = st.selectbox("Irrigation", CAT_OPTS["Irrigation"])
        inputs["Previous_Crop"] = st.selectbox("Previous crop", CAT_OPTS["Previous_Crop"])

    st.divider()

    # The predict button triggers validation first, then the model run.
    if st.button("🔮 Predict Yield", type="primary", use_container_width=True):
        # --- STEP 1: validate the inputs (Chapter 3.12 InputData.validate) ---
        errors, warnings = validate_inputs(inputs)
        for w in warnings:
            st.warning("⚠️ " + w)
        if errors:
            # impossible values -> show each problem and STOP (do not predict)
            for e in errors:
                st.error("❌ " + e)
            st.stop()

        # --- STEP 2: run the model ------------------------------------------
        yhat, X_scaled = predict(model_choice, inputs)
        rmse = RESULTS[model_choice]["rmse"]          # typical error band
        drivers = driver_ranking(model_choice, X_scaled)

        # --- headline result -------------------------------------------------
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted yield", f"{yhat:.1f} ton/ha")
        r2.metric("Typical error range", f"± {rmse:.1f} ton/ha")
        r3.metric("Model used", model_choice)
        st.caption(f"Expected yield is roughly **{yhat-rmse:.1f} – {yhat+rmse:.1f} ton/ha** "
                   f"(prediction ± the model's test RMSE).")

        # --- what drove this prediction -------------------------------------
        st.subheader("Main factors influencing this prediction")
        st.bar_chart(drivers)
        if hasattr(models[model_choice], "coef_"):
            st.caption("Bars show each factor's contribution to *this* prediction "
                       "(positive raises the yield, negative lowers it).")
        else:
            st.caption("Bars show the factors this model relies on most overall.")

        # --- save to history + offer a downloadable report ------------------
        db_save(model_choice, yhat, inputs)
        st.success("Prediction saved to history. ✅")

        # build the report (PDF if reportlab is available, otherwise plain text)
        if REPORTLAB_OK:
            data = build_pdf_report(model_choice, yhat, rmse, inputs, drivers)
            st.download_button("⬇️ Download prediction report (PDF)", data=data,
                               file_name="maize_yield_prediction.pdf", mime="application/pdf")
        else:
            data = build_text_report(model_choice, yhat, rmse, inputs, drivers)
            st.download_button("⬇️ Download prediction report (TXT)", data=data,
                               file_name="maize_yield_prediction.txt", mime="text/plain")


# ==========================================================================
# PAGE 2 — MODEL PERFORMANCE
# ==========================================================================
elif page == "Model Performance":
    st.title("Model Performance")
    st.write("How the three trained models compare on the held-out test set "
             "(20% of the maize data the models never saw during training).")

    # metrics table
    perf = pd.DataFrame(RESULTS).T[["rmse", "mae", "r2", "accuracy_pct", "cv_r2_mean"]]
    perf.columns = ["RMSE ↓", "MAE ↓", "R² ↑", "Accuracy % ↑", "CV R² ↑"]
    perf = perf.sort_values("RMSE ↓")
    st.dataframe(perf.style.format("{:.3f}").highlight_min(subset=["RMSE ↓", "MAE ↓"], color="#c8e6c9")
                            .highlight_max(subset=["R² ↑", "Accuracy % ↑", "CV R² ↑"], color="#c8e6c9"),
                 use_container_width=True)

    st.success(f"🏆 Best model by RMSE: **{perf.index[0]}**")

    # bar charts of the key metrics
    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("Error — lower is better")
        st.bar_chart(perf[["RMSE ↓", "MAE ↓"]])
    with cc2:
        st.subheader("R² — higher is better")
        st.bar_chart(perf[["R² ↑"]])

    # show the saved comparison figure if it exists
    fig = PROJ / "outputs" / "figures" / "fig8_model_comparison.png"
    if fig.exists():
        st.image(str(fig), caption="Model comparison (from training pipeline)", use_container_width=True)

    st.info("On this dataset the three models perform almost identically (R² ≈ 0.98). "
            "Yield is close to a linear function of fertiliser and rainfall, so the simple "
            "Linear Regression baseline matches the ensembles — a sign the ensembles are not "
            "overfitting.")


# ==========================================================================
# PAGE 3 — PREDICTION HISTORY
# ==========================================================================
elif page == "Prediction History":
    st.title("Prediction History")
    hist = db_history()

    if hist.empty:
        st.info("No predictions yet. Make one on the **Predict Yield** page.")
    else:
        st.write(f"**{len(hist)}** prediction(s) saved.")
        # show the core columns; the raw inputs stay in the JSON column
        show = hist[["id", "created_at", "model_used", "predicted_yield"]].copy()
        show["predicted_yield"] = show["predicted_yield"].round(1)
        st.dataframe(show, use_container_width=True, hide_index=True)

        # a quick trend chart of predicted yields over time
        st.subheader("Predicted yields over time")
        st.line_chart(hist.sort_values("id").set_index("id")["predicted_yield"])

        if st.button("🗑️ Clear history"):
            db_clear()
            st.rerun()      # refresh the page so the cleared table shows immediately
