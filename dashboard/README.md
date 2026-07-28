# Maize Yield Prediction — Interactive Dashboard

A Streamlit dashboard that sits on top of the trained models and lets anyone predict
maize yield from weather, soil and management inputs — no coding required.

## What it does
- **Predict Yield** — enter a plot's conditions, pick a model, get the predicted yield
  (in ton/ha) with a typical-error band and the factors driving that prediction.
- **Model Performance** — compares Linear Regression, Random Forest and XGBoost on the
  test set (RMSE, MAE, R², accuracy).
- **Prediction History** — every prediction is saved to a small SQLite database
  (`predictions.db`) and shown as a table + trend chart. This mirrors the
  `PredictionRecord` table and `DatabaseManager` class described in Chapter 3.12.

## How to run

1. Install dependencies (from the project root):
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure the trained models exist. If `outputs/models/` is empty, build them first:
   ```bash
   python outputs/train_models.py
   ```

3. Launch the dashboard (from the project root):
   ```bash
   streamlit run dashboard/app.py
   ```

4. Your browser opens automatically at <http://localhost:8501>.

## Files
- `app.py` — the dashboard (fully commented).
- `predictions.db` — created automatically on first run; holds the prediction history.

## How it connects to the rest of the project
`outputs/train_models.py` trains the three models and saves them to `outputs/models/`
(`*.joblib`) together with `scaler.joblib` and `model_metadata.json`. The dashboard loads
those files and applies the **exact same** preprocessing (one-hot encoding + standardisation)
used during training, so its predictions match the trained models precisely.
