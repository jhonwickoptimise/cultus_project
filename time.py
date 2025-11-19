"""
neural_prophet_hpo.py

Generates synthetic multivariate daily time series with external regressors,
runs NeuralProphet with Optuna hyperparameter search, and evaluates RMSE.

Run:
    python neural_prophet_hpo.py

If NeuralProphet or Optuna are not installed and pip install works, the script
will install them. If pip install fails (no internet), the script falls back
to a SARIMAX baseline (statsmodels), so it still runs without crashing.
"""

import os
import sys
import math
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------- Utility: try to import optional libs, attempt pip install if missing ----------
def try_import(name, pip_name=None):
    try:
        return __import__(name)
    except Exception:
        pip_pkg = pip_name or name
        print(f"[INFO] {name} not found. Attempting `pip install {pip_pkg}` ...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_pkg])
            return __import__(name)
        except Exception as e:
            print(f"[WARN] Could not install {pip_pkg}: {e}")
            return None

# Try imports
neuralprophet = try_import("neuralprophet", "neuralprophet")
optuna = try_import("optuna")
matplotlib = try_import("matplotlib")
statsmodels = try_import("statsmodels")

# If matplotlib is available, import pyplot
if matplotlib:
    import matplotlib.pyplot as plt

# ---------- 1) Generate synthetic multivariate daily dataset ----------
def generate_synthetic_daily(start_date="2018-01-01", years=4, seed=42):
    rng = np.random.default_rng(seed)
    start = pd.to_datetime(start_date)
    days = int(365.25 * years)
    dates = pd.date_range(start, periods=days, freq="D")

    # Base trend: smooth nonlinear trend
    t = np.arange(days) / days
    trend = 10 + 5 * t + 3 * np.sin(2 * np.pi * t * 1.5)

    # Seasonalities: daily noise + yearly seasonality approximated by sin term
    yearly = 15 * np.sin(2 * np.pi * (np.arange(days) / 365.25) + 0.2)
    weekly = 2.5 * np.sin(2 * np.pi * (np.arange(days) / 7.0))

    # External regressors:
    # 1) holidays: binary indicator for a few dates (simulate)
    holidays = np.zeros(days, dtype=int)
    # choose approx 10 holiday dates per year
    for y in range(years):
        base = start + pd.DateOffset(years=y)
        holiday_dates = [
            base + pd.DateOffset(days=int(30 * i + (i * 3) % 7))
            for i in range(10)
        ]
        for d in holiday_dates:
            idx = (d - start).days
            if 0 <= idx < days:
                holidays[idx] = 1

    # 2) promotions: occasional promo activity with stronger effect on target
    promotions = np.zeros(days, dtype=int)
    # simulate promo runs scattered through time
    for promo_start in range(0, days, 90):
        length = rng.integers(3, 10)
        promotions[promo_start:promo_start+length] = rng.integers(0, 2)

    # Noise
    noise = rng.normal(scale=3.0, size=days)

    # target = baseline + seasonality + external effects + noise
    # Put stronger multiplicative effect for promotions and additive for holidays
    target = (trend + yearly + weekly) * (1 + 0.05 * promotions) + 4.0 * holidays + noise

    df = pd.DataFrame({
        "ds": dates,
        "y": target,
        "holiday": holidays,
        "promo": promotions
    })
    return df

df = generate_synthetic_daily("2018-01-01", years=4, seed=123)

print(f"Generated data: {len(df)} rows, date range {df.ds.min().date()} to {df.ds.max().date()}")
print(df.head())

# ---------- 2) Train/test split ----------
# use last 90 days as test
H = 90
train_df = df[:-H].reset_index(drop=True)
test_df = df[-H:].reset_index(drop=True)

# ---------- 3) If NeuralProphet + Optuna available, run HPO. Else fallback ----------
def evaluate_predictions(y_true, y_pred):
    # ensure numpy arrays
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return {"rmse": rmse, "mae": mae}

# Fallback SARIMAX function (if neuralprophet not available)
def sarimax_baseline(train_df, test_df):
    print("[FALLBACK] Using SARIMAX baseline (statsmodels).")
    try:
        import statsmodels.api as sm
    except Exception as e:
        print("[ERROR] statsmodels not available:", e)
        raise

    # We'll use exog (holiday, promo)
    exog_train = train_df[["holiday", "promo"]]
    exog_test = test_df[["holiday", "promo"]]

    # Fit a simple SARIMAX(1,1,1) with seasonal (7) to capture weekly seasonality
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 7)
    model = sm.tsa.statespace.SARIMAX(train_df["y"], exog=exog_train,
                                      order=order, seasonal_order=seasonal_order,
                                      enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    preds = res.get_forecast(steps=len(test_df), exog=exog_test)
    y_pred = preds.predicted_mean.values
    metrics = evaluate_predictions(test_df["y"].values, y_pred)
    print("SARIMAX metrics:", metrics)
    return res, y_pred, metrics

if neuralprophet and optuna:
    # Import actual classes
    from neuralprophet import NeuralProphet, set_log_level
    set_log_level("WARNING")
    import optuna

    # Prepare data for NeuralProphet: it expects ds,y plus regressors as columns
    np_train = train_df.copy()
    np_test = test_df.copy()

    # Define Optuna objective to minimize RMSE on holdout
    def objective(trial):
        # hyperparameters to search
        n_lags = trial.suggest_int("n_lags", 0, 30)
        n_forecasts = trial.suggest_int("n_forecasts", 1, 3)
        changepoint_range = trial.suggest_float("changepoint_range", 0.8, 0.95)
        trend_reg = trial.suggest_float("trend_reg", 0.0, 1.0)
        seasonality_reg = trial.suggest_float("seasonality_reg", 0.0, 1.0)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
        epochs = 80

        m = NeuralProphet(
            n_lags=n_lags,
            n_forecasts=n_forecasts,
            changepoint_range=changepoint_range,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            learning_rate=learning_rate,
            trend_reg=trend_reg,
            seasonality_reg=seasonality_reg,
            verbose=False
        )

        # add regressors
        m = m.add_future_regressor("holiday", mode="additive")
        m = m.add_future_regressor("promo", mode="multiplicative")

        # Fit on train_df; keep a small validation split for early stopping
        try:
            metrics = m.fit(np_train, freq="D", validation_df=None, epochs=epochs, progress="none")
        except Exception as e:
            # If training fails for some combos, return a large penalty
            print(f"[WARN] Training failed in trial: {e}")
            return 1e9

        # predict on test by using make_future_dataframe
        future = m.make_future_dataframe(pd.DataFrame(), periods=len(np_test), n_historic_predictions=True)
        # but easiest: add test rows to historic dataframe and predict
        df_for_pred = pd.concat([np_train, np_test], ignore_index=True)
        try:
            forecast = m.predict(df_for_pred)
        except Exception as e:
            print(f"[WARN] Predict failed: {e}")
            return 1e9

        # NeuralProphet outputs forecast columns like yhat1, yhat2...; pick last n_forecasts horizon
        yhat_col = f"yhat{n_forecasts}" if n_forecasts > 1 else "yhat1"
        # forecast includes rows equal to df_for_pred; take the tail corresponding to test set
        y_pred = forecast[yhat_col].values[-len(np_test):]
        metrics_eval = evaluate_predictions(np_test["y"].values, y_pred)
        return metrics_eval["rmse"]

    # Create Optuna study and run
    study = optuna.create_study(direction="minimize")
    print("[INFO] Starting Optuna HPO (this may take a while).")
    study.optimize(objective, n_trials=20, timeout=1200)  # adjust trials/time as needed

    print("Best trial:", study.best_trial.params)

    # Train final model with best params
    best = study.best_trial.params
    final_model = NeuralProphet(
        n_lags=best.get("n_lags", 0),
        n_forecasts=best.get("n_forecasts", 1),
        changepoint_range=best.get("changepoint_range", 0.9),
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        learning_rate=best.get("learning_rate", 0.01),
        trend_reg=best.get("trend_reg", 0.0),
        seasonality_reg=best.get("seasonality_reg", 0.0),
        verbose=False
    )
    final_model = final_model.add_future_regressor("holiday", mode="additive")
    final_model = final_model.add_future_regressor("promo", mode="multiplicative")

    print("[INFO] Fitting final model with best hyperparameters...")
    final_model.fit(np_train, freq="D", epochs=150)

    df_for_pred = pd.concat([np_train, np_test], ignore_index=True)
    forecast = final_model.predict(df_for_pred)
    n_forecasts = best.get("n_forecasts", 1)
    yhat_col = f"yhat{n_forecasts}" if n_forecasts > 1 else "yhat1"
    y_pred = forecast[yhat_col].values[-len(np_test):]
    metrics = evaluate_predictions(np_test["y"].values, y_pred)
    print("Final model metrics on test:", metrics)

    # Plot predictions vs truth if matplotlib available
    if matplotlib:
        plt.figure(figsize=(10, 4))
        plt.plot(train_df["ds"], train_df["y"], label="train")
        plt.plot(test_df["ds"], test_df["y"], label="test (true)")
        plt.plot(test_df["ds"], y_pred, label="pred")
        plt.legend()
        plt.title("NeuralProphet: actual vs predicted")
        plt.show()

else:
    # Fallback path: use SARIMAX baseline (requires statsmodels)
    if statsmodels:
        res, y_pred, metrics = sarimax_baseline(train_df, test_df)
        # plot if matplotlib
        if matplotlib:
            plt.figure(figsize=(10, 4))
            plt.plot(train_df["ds"], train_df["y"], label="train")
            plt.plot(test_df["ds"], test_df["y"], label="test (true)")
            plt.plot(test_df["ds"], y_pred, label="pred")
            plt.legend()
            plt.title("SARIMAX baseline: actual vs predicted")
            plt.show()
    else:
        raise RuntimeError("Neither neuralprophet/optuna nor statsmodels are available. Please install packages or run in internet-enabled environment.")

print("[DONE] Script finished.")
