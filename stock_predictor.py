"""
Stock Price Predictor
----------------------
Predicts the next day's closing price of a stock from its recent price
history, using two models:
  1. Linear Regression        - simple, interpretable baseline
  2. Random Forest Regressor  - captures non-linear patterns, usually stronger

Data:
This script ships with a synthetic-but-realistic daily OHLCV price series
(3 years, generated with geometric Brownian motion + seasonal drift + noise)
so the whole pipeline runs anywhere with zero API keys and no internet
access -- see `data/synthetic_prices.csv`.

To use REAL market data instead, drop any CSV with the columns
`Date, Open, High, Low, Close, Volume` (e.g. exported from Yahoo Finance,
NSE/BSE Bhavcopy, Alpha Vantage, or the `yfinance` package on a machine
that has internet access) into `data/` and set CSV_PATH below to point at
it. No other code changes are required.

Usage:
    python stock_predictor.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_SEED = 42
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "synthetic_prices.csv") 
TICKER_NAME = "SYNTH"

np.random.seed(RANDOM_SEED)


def generate_synthetic_prices(
    n_days=756, start_price=1500.0, path=CSV_PATH
):
    """
    Generate a realistic-looking daily OHLCV series with geometric Brownian
    motion, a mild upward drift, yearly seasonality, and volume that reacts
    to volatility -- then cache it to CSV so results are reproducible.
    """
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["Date"])

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    daily_drift = 0.0004
    daily_vol = 0.016
    seasonal = 0.05 * np.sin(2 * np.pi * np.arange(n_days) / 252)
    shocks = np.random.normal(daily_drift, daily_vol, n_days) + seasonal / n_days

    log_returns = shocks
    close = start_price * np.exp(np.cumsum(log_returns))

    daily_range = close * np.random.uniform(0.005, 0.02, n_days)
    open_ = close * (1 + np.random.normal(0, 0.004, n_days))
    high = np.maximum(open_, close) + daily_range * np.random.uniform(0.2, 0.6, n_days)
    low = np.minimum(open_, close) - daily_range * np.random.uniform(0.2, 0.6, n_days)
    volume = (1_000_000 * (1 + 8 * np.abs(log_returns) / daily_vol)).astype(int)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": open_.round(2),
            "High": high.round(2),
            "Low": low.round(2),
            "Close": close.round(2),
            "Volume": volume,
        }
    )
    df.to_csv(path, index=False)
    return df


def engineer_features(df):
    """Build lag / rolling-window / calendar features for next-day prediction."""
    df = df.sort_values("Date").reset_index(drop=True).copy()

    for lag in (1, 2, 3, 5, 10):
        df[f"close_lag_{lag}"] = df["Close"].shift(lag)

    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["volatility_10"] = df["Close"].pct_change().rolling(10).std()
    df["momentum_5"] = df["Close"] - df["Close"].shift(5)
    df["daily_range"] = df["High"] - df["Low"]
    df["volume_change"] = df["Volume"].pct_change()
    df["day_of_week"] = df["Date"].dt.dayofweek

    # target: next trading day's close
    df["target"] = df["Close"].shift(-1)

    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5", "close_lag_10",
    "ma_5", "ma_10", "ma_20", "volatility_10", "momentum_5",
    "daily_range", "volume_change", "day_of_week",
]


def chronological_split(df, test_frac=0.15):
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    print(f"[{name}] RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}  MAPE={mape:.2f}%")
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape_pct": mape}


def plot_predictions(dates, y_true, preds_by_model, path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, y_true, label="Actual", linewidth=2, color="black")
    for name, preds in preds_by_model.items():
        ax.plot(dates, preds, label=name, linewidth=1.5, alpha=0.85)
    ax.set_title(f"{TICKER_NAME}: Actual vs Predicted Next-Day Close (Test Set)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_feature_importance(model, feature_names, path):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    importances.plot(kind="barh", ax=ax, color="#2b6cb0")
    ax.set_title("Random Forest - Feature Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    raw = generate_synthetic_prices()
    df = engineer_features(raw)

    train_df, test_df = chronological_split(df)
    X_train, y_train = train_df[FEATURE_COLS], train_df["target"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["target"]
    print(f"Train rows: {len(train_df)}   Test rows: {len(test_df)}")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # naive baseline: "tomorrow's close = today's close"
    naive_pred = test_df["Close"].values

    results = {
        "linear_regression": evaluate("Linear Regression", y_test.values, lr_pred),
        "random_forest": evaluate("Random Forest", y_test.values, rf_pred),
        "naive_baseline": evaluate("Naive (no-change)", y_test.values, naive_pred),
    }

    plot_predictions(
        test_df["Date"],
        y_test.values,
        {"Linear Regression": lr_pred, "Random Forest": rf_pred},
        os.path.join(OUTPUT_DIR, "predictions_vs_actual.png"),
    )
    plot_feature_importance(rf, FEATURE_COLS, os.path.join(OUTPUT_DIR, "feature_importance.png"))

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    test_df.assign(lr_pred=lr_pred, rf_pred=rf_pred).to_csv(
        os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False
    )

    print("\nSaved metrics, plots, and prediction CSV to outputs/.")


if __name__ == "__main__":
    main()
