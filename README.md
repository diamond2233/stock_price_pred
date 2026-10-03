# Stock Price Predictor

Predicts a stock's next-day closing price from its recent trading history,
comparing a **Linear Regression** baseline against a **Random Forest
Regressor**, and against the honest "naive" baseline of just guessing
tomorrow's close equals today's close.

The repo ships with a synthetic-but-realistic 3-year daily OHLCV series
(generated with geometric Brownian motion + seasonal drift, see
`data/synthetic_prices.csv`) so it runs anywhere with no API keys and no
internet access. Swapping in real market data is a one-line change — see
below.

## Result

| Model | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|
| Linear Regression | 29.28 | 23.09 | 0.832 | 1.32% |
| Random Forest | 29.55 | 23.75 | 0.829 | 1.35% |
| Naive (no-change) baseline | 28.94 | 22.83 | 0.836 | 1.30% |

**The honest takeaway:** both models land essentially on top of the naive
"no-change" baseline. This isn't a bug — it's the expected, well-documented
result for short-horizon daily price prediction (asset prices behave close
to a random walk, so next-day close is dominated by today's close). It's
worth stating plainly in a write-up or interview rather than glossing over:
it shows the model is evaluated honestly with a chronological
train/test split and a proper baseline, rather than tuned until a metric
looks good.

## Project structure

```
stock-price-predictor/
├── stock_predictor.py            # data gen, features, both models, evaluation
├── requirements.txt
├── data/
│   └── synthetic_prices.csv      # generated OHLCV series (cached, reproducible)
└── outputs/
    ├── predictions_vs_actual.png # actual vs. both models' predictions, test period
    ├── feature_importance.png    # what the random forest actually uses
    ├── test_predictions.csv
    └── metrics.json
```

## How it works

1. **Data** — daily Open/High/Low/Close/Volume for ~3 years (756 trading
   days), generated once and cached to CSV for reproducibility.
2. **Features** — lagged closes (1/2/3/5/10 days), moving averages
   (5/10/20-day), rolling volatility, 5-day momentum, daily trading range,
   volume change, and day-of-week.
3. **Split** — chronological (not random/shuffled) 85/15 train/test split,
   since shuffling time-series data leaks future information into training.
4. **Models** — `LinearRegression` and `RandomForestRegressor` from
   scikit-learn, evaluated with RMSE, MAE, R², and MAPE against each other
   and against the naive baseline.

## Running it

```bash
pip install -r requirements.txt
python stock_predictor.py
```

## Using real market data instead

Any CSV with `Date, Open, High, Low, Close, Volume` columns works. Export
one from Yahoo Finance, NSE/BSE Bhavcopy, Alpha Vantage, or (on a machine
with internet access):

```python
import yfinance as yf
yf.download("AAPL", period="3y").reset_index().to_csv("data/real_prices.csv", index=False)
```

Then set `CSV_PATH` at the top of `stock_predictor.py` to point at that
file — the feature engineering, models, and evaluation all work unchanged.

## Tech stack

Python, scikit-learn, pandas, NumPy, Matplotlib.
