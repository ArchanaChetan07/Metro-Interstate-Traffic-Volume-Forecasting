![CI](https://github.com/ArchanaChetan07/Metro-Interstate-Traffic-Volume-Forecasting/actions/workflows/ci.yml/badge.svg)

Time-series traffic volume forecasting — 5-model comparison (Linear / GBR / ARIMA / LSTM / Hybrid) on UCI I-94 Metro Interstate data, 168h horizon — Python, scikit-learn, statsmodels, TensorFlow; R Quarto for EDA only.

Verified: GradientBoosting R² **0.942** (tabular test) · LSTM R² **0.966** · Hybrid residual R² **0.932** · Linear baseline R² **0.381** — ADF **−21.89** (p=0) — **17** pytest passing. Prior notebook “Hybrid R² 0.995” was **simulated** (sine + noise), not held-out I-94 forecasts.

```bash
docker build -t metro-traffic .
docker run --rm metro-traffic
# Full 5-model (slow; needs TensorFlow): pip install -r requirements.txt && PYTHONPATH=. python scripts/run_benchmark.py --full --lstm-epochs 5
```

## Overview

End-to-end forecasting for westbound **I-94** (Minneapolis–St. Paul) using the [UCI Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) dataset. Logic lives in the `metro_traffic/` package; notebooks / Quarto remain exploratory.

## Data

| Item | Detail |
|------|--------|
| Source | UCI I-94 Metro Interstate, 2012–2018 |
| Modeling window | 2015-06-11 → 2018-09-30 (aligned with project EDA) |
| Features (9) | `temp`, `rain_1h`, `snow_1h`, `clouds_all`, `is_holiday`, `hour`, `month`, `is_weekend`, `rush_hour` |
| Split | Chronological **70 / 15 / 15** (`shuffle=False`) |
| Horizon | Last **168** hours for series models (one-step walk-forward) |
| File | `data/Metro_Interstate_Traffic_Volume.csv.gz` |

## Stationarity Check

ADF on the hourly traffic series: statistic **−21.8925**, p-value **0.0** → reject unit root; series is stationary for the ARIMA component. Reproduced in `scripts/run_benchmark.py` → `artifacts/benchmark_report.json`.

## Models Compared

| Model | Approach |
|-------|----------|
| Linear Regression | Standardized numeric features + holiday / weekend / rush flags |
| Gradient Boosting | Same tabular features; `n_estimators=100`, `random_state=42` |
| ARIMA(4,1,2) | Univariate volume; **one-step** walk-forward over last 168h |
| LSTM | Univariate sequence (lookback 24); seed **42**, 5 epochs; one-step walk-forward |
| Hybrid | **ARIMA on train volume + LSTM on train ARIMA residuals**; one-step `yhat = ARIMA + residual LSTM` |

## Results (verified this session)

**Tabular models** — full chronological 15% test (`seed=42`):

| Model | MAE | RMSE | R² |
|-------|----:|-----:|---:|
| Linear Regression | 1385.3 | 1553.8 | **0.381** |
| Gradient Boosting | 311.6 | 475.7 | **0.942** |

**Series models** — last 168h **one-step** walk-forward (`seed=42`, LSTM/Hybrid epochs=5):

| Model | MAE | RMSE | R² |
|-------|----:|-----:|---:|
| ARIMA(4,1,2) | 459.8 | 673.7 | **0.885** |
| LSTM | 257.2 | 365.1 | **0.966** |
| Hybrid (ARIMA + residual LSTM) | 364.7 | 516.9 | **0.932** |

Open-loop 168h multi-step (no observational feedback) yields strongly negative R² for ARIMA/LSTM/Hybrid — not used as headline scores.

### Historic notebook claims (not verified)

Notebook/Quarto tables listed Hybrid R² **0.995**, LSTM **0.992**, GBR **0.978**, ARIMA **0.967**, LR **0.945**. Audit found those scores came from **simulated** `predictions = y_true + noise` on a sine wave, not from model forecasts on held-out I-94. See `artifacts/AUDIT.md`.

## Architecture Note on Hybrid

Hybrid is a **residual hybrid**, not an average of ARIMA and LSTM outputs:

1. Fit ARIMA(4,1,2) on **train** hourly volume only.  
2. Train LSTM on **train ARIMA residuals** only.  
3. At test time (one-step): `yhat_t = ARIMA_forecast_t + LSTM_residual_t`; append observation **after** predicting (no `y_t` in forming `yhat_t`).

Under this leakage-safe protocol Hybrid (0.932) does **not** exceed LSTM (0.966), so the old 0.995 figure cannot be attributed to a correctly evaluated hybrid.

## How to Run

**Fast path (LR + GBR + ADF) — Docker / CI**

```bash
docker build -t metro-traffic .
docker run --rm metro-traffic
# or locally:
pip install -r requirements-fast.txt
PYTHONPATH=. python scripts/run_benchmark.py
```

**Full 5-model comparison (manual; TensorFlow; several minutes)**

```bash
pip install -r requirements.txt
PYTHONPATH=. python scripts/run_benchmark.py --full --lstm-epochs 5
# writes artifacts/benchmark_report.json
```

CI runs pytest + fast benchmark only. LSTM / Hybrid / full ARIMA walk-forward are **not** retrained on every push.

**R:** `Project_EDA.qmd` is exploratory Quarto (tidyverse / fpp3). Production metrics path is Python.

**Notebooks:** `Metro_Interstate_Traffic_Volume.ipynb` / EDA artifacts remain exploratory; reproducible numbers come from `metro_traffic/` + `scripts/run_benchmark.py` (see `scripts/package_demo.py`).

## Tests

```bash
pip install -r requirements-fast.txt
PYTHONPATH=. pytest tests/ -v
```

**17** tests: feature engineering, metrics on synthetic cases, chronological / leakage guards, GBR/LR regression floors, plus legacy unit tests.

## Tech Stack

Python 3.10+, pandas, scikit-learn, statsmodels, TensorFlow/Keras (full path), pytest, Docker, GitHub Actions. R (Quarto) for EDA only.

## License

See repository license / UCI dataset terms for the underlying traffic data.
