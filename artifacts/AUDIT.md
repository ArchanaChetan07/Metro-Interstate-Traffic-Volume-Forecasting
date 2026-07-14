# Audit notes (session verification)

## Claimed notebook metrics are simulated

Cells in `Metro_Interstate_Traffic_Volume.ipynb` / related modeling notebooks build:

```text
y_test_values = sine wave
predictions = y_test_values + np.random.normal(...)
```

and then report MAE / RMSE / R². Comments say "Simulated data for demonstration."
The Hybrid ARIMA+LSTM sketch exists elsewhere, but the headline metrics table was **not**
computed from those model outputs on held-out I-94 data.

**Do not treat** R² 0.945 / 0.978 / 0.967 / 0.992 / 0.995 as verified model scores.

## Reproduced (this session)

Source: `artifacts/benchmark_report.json` from
`python scripts/run_benchmark.py --full --lstm-epochs 5` (seed=42).

| Check | Result |
|-------|--------|
| ADF statistic | **-21.8925** (p=**0.0**) — matches claim ≈ -21.89 |
| LinearRegression R² | **0.381** on chronological 15% test |
| GradientBoosting R² | **0.942** on chronological 15% test |
| ARIMA(4,1,2) R² | **0.885** last-168h one-step walk-forward |
| LSTM R² | **0.966** last-168h one-step walk-forward (seed=42, 5 epochs) |
| Hybrid residual R² | **0.932** last-168h one-step walk-forward (seed=42, 5 epochs) |

Open-loop 168h multi-step ARIMA/LSTM/Hybrid (no observational feedback) produced
**negative** R² — expected for long open-loop drift. Walk-forward one-step is the
fair protocol used for series models in the verified table.

## Hybrid architecture + leakage check

Hybrid = **ARIMA on train volume** + **LSTM on train ARIMA residuals**;
test `yhat = ARIMA_one_step + LSTM_residual_one_step`.
Fit uses train only. Each step's target `y_t` is **not** used to form `yhat_t`
(observations append after predicting). Chronological cut; no random shuffle.

Under this protocol Hybrid (**0.932**) does **not** beat LSTM (**0.966**) —
further evidence the claimed Hybrid R² 0.995 was not a genuine residual-hybrid score.

## R usage

`Project_EDA.qmd` (+ Quarto-style EDA files) use **R / tidyverse / fpp3** for EDA.
The production forecast + metrics pipeline is **pure Python** (`metro_traffic/`).
