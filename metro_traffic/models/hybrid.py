"""Residual Hybrid: ARIMA on volume + LSTM on ARIMA residuals.

Architecture (leakage-safe):
  1) Fit ARIMA on the **train** series only.
  2) Train LSTM on **train ARIMA residuals** only.
  3) One-step walk-forward test: yhat = ARIMA_one_step + LSTM_residual_one_step.
     Observations are appended only **after** each forecast (not used to form yhat_t).

This is not an ensemble average; residual LSTM is trained on ARIMA residuals.
"""

from __future__ import annotations

import numpy as np

from metro_traffic.evaluate import regression_metrics
from metro_traffic.models.arima import fit_arima
from metro_traffic.models.lstm import train_lstm_on_series


def train_hybrid(
    train_series,
    *,
    order=(4, 1, 2),
    lookback: int = 24,
    epochs: int = 5,
    seed: int = 42,
    verbose: int = 0,
):
    """Fit ARIMA on train; LSTM on ARIMA residuals (train only)."""
    arima_fit = fit_arima(train_series, order=order)
    fitted = np.asarray(arima_fit.fittedvalues, dtype=float)
    train = np.asarray(train_series, dtype=float)
    if len(fitted) < len(train):
        pad = len(train) - len(fitted)
        fitted = np.concatenate([train[:pad], fitted])
    residuals = train - fitted
    lstm_model, scaler = train_lstm_on_series(
        residuals, lookback=lookback, epochs=epochs, seed=seed, verbose=verbose
    )
    return {
        "arima_fit": arima_fit,
        "lstm_model": lstm_model,
        "scaler": scaler,
        "lookback": lookback,
        "train_residuals": residuals,
    }


def hybrid_onestep_forecast(bundle, full_series, *, start_idx: int, steps: int) -> np.ndarray:
    """Hybrid one-step: ARIMA one-step + LSTM residual one-step (leakage-safe)."""
    lookback = bundle["lookback"]
    ar_fit = bundle["arima_fit"]
    s = np.asarray(full_series, dtype=float).ravel()
    res_ar = ar_fit
    resid_hist = list(np.asarray(bundle["train_residuals"], dtype=float).ravel())
    preds = []
    for t in range(start_idx, start_idx + steps):
        ar_f = float(np.asarray(res_ar.forecast(steps=1)).ravel()[0])
        window = np.asarray(resid_hist[-lookback:], dtype=float).reshape(-1, 1)
        scaled = bundle["scaler"].transform(window).reshape(1, lookback, 1)
        rhat_s = float(bundle["lstm_model"].predict(scaled, verbose=0)[0, 0])
        rhat = float(bundle["scaler"].inverse_transform([[rhat_s]])[0, 0])
        yhat = ar_f + rhat
        preds.append(yhat)
        y_obs = float(s[t])
        res_ar = res_ar.append([y_obs], refit=False)
        resid_hist.append(y_obs - ar_f)
    return np.asarray(preds, dtype=float)


def series_metrics(y_true, y_pred) -> dict[str, float]:
    return regression_metrics(y_true, y_pred)
