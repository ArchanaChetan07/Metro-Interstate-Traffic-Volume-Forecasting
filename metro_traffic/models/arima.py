"""ARIMA and ADF stationarity helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from metro_traffic.evaluate import regression_metrics


@dataclass
class ADFResult:
    statistic: float
    pvalue: float


def run_adf(series) -> ADFResult:
    stat, pvalue, *_ = adfuller(np.asarray(series, dtype=float))
    return ADFResult(statistic=float(stat), pvalue=float(pvalue))


def fit_arima(train_series, order=(4, 1, 2)):
    model = ARIMA(np.asarray(train_series, dtype=float), order=order)
    return model.fit()


def arima_forecast(fit, steps: int) -> np.ndarray:
    """Open-loop multi-step forecast (no feedback from observations)."""
    return np.asarray(fit.forecast(steps=steps), dtype=float)


def arima_onestep_forecast(fit, test_series) -> np.ndarray:
    """One-step-ahead ARIMA forecasts over test_series via append(refit=False)."""
    res = fit
    preds = []
    for x in np.asarray(test_series, dtype=float).ravel():
        f = float(np.asarray(res.forecast(steps=1)).ravel()[0])
        preds.append(f)
        res = res.append([x], refit=False)
    return np.asarray(preds, dtype=float)


def series_metrics(y_true, y_pred) -> dict[str, float]:
    return regression_metrics(y_true, y_pred)
