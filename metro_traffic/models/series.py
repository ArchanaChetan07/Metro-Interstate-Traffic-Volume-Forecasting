"""Backward-compatible re-exports for series models (ARIMA / LSTM / Hybrid)."""

from metro_traffic.models.arima import (
    ADFResult,
    arima_forecast,
    arima_onestep_forecast,
    fit_arima,
    run_adf,
    series_metrics,
)
from metro_traffic.models.hybrid import hybrid_onestep_forecast, train_hybrid
from metro_traffic.models.lstm import create_sequences, lstm_onestep_forecast, train_lstm_on_series

__all__ = [
    "ADFResult",
    "arima_forecast",
    "arima_onestep_forecast",
    "create_sequences",
    "fit_arima",
    "hybrid_onestep_forecast",
    "lstm_onestep_forecast",
    "run_adf",
    "series_metrics",
    "train_hybrid",
    "train_lstm_on_series",
]
