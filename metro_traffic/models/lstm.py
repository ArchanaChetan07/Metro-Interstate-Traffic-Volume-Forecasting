"""Univariate LSTM for hourly traffic volume."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from metro_traffic.evaluate import regression_metrics


def create_sequences(data: np.ndarray, time_steps: int = 24):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.asarray(X), np.asarray(y)


def _build_lstm(input_steps: int, seed: int = 42):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM

    tf.random.set_seed(seed)
    np.random.seed(seed)
    model = Sequential(
        [
            LSTM(50, return_sequences=True, input_shape=(input_steps, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm_on_series(
    train_series,
    *,
    lookback: int = 24,
    epochs: int = 5,
    batch_size: int = 32,
    seed: int = 42,
    verbose: int = 0,
):
    """Univariate LSTM trained only on the provided train series values."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(np.asarray(train_series, dtype=float).reshape(-1, 1))
    X, y = create_sequences(scaled, time_steps=lookback)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = _build_lstm(lookback, seed=seed)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=False)
    return model, scaler


def lstm_onestep_forecast(model, scaler, series, *, start_idx: int, steps: int, lookback: int = 24) -> np.ndarray:
    """One-step-ahead predictions using true history windows only (no y_t leakage)."""
    s = np.asarray(series, dtype=float).ravel()
    preds = []
    for t in range(start_idx, start_idx + steps):
        window = s[t - lookback : t].reshape(-1, 1)
        scaled = scaler.transform(window).reshape(1, lookback, 1)
        yhat_s = float(model.predict(scaled, verbose=0)[0, 0])
        yhat = float(scaler.inverse_transform([[yhat_s]])[0, 0])
        preds.append(yhat)
    return np.asarray(preds, dtype=float)


def series_metrics(y_true, y_pred) -> dict[str, float]:
    return regression_metrics(y_true, y_pred)
