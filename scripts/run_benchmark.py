#!/usr/bin/env python3
"""Honest end-to-end traffic benchmark (fixed seeds).

Fast path (default): ADF + LinearRegression + GradientBoosting on chronological
held-out test (and last-168h slice).

Full path (--full): also ARIMA(4,1,2), LSTM, and residual Hybrid on a 168h
forecast horizon with train series excluding that horizon.

Historical notebook R² (0.945..0.995) came from sine+noise simulation cells and
are NOT reproduced here — see artifacts/benchmark_report.json → note.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metro_traffic.features import (  # noqa: E402
    FEATURE_COLS,
    assert_no_lag_nan,
    build_xy,
    clean_and_engineer,
    chronological_split,
    load_raw,
    make_hourly_series,
)
from metro_traffic.models.series import (  # noqa: E402
    arima_onestep_forecast,
    fit_arima,
    hybrid_onestep_forecast,
    lstm_onestep_forecast,
    run_adf,
    train_hybrid,
    train_lstm_on_series,
)
from metro_traffic.models.tabular import eval_model, train_gbr, train_linear  # noqa: E402


HORIZON = 168
SEED = 42


def _last_n(y: pd.Series | np.ndarray, n: int):
    arr = np.asarray(y, dtype=float).ravel()
    return arr[-n:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Include ARIMA/LSTM/Hybrid")
    parser.add_argument("--lstm-epochs", type=int, default=5)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "benchmark_report.json")
    args = parser.parse_args()

    np.random.seed(SEED)

    raw = load_raw()
    df = clean_and_engineer(raw)
    X, y = build_xy(df)
    assert_no_lag_nan(X)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = chronological_split(X, y)

    report: dict = {
        "note": (
            "Prior README numbers (Hybrid R²≈0.995 etc.) were produced by notebook "
            "cells that scored simulated predictions = y_true + Gaussian noise on a "
            "sine wave — not model forecasts on I-94 test data. This report replaces "
            "those with held-out evaluations from metro_traffic/."
        ),
        "n_rows_window": int(len(df)),
        "features": FEATURE_COLS,
        "split": {
            "train": int(len(X_tr)),
            "val": int(len(X_va)),
            "test": int(len(X_te)),
            "scheme": "chronological 70/15/15 shuffle=False",
        },
        "models": {},
    }

    # --- ADF on hourly series ---
    hourly = make_hourly_series(df)
    adf = run_adf(hourly)
    report["adf"] = {"statistic": adf.statistic, "pvalue": adf.pvalue}

    # --- Tabular models ---
    lr = train_linear(X_tr, y_tr)
    gbr = train_gbr(X_tr, y_tr, random_state=SEED)
    report["models"]["LinearRegression"] = {
        **eval_model(lr, X_te, y_te),
        "eval_slice": "full_chronological_test_15pct",
    }
    report["models"]["GradientBoosting"] = {
        **eval_model(gbr, X_te, y_te),
        "eval_slice": "full_chronological_test_15pct",
    }
    # Aligned 168h slice at end of tabular test
    n168 = min(HORIZON, len(X_te))
    report["models"]["LinearRegression_last168"] = {
        **eval_model(lr, X_te.iloc[-n168:], y_te.iloc[-n168:]),
        "eval_slice": f"last_{n168}_of_tabular_test",
    }
    report["models"]["GradientBoosting_last168"] = {
        **eval_model(gbr, X_te.iloc[-n168:], y_te.iloc[-n168:]),
        "eval_slice": f"last_{n168}_of_tabular_test",
    }

    if args.full:
        # Series path: walk-forward **one-step-ahead** over the final HORIZON hours.
        # Using true past windows only (append after each step) — no y_test→prediction fabrication.
        y_all = hourly.astype(float)
        start_idx = len(y_all) - HORIZON
        y_train_h = y_all.iloc[:start_idx]
        y_test_h = y_all.iloc[start_idx:]

        ar_fit = fit_arima(y_train_h, order=(4, 1, 2))
        ar_pred = arima_onestep_forecast(ar_fit, y_test_h)
        report["models"]["ARIMA(4,1,2)"] = {
            **eval_model_series(y_test_h, ar_pred),
            "eval_slice": "last_168h_onestep_walkforward",
            "train_points": int(len(y_train_h)),
        }

        lstm_model, scaler = train_lstm_on_series(
            y_train_h, lookback=24, epochs=args.lstm_epochs, seed=SEED, verbose=0
        )
        lstm_pred = lstm_onestep_forecast(
            lstm_model, scaler, y_all, start_idx=start_idx, steps=HORIZON, lookback=24
        )
        report["models"]["LSTM"] = {
            **eval_model_series(y_test_h, lstm_pred),
            "eval_slice": "last_168h_onestep_walkforward",
            "epochs": args.lstm_epochs,
            "seed": SEED,
        }

        hybrid = train_hybrid(
            y_train_h,
            order=(4, 1, 2),
            lookback=24,
            epochs=args.lstm_epochs,
            seed=SEED,
            verbose=0,
        )
        hy_pred = hybrid_onestep_forecast(
            hybrid, y_all, start_idx=start_idx, steps=HORIZON
        )
        report["models"]["Hybrid_ARIMA_LSTM_residual"] = {
            **eval_model_series(y_test_h, hy_pred),
            "eval_slice": "last_168h_onestep_walkforward",
            "architecture": (
                "ARIMA(4,1,2) on train volume + LSTM on train ARIMA residuals; "
                "one-step walk-forward: yhat=ARIMA_forecast + LSTM_residual "
                "(updates with observations after each step; no y_t in the prediction of y_t)"
            ),
            "epochs": args.lstm_epochs,
            "seed": SEED,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")
    return 0


def eval_model_series(y_true, y_pred):
    from metro_traffic.evaluate import regression_metrics

    return regression_metrics(y_true, y_pred)


if __name__ == "__main__":
    raise SystemExit(main())
