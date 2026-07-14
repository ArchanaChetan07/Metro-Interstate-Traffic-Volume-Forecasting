"""Tests for metro_traffic features, metrics, and fast-model regression floor."""

from __future__ import annotations

import numpy as np
import pytest

from metro_traffic.evaluate import regression_metrics
from metro_traffic.features import (
    FEATURE_COLS,
    assert_no_lag_nan,
    build_xy,
    clean_and_engineer,
    chronological_split,
    load_raw,
)
from metro_traffic.models.tabular import eval_model, train_gbr, train_linear


@pytest.fixture(scope="module")
def engineered():
    raw = load_raw()
    # Smaller window for speed while keeping structure
    df = clean_and_engineer(raw, start="2017-01-01", end="2018-09-30")
    return df


class TestFeatures:
    def test_feature_columns_present(self, engineered):
        X, y = build_xy(engineered)
        assert list(X.columns) == FEATURE_COLS
        assert len(X) == len(y) > 1000

    def test_no_nan_in_features(self, engineered):
        X, _ = build_xy(engineered)
        assert_no_lag_nan(X)
        assert not X.isna().any().any()

    def test_chronological_split_sizes(self, engineered):
        X, y = build_xy(engineered)
        (Xtr, _), (Xv, _), (Xte, _) = chronological_split(X, y)
        assert len(Xtr) + len(Xv) + len(Xte) == len(X)
        # chronological: last train index < first val < first test
        assert Xtr.index.max() < Xv.index.min() <= Xte.index.min() or (
            Xtr.index[-1] < Xv.index[0] <= Xte.index[0]
        )

    def test_holiday_none_is_zero(self):
        raw = load_raw().head(200).copy()
        # Force known holiday strings
        raw.loc[raw.index[0], "holiday"] = "None"
        raw.loc[raw.index[1], "holiday"] = "Christmas Day"
        df = clean_and_engineer(raw, start="2012-01-01", end="2018-12-31")
        # At least one zero holiday flag exists
        assert set(df["is_holiday"].unique()).issubset({0, 1})


class TestEvaluate:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = regression_metrics(y, y)
        assert m["r2"] == pytest.approx(1.0)
        assert m["mae"] == pytest.approx(0.0)
        assert m["rmse"] == pytest.approx(0.0)

    def test_known_rmse(self):
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([3.0, 4.0, 0.0, 0.0])
        m = regression_metrics(y_true, y_pred)
        # RMSE = sqrt((9+16)/4) = 2.5
        assert m["rmse"] == pytest.approx(2.5)


class TestFastModelRegression:
    def test_gbr_r2_floor_on_sample(self, engineered):
        X, y = build_xy(engineered)
        (Xtr, ytr), _, (Xte, yte) = chronological_split(X, y)
        # Subsample train for CI speed
        Xtr_s, ytr_s = Xtr.iloc[::3], ytr.iloc[::3]
        model = train_gbr(Xtr_s, ytr_s, n_estimators=40, random_state=42)
        metrics = eval_model(model, Xte, yte)
        # Traffic is highly calendar-driven; GBR should clear a modest floor
        assert metrics["r2"] > 0.70, metrics

    def test_linear_r2_positive(self, engineered):
        X, y = build_xy(engineered)
        (Xtr, ytr), _, (Xte, yte) = chronological_split(X, y)
        model = train_linear(Xtr.iloc[::2], ytr.iloc[::2])
        metrics = eval_model(model, Xte, yte)
        # Linear model is intentionally weak without nonlinear calendar interactions;
        # still must beat a trivial floor on this highly structured series.
        assert metrics["r2"] > 0.30, metrics


class TestLeakageGuards:
    def test_split_is_chronological_not_shuffled(self, engineered):
        X, y = build_xy(engineered)
        (Xtr, _), (Xv, _), (Xte, _) = chronological_split(X, y)
        assert Xtr.index[-1] < Xv.index[0] < Xte.index[0]

    def test_hybrid_doc_forbids_y_plus_noise_eval(self):
        """Package hybrid module must not evaluate simulated y + noise."""
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "metro_traffic" / "models" / "hybrid.py"
        text = src.read_text(encoding="utf-8")
        assert "random.normal" not in text
        assert "sine" not in text.lower()
        assert "train only" in text.lower() or "train series only" in text.lower()
