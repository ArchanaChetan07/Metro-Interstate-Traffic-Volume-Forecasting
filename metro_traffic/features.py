"""Feature engineering for UCI Metro Interstate traffic volume."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

FEATURE_COLS = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "is_holiday",
    "hour",
    "month",
    "is_weekend",
    "rush_hour",
]
TARGET_COL = "traffic_volume"


def default_data_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    gz = root / "data" / "Metro_Interstate_Traffic_Volume.csv.gz"
    csv = root / "data" / "Metro_Interstate_Traffic_Volume.csv"
    if gz.exists():
        return gz
    if csv.exists():
        return csv
    raise FileNotFoundError("Place UCI Metro_Interstate_Traffic_Volume.csv(.gz) under data/")


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else default_data_path()
    df = pd.read_csv(path, compression="gzip" if path.suffix == ".gz" else None)
    return df


def clean_and_engineer(
    df: pd.DataFrame,
    *,
    start: str = "2015-06-11",
    end: str = "2018-09-30",
) -> pd.DataFrame:
    """Kelvin→°F, outlier caps, holiday/calendar/rush features; filter modeling window."""
    out = df.copy()
    out["date_time"] = pd.to_datetime(out["date_time"])
    # Kelvin to Fahrenheit
    out["temp"] = (out["temp"] * (9 / 5)) - 459.67
    # Caps matching Quarto EDA notes
    out.loc[out["temp"] < -100, "temp"] = 20.6
    out.loc[out["rain_1h"] > 2500, "rain_1h"] = 21.4

    # Holiday: UCI uses string 'None' when not a holiday
    hol = out["holiday"].astype(str)
    out["is_holiday"] = (~hol.isin(["None", "nan", ""])).astype(int)

    out["hour"] = out["date_time"].dt.hour
    out["month"] = out["date_time"].dt.month
    out["day_name"] = out["date_time"].dt.day_name()
    out["is_weekend"] = out["day_name"].isin(["Saturday", "Sunday"]).astype(int)
    out["rush_hour"] = (
        ((out["hour"] >= 6) & (out["hour"] <= 9))
        | ((out["hour"] >= 16) & (out["hour"] <= 19))
    ).astype(int)

    out = out[(out["date_time"] >= start) & (out["date_time"] <= end)].copy()
    out = out.sort_values("date_time").reset_index(drop=True)
    return out


def make_hourly_series(df: pd.DataFrame) -> pd.Series:
    """Resample to hourly mean volume (ffill gaps) for ADF / ARIMA / LSTM."""
    ts = (
        df[["date_time", TARGET_COL]]
        .set_index("date_time")
        .resample("h")
        .mean()
        .ffill()
    )
    return ts[TARGET_COL]


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
):
    """70/15/15 chronological split (no shuffle — leakage-safe)."""
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train : n_train + n_val], y.iloc[n_train : n_train + n_val]
    X_test, y_test = X.iloc[n_train + n_val :], y.iloc[n_train + n_val :]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(float).copy()
    return X, y


def assert_no_lag_nan(X: pd.DataFrame, cols: Iterable[str] | None = None) -> None:
    cols = list(cols) if cols is not None else list(X.columns)
    if X[cols].isna().any().any():
        raise ValueError("NaNs present in feature matrix")
