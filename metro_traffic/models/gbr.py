"""Gradient Boosting Regressor on engineered tabular features."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metro_traffic.evaluate import regression_metrics

NUM_FEATURES = ["temp", "rain_1h", "snow_1h", "clouds_all", "hour", "month"]


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("num", StandardScaler(), NUM_FEATURES)],
        remainder="passthrough",
    )


def train_gbr(X_train, y_train, *, n_estimators: int = 100, random_state: int = 42) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("preprocessor", _preprocessor()),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=n_estimators, random_state=random_state
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def eval_model(model, X_test, y_test) -> dict[str, float]:
    return regression_metrics(y_test, model.predict(X_test))
