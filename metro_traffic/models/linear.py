"""Linear Regression on engineered tabular features."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metro_traffic.evaluate import regression_metrics

NUM_FEATURES = ["temp", "rain_1h", "snow_1h", "clouds_all", "hour", "month"]


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("num", StandardScaler(), NUM_FEATURES)],
        remainder="passthrough",
    )


def train_linear(X_train, y_train) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("preprocessor", _preprocessor()),
            ("regressor", LinearRegression()),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def eval_model(model, X_test, y_test) -> dict[str, float]:
    return regression_metrics(y_test, model.predict(X_test))
