import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


class TestTrafficDataProcessing:

    def test_datetime_parsing(self):
        df = pd.DataFrame({"date_time": ["2012-10-02 09:00:00", "2012-10-02 10:00:00"]})
        df["date_time"] = pd.to_datetime(df["date_time"])
        assert np.issubdtype(df["date_time"].dtype, np.datetime64)

    def test_hour_feature_extraction(self):
        df = pd.DataFrame({"date_time": pd.to_datetime(["2012-10-02 09:00:00", "2012-10-02 14:00:00"])})
        df["hour"] = df["date_time"].dt.hour
        assert df["hour"].tolist() == [9, 14]

    def test_outlier_volume_removed(self):
        df = pd.DataFrame({"traffic_volume": [1000, 2000, 99999, 1500]})
        df_clean = df[df["traffic_volume"] < 10000]
        assert 99999 not in df_clean["traffic_volume"].values

    def test_categorical_encoding(self):
        df = pd.DataFrame({"weather": ["Clear", "Rain", "Clear", "Snow"]})
        le = LabelEncoder()
        df["weather_enc"] = le.fit_transform(df["weather"])
        assert df["weather_enc"].dtype in [np.int32, np.int64, int]

    def test_no_nulls_in_key_columns(self):
        df = pd.DataFrame({"traffic_volume": [1000, 2000, None], "temp": [288.0, 290.0, 285.0]})
        df_clean = df.dropna()
        assert df_clean.isnull().sum().sum() == 0


class TestTrafficForecasting:

    def test_model_r2_positive(self):
        np.random.seed(42)
        X = np.random.rand(200, 6)
        y = X[:, 0] * 5000 + X[:, 1] * 2000 + np.random.randn(200) * 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        assert r2 > 0, f"R2 score {r2:.3f} should be positive"

    def test_rmse_reasonable(self):
        np.random.seed(0)
        X = np.random.rand(100, 4)
        y = X[:, 0] * 3000
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        assert rmse < 10000
