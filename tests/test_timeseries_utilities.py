import numpy as np
import pandas as pd
import pytest

from stats_transformer.models.timeseries import (
    ForecastEvaluator,
    StationarityDiagnostics,
    TimeSeriesFeatureBuilder,
)


def test_time_series_feature_builder_adds_panel_lags_and_horizons():
    df = pd.DataFrame(
        {
            "country": ["USA", "USA", "USA", "CAN", "CAN", "CAN"],
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                ]
            ),
            "y": [1, 2, 3, 10, 20, 30],
            "x": [4, 5, 6, 40, 50, 60],
        }
    )

    builder = TimeSeriesFeatureBuilder(date_column="date", entity_column="country")
    with_lags = builder.add_lags(df, columns=["y", "x"], lags=2)
    with_horizons = builder.add_horizons(with_lags, target="y", horizons=[1])

    usa = with_horizons[with_horizons["country"] == "USA"].reset_index(drop=True)
    can = with_horizons[with_horizons["country"] == "CAN"].reset_index(drop=True)

    assert pd.isna(usa.loc[0, "y_lag1"])
    assert usa.loc[1, "y_lag1"] == 1
    assert usa.loc[2, "x_lag2"] == 4
    assert usa.loc[0, "y_h1"] == 2
    assert can.loc[1, "y_lag1"] == 10
    assert can.loc[2, "x_lag2"] == 40


def test_forecast_evaluator_returns_error_metrics():
    df = pd.DataFrame(
        {
            "actual": [100, 120, 80, 0],
            "forecast": [110, 115, 90, 2],
        }
    )

    evaluator = ForecastEvaluator(actual_column="actual", forecast_column="forecast")
    metrics = evaluator.evaluate(df)

    assert metrics["num_observations"] == 4
    assert metrics["mae"] == pytest.approx(6.75)
    assert metrics["rmse"] == pytest.approx(np.sqrt((10**2 + (-5) ** 2 + 10**2 + 2**2) / 4))
    assert metrics["mape"] == pytest.approx(((0.10 + 5 / 120 + 0.125) / 3) * 100)


def test_stationarity_diagnostics_returns_adf_and_kpss_results():
    np.random.seed(42)
    n = 80
    stationary = np.random.normal(size=n)
    trend = np.arange(n) + np.random.normal(scale=0.5, size=n)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=n, freq="ME"),
            "stationary": stationary,
            "trend": trend,
        }
    )

    diagnostics = StationarityDiagnostics(date_column="date")
    results = diagnostics.fit(df, columns=["stationary", "trend"])
    metadata = diagnostics.get_model_metadata()

    assert set(results["test"]) == {"adf", "kpss"}
    assert set(results["column"]) == {"stationary", "trend"}
    assert metadata["num_tests"] == 4
    assert metadata["tests"] == ["adf", "kpss"]
