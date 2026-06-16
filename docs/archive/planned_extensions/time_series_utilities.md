# Time-Series Utilities

This extension implements the first reusable utilities from the planned time-series roadmap.

## Implemented

### `TimeSeriesFeatureBuilder`

Builds lag, lead, and forecast-horizon columns while respecting optional panel entities.

```python
from stats_transformer.models.timeseries import TimeSeriesFeatureBuilder

builder = TimeSeriesFeatureBuilder(date_column="date", entity_column="country")
df = builder.add_lags(df, columns=["gdp_growth", "inflation"], lags=4)
df = builder.add_horizons(df, target="gdp_growth", horizons=[1, 4, 8])
```

### `ForecastEvaluator`

Computes scalar forecast error metrics: MAE, MSE, RMSE, mean error, MAPE, and number of observations.

```python
from stats_transformer.models.timeseries import ForecastEvaluator

metrics = ForecastEvaluator("actual", "forecast").evaluate(df)
```

### `StationarityDiagnostics`

Runs ADF and KPSS tests for one or more columns and returns tidy diagnostic rows.

```python
from stats_transformer.models.timeseries import StationarityDiagnostics

diagnostics = StationarityDiagnostics(date_column="date")
results = diagnostics.fit(df, columns=["gdp_growth", "inflation"])
metadata = diagnostics.get_model_metadata()
```

## Still Planned

- Config-driven execution through `Pipeline`.
- Entity-wise stationarity diagnostics for large panels.
- Forecast comparison tests such as Diebold-Mariano.
- Visualization helpers for diagnostic p-values and rolling forecast errors.
