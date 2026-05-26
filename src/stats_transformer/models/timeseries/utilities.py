import numpy as np
import pandas as pd


class TimeSeriesFeatureBuilder:
    """Build lag, lead, and forecast-horizon columns for time-series data."""

    def __init__(self, date_column="date", entity_column=None):
        self.date_column = date_column
        self.entity_column = entity_column

    def add_lags(self, df, columns, lags, suffix="lag"):
        """Return a copy of df with lagged columns added."""
        return self._add_shifted_columns(df, columns, lags, suffix=suffix, direction=1)

    def add_leads(self, df, columns, leads, suffix="lead"):
        """Return a copy of df with lead columns added."""
        return self._add_shifted_columns(df, columns, leads, suffix=suffix, direction=-1)

    def add_horizons(self, df, target, horizons, suffix="h"):
        """Return a copy of df with forward target horizons added."""
        return self.add_leads(df, [target], horizons, suffix=suffix)

    def _add_shifted_columns(self, df, columns, periods, suffix, direction):
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(periods, int):
            periods = range(1, periods + 1)

        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        work = self._sort(df)
        grouped = self._group(work)
        for col in columns:
            for period in periods:
                if period < 1:
                    raise ValueError("Lag, lead, and horizon periods must be positive")
                new_col = f"{col}_{suffix}{period}"
                work[new_col] = grouped[col].shift(direction * period)
        return work

    def _sort(self, df):
        sort_columns = []
        if self.entity_column:
            sort_columns.append(self.entity_column)
        if self.date_column:
            sort_columns.append(self.date_column)
        if not sort_columns:
            return df.copy()
        missing = [col for col in sort_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return df.sort_values(sort_columns).copy()

    def _group(self, df):
        if self.entity_column:
            return df.groupby(self.entity_column, sort=False)
        return _SingleGroup(df)


class _SingleGroup:
    def __init__(self, df):
        self.df = df

    def __getitem__(self, column):
        return self.df[column]


class ForecastEvaluator:
    """Evaluate forecasts with common scalar error metrics."""

    def __init__(self, actual_column, forecast_column):
        self.actual_column = actual_column
        self.forecast_column = forecast_column

    def evaluate(self, df):
        missing = [col for col in [self.actual_column, self.forecast_column] if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        clean = df[[self.actual_column, self.forecast_column]].replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            raise ValueError("No valid forecast observations after dropping missing values")

        actual = clean[self.actual_column].to_numpy(dtype=float)
        forecast = clean[self.forecast_column].to_numpy(dtype=float)
        errors = forecast - actual
        nonzero = actual != 0

        metrics = {
            "num_observations": int(len(clean)),
            "mae": float(np.mean(np.abs(errors))),
            "mse": float(np.mean(errors ** 2)),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mean_error": float(np.mean(errors)),
        }
        metrics["mape"] = (
            float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100)
            if np.any(nonzero)
            else np.nan
        )
        return metrics
