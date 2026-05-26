import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss


class StationarityDiagnostics:
    """Run stationarity diagnostics for one or more time-series columns."""

    def __init__(self, date_column="date", regression="c", autolag="AIC", nlags="auto"):
        self.date_column = date_column
        self.regression = regression
        self.autolag = autolag
        self.nlags = nlags
        self.results_ = None

    def fit(self, df, columns):
        if isinstance(columns, str):
            columns = [columns]
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        work = df.sort_values(self.date_column).copy() if self.date_column in df.columns else df.copy()
        rows = []
        for column in columns:
            series = work[column].replace([np.inf, -np.inf], np.nan).dropna()
            if len(series) < 8:
                raise ValueError(f"Column '{column}' needs at least 8 observations for stationarity tests")
            rows.append(self._adf_row(column, series))
            rows.append(self._kpss_row(column, series))

        self.results_ = pd.DataFrame(rows)
        return self.results_

    def get_model_metadata(self):
        if self.results_ is None:
            raise ValueError("Diagnostics must be fitted before metadata is available")
        return {
            "num_tests": int(len(self.results_)),
            "columns": sorted(self.results_["column"].unique().tolist()),
            "tests": sorted(self.results_["test"].unique().tolist()),
        }

    def summary_df(self):
        if self.results_ is None:
            raise ValueError("Diagnostics must be fitted before results are available")
        return self.results_.copy()

    def _adf_row(self, column, series):
        statistic, p_value, used_lag, nobs, critical_values, _icbest = adfuller(
            series,
            regression=self.regression,
            autolag=self.autolag,
        )
        return {
            "column": column,
            "test": "adf",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "used_lag": int(used_lag),
            "num_observations": int(nobs),
            "stationary_at_5pct": bool(p_value < 0.05),
            "critical_values": critical_values,
        }

    def _kpss_row(self, column, series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            statistic, p_value, used_lag, critical_values = kpss(
                series,
                regression=self.regression,
                nlags=self.nlags,
            )
        return {
            "column": column,
            "test": "kpss",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "used_lag": int(used_lag),
            "num_observations": int(len(series)),
            "stationary_at_5pct": bool(p_value >= 0.05),
            "critical_values": critical_values,
        }
