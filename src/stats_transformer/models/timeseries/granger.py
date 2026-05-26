import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


class GrangerCausalityTester:
    """Run Granger causality tests for a single time series or panel."""

    def __init__(
        self,
        entity_column=None,
        date_column="date",
        max_lag=4,
        standardize=False,
        test="ssr_ftest",
    ):
        if max_lag < 1:
            raise ValueError("max_lag must be at least 1")

        self.entity_column = entity_column
        self.date_column = date_column
        self.max_lag = max_lag
        self.standardize = standardize
        self.test = test
        self.results_ = None

    def fit(self, df, caused, causing, drop_na=True):
        """Run Granger tests and return tidy test results."""
        self._validate_columns(df, caused, causing)
        results = []

        if self.entity_column:
            groups = df.groupby(self.entity_column, sort=False)
            for entity, group in groups:
                results.extend(self._fit_group(group, caused, causing, entity, drop_na))
        else:
            results.extend(self._fit_group(df, caused, causing, None, drop_na))

        self.results_ = pd.DataFrame(results)
        return self.results_

    def get_model_metadata(self):
        """Return a compact summary of the fitted Granger tests."""
        if self.results_ is None:
            raise ValueError("Tester must be fitted before metadata is available")
        if self.results_.empty:
            return {
                "max_lag": self.max_lag,
                "test": self.test,
                "num_tests": 0,
                "best_lag": None,
                "min_p_value": None,
            }

        test_results = self.results_[self.results_["test"] == self.test]
        if test_results.empty:
            test_results = self.results_
        best_idx = test_results["p_value"].idxmin()
        best = test_results.loc[best_idx]

        metadata = {
            "max_lag": self.max_lag,
            "test": self.test,
            "num_tests": int(len(self.results_)),
            "best_lag": int(best["lag"]),
            "min_p_value": float(best["p_value"]),
        }
        if "entity" in best and pd.notna(best["entity"]):
            metadata["best_entity"] = best["entity"]
        return metadata

    def summary_df(self):
        """Return fitted results as a DataFrame."""
        if self.results_ is None:
            raise ValueError("Tester must be fitted before results are available")
        return self.results_.copy()

    def _validate_columns(self, df, caused, causing):
        required = [caused, causing]
        if self.entity_column:
            required.append(self.entity_column)
        if self.date_column:
            required.append(self.date_column)

        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def _fit_group(self, group, caused, causing, entity, drop_na):
        work = group.copy()
        if self.date_column:
            work = work.sort_values(self.date_column)
        work = work[[caused, causing]].replace([np.inf, -np.inf], np.nan)
        if drop_na:
            work = work.dropna()

        if self.standardize:
            work = self._standardize(work)

        min_observations = self.max_lag + 3
        if len(work) < min_observations:
            raise ValueError(
                f"At least {min_observations} observations are required for max_lag={self.max_lag}"
            )

        raw_results = grangercausalitytests(work[[caused, causing]], maxlag=self.max_lag)
        rows = []
        for lag, lag_result in raw_results.items():
            test_stats = lag_result[0]
            for test_name, values in test_stats.items():
                statistic = values[0]
                p_value = values[1]
                df_denom = values[2] if len(values) > 2 else np.nan
                df_num = values[3] if len(values) > 3 else np.nan
                row = {
                    "caused": caused,
                    "causing": causing,
                    "lag": int(lag),
                    "test": test_name,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "df_denom": float(df_denom),
                    "df_num": float(df_num),
                    "num_observations": int(len(work)),
                }
                if self.entity_column:
                    row["entity"] = entity
                rows.append(row)
        return rows

    @staticmethod
    def _standardize(df):
        means = df.mean()
        stds = df.std(ddof=0).replace(0, 1)
        return (df - means) / stds
