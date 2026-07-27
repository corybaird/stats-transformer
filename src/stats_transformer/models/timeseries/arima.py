import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    """Fit univariate ARIMA models and expose tidy forecast metadata."""

    def __init__(self, target, order=(1, 0, 0), date_column="date", trend=None, **kwargs):
        self.target = target
        self.order = order
        self.date_column = date_column
        self.trend = trend
        self.kwargs = kwargs
        self.df_clean = None
        self.model_spec = None
        self.model = None

    def fit(self, df, drop_na=True):
        """Fit the ARIMA model and return scalar fit metrics."""
        self._validate_columns(df)
        work = df.copy()
        if self.date_column and self.date_column in work.columns:
            work = work.sort_values(self.date_column)
        work[self.target] = work[self.target].replace([np.inf, -np.inf], np.nan)
        if drop_na:
            work = work.dropna(subset=[self.target])
        if work.empty:
            raise ValueError("DataFrame is empty after dropping missing target values")

        self.df_clean = work
        self.model_spec = ARIMA(
            work[self.target],
            order=self.order,
            trend=self.trend,
            **self.kwargs,
        )
        self.model = self.model_spec.fit()
        return self.get_model_metrics()

    def forecast(self, steps=1, alpha=0.05):
        """Return forecast means and confidence intervals as a DataFrame."""
        self._require_fit()
        if steps < 1:
            raise ValueError("steps must be at least 1")

        forecast_result = self.model.get_forecast(steps=steps)
        frame = forecast_result.summary_frame(alpha=alpha).reset_index(drop=True)
        mean_col = "mean" if "mean" in frame.columns else frame.columns[0]
        lower_col = self._find_column(frame, "lower")
        upper_col = self._find_column(frame, "upper")

        output = pd.DataFrame(
            {
                "step": range(1, steps + 1),
                "forecast": frame[mean_col].astype(float).to_numpy(),
            }
        )
        if lower_col:
            output["lower_ci"] = frame[lower_col].astype(float).to_numpy()
        if upper_col:
            output["upper_ci"] = frame[upper_col].astype(float).to_numpy()
        return output

    def get_summary(self):
        """Return the statsmodels text summary."""
        self._require_fit()
        return str(self.model.summary())

    def get_model_metrics(self):
        """Return scalar model fit metrics."""
        self._require_fit()
        return {
            "aic": float(self.model.aic),
            "bic": float(self.model.bic),
            "hqic": float(self.model.hqic),
            "llf": float(self.model.llf),
            "num_observations": int(self.model.nobs),
            "order": self.order,
        }

    def get_model_metadata(self):
        """Return package-friendly metadata for reporting and downstream tools."""
        metrics = self.get_model_metrics()
        params = {}
        conf_int = self.model.conf_int()
        for name, value in self.model.params.items():
            params[name] = {
                "value": float(value),
                "std_err": float(self.model.bse[name]),
                "z_value": float(self.model.zvalues[name]),
                "p_value": float(self.model.pvalues[name]),
                "ci_lower": float(conf_int.loc[name, 0]),
                "ci_upper": float(conf_int.loc[name, 1]),
            }
        return {
            "model_type": "ARIMA",
            "target": self.target,
            "order": self.order,
            "metrics": metrics,
            "parameters": params,
        }

    def _validate_columns(self, df):
        if self.target not in df.columns:
            raise ValueError(f"Missing columns: ['{self.target}']")
        if self.date_column and self.date_column not in df.columns:
            raise ValueError(f"Missing columns: ['{self.date_column}']")

    def _require_fit(self):
        if self.model is None:
            raise ValueError("Model must be fitted before results are available")

    @staticmethod
    def _find_column(frame, pattern):
        matches = [col for col in frame.columns if pattern in col.lower()]
        return matches[0] if matches else None
