import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from stats_transformer.models.base import ModelBase


class ARIMAModel(ModelBase):
    """Fit univariate ARIMA models and expose tidy forecast metadata."""

    _is_multivariate = True

    def __init__(self, target=None, order=(1, 0, 0), date_column="date", trend=None, **kwargs):
        super().__init__(**kwargs)
        model_params = self.params.get("model", {})
        self.target = target or model_params.get("target_variable") or getattr(self, "target", None)
        if not self.target and model_params.get("target_variables"):
            self.target = model_params.get("target_variables")[0]
        self.target_variables = [self.target] if self.target else []
        self.independent_variables = []
        self.order = tuple(model_params.get("order", order))
        self.date_column = date_column or model_params.get("date_column")
        self.time_column = self.date_column
        self.trend = trend or model_params.get("trend")
        self.kwargs = kwargs
        self.model_spec = None
        self.model = None

    def build_model(self):
        if self.df_clean is None:
            raise ValueError("No cleaned data available")
        work = self.df_clean.copy()
        if self.date_column and self.date_column in work.columns:
            work = work.sort_values(self.date_column)
        self.model_spec = ARIMA(
            work[self.target],
            order=self.order,
            trend=self.trend,
        )
        self.model = self.model_spec.fit()
        return self.model

    def fit(self, df, drop_na=True):
        """Fit the ARIMA model and return scalar fit metrics."""
        self.load_data(df)
        self.build_model()
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
