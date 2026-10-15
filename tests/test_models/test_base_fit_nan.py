import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from stats_transformer.models.base import ModelBase


class _MinimalOLS(ModelBase):
    """Concrete ModelBase that builds X/y inside build_model, like most models."""

    def build_model(self) -> sm.OLS:
        assert self.df_clean is not None
        self.X = sm.add_constant(self.df_clean[["x1", "x2"]])
        self.y = self.df_clean["target"]
        self.model = sm.OLS(self.y, self.X).fit()
        return self.model

    def get_summary(self) -> sm.OLS:
        return self.model.summary()

    def get_model_metrics(self) -> dict:
        return {"num_observations": int(self.model.nobs)}


def _make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "target": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x1": [2.0, 4.0, 6.0, 8.0, 10.0],
        "x2": [1.0, 1.0, 2.0, 2.0, 3.0]
    })


def test_fit_clean_data_unchanged_results():
    model = _MinimalOLS(target="target", independent_variables=["x1", "x2"])
    metrics = model.fit(_make_df())
    assert metrics["num_observations"] == 5


def test_fit_inf_regressor_raises_valueerror():
    # inf survives load_data's NaN dropna, so it must surface from build_model
    df = _make_df()
    df.loc[2, "x1"] = np.inf
    model = _MinimalOLS(target="target", independent_variables=["x1", "x2"])
    with pytest.raises(ValueError, match=r"1 rows contain NaN/inf in \['x1'\]"):
        model.fit(df)


class _LagOLS(_MinimalOLS):
    """Adds a lagged regressor, whose first row is NaN after build_model."""

    def build_model(self) -> sm.OLS:
        assert self.df_clean is not None
        lagged = self.df_clean.assign(x1_lag1=self.df_clean["x1"].shift(1))
        self.X = sm.add_constant(lagged[["x1", "x2", "x1_lag1"]])
        self.y = lagged["target"]
        self.model = sm.OLS(self.y, self.X).fit()
        return self.model


def test_fit_nan_from_features_raises_valueerror():
    # NaN created inside build_model (e.g. a lag) must raise, not trigger
    # a silent row drop and refit
    model = _LagOLS(target="target", independent_variables=["x1", "x2"])
    with pytest.raises(ValueError, match=r"1 rows contain NaN/inf in \['x1_lag1'\]"):
        model.fit(_make_df())