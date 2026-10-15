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
    with pytest.raises(ValueError, match=r"1 rows contain NaN/inf in \['x1'\].*original error: MissingDataError: exog contains inf or nans") as excinfo:
        model.fit(df)
    assert isinstance(excinfo.value.__cause__, Exception)  # chained to the original error


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


def test_original_error_propagates_when_check_cannot_run():
    # build_model fails without setting X/y: the RuntimeError must propagate
    # unchanged, untouched by the NaN/inf check
    class _NoXY(_MinimalOLS):
        def build_model(self):
            raise RuntimeError("boom")

    model = _NoXY(target="target", independent_variables=["x1", "x2"])
    with pytest.raises(RuntimeError, match="^boom$") as excinfo:
        model.fit(_make_df())
    assert not isinstance(excinfo.value, ValueError)


def test_numpy_x_with_nan_raises_valueerror():
    # X kept as a 2-D numpy array holding NaN: the ValueError must name the
    # NaN column instead of hiding the real error
    class _NumpyNaNX(_MinimalOLS):
        def build_model(self):
            self.X = np.array([[1.0], [2.0], [np.nan]])
            self.y = pd.Series([1.0, 2.0, 3.0], name="target")
            raise RuntimeError("exog contains inf or nans")

    model = _NumpyNaNX(target="target", independent_variables=["x1", "x2"])
    with pytest.raises(ValueError, match=r"1 rows contain NaN/inf in \['X0'\]"):
        model.fit(_make_df())


def test_numpy_x_clean_propagates_original_error():
    # Clean 2-D numpy X: no NaN/inf ValueError, the original error propagates
    class _NumpyCleanX(_MinimalOLS):
        def build_model(self):
            self.X = np.arange(6, dtype=float).reshape(3, 2)
            self.y = pd.Series([1.0, 2.0, 3.0], name="target")
            raise RuntimeError("exog contains inf or nans")

    model = _NumpyCleanX(target="target", independent_variables=["x1", "x2"])
    with pytest.raises(RuntimeError, match="^exog contains inf or nans$"):
        model.fit(_make_df())


class _DropNaOLS(_MinimalOLS):
    """Like statsmodels missing='drop': fits after dropping NaN rows itself,
    but leaves the NaN row in self.X."""

    def build_model(self) -> sm.OLS:
        assert self.df_clean is not None
        lagged = self.df_clean.assign(x1_lag1=self.df_clean["x1"].shift(1))
        self.X = sm.add_constant(lagged[["x1", "x1_lag1"]])
        self.y = lagged["target"]
        mask = self.X["x1_lag1"].notna()
        self.model = sm.OLS(self.y[mask], self.X[mask]).fit()
        return self.model


def test_refit_succeeds_when_stale_x_holds_nan():
    # A first fit may legitimately leave NaN in self.X; the second fit must
    # not trip over the stale X/y and must succeed unchanged
    model = _DropNaOLS(target="target", independent_variables=["x1"])
    first = model.fit(_make_df())
    assert first["num_observations"] == 4
    assert model.X["x1_lag1"].isna().sum() == 1
    second = model.fit(_make_df())
    assert second["num_observations"] == 4