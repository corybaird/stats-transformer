import numpy as np
import pandas as pd
import pytest
from linearmodels.iv import IV2SLS
from stats_transformer.models.timeseries.reduced_form.local_projections_iv import LocalProjectionsIVModel


def create_synthetic_data(n=200, seed=42):
    np.random.seed(seed)
    y1 = np.cumsum(np.random.normal(size=n))
    y2 = np.cumsum(np.random.normal(size=n)) + 0.5 * y1
    z = np.random.normal(size=n) + 0.3 * y1
    df = pd.DataFrame({"y1": y1, "y2": y2, "z": z, "date": pd.date_range("2000-01-01", periods=n, freq="QE")})
    return df

def create_multiplier_dgp(n=4000, seed=7, beta=0.8):
    np.random.seed(seed)
    z = np.random.normal(size=n)
    x = 2.0 * z + np.random.normal(size=n)
    y = beta * x + 0.5 * np.random.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z})

def test_multiplier_recovers_known_dgp_multiplier():
    beta = 0.8
    df = create_multiplier_dgp(beta=beta)
    model = LocalProjectionsIVModel(target_variable="y", shock_variable="x", instrument_variable="z", horizons=5, multiplier=True)
    model.fit(df)
    # Integral of a static response: every horizon multiplier recovers beta
    assert np.allclose(model.irf_coefficients, beta, atol=0.1)
def test_cumulative_and_multiplier_are_mutually_exclusive():
    df = create_synthetic_data()
    with pytest.raises(ValueError, match="mutually exclusive"):
        LocalProjectionsIVModel(target_variable="y1", shock_variable="y2", instrument_variable="z", horizons=2, cumulative=True, multiplier=True).fit(df)

def test_hc3_is_rejected_for_iv():
    df = create_synthetic_data()
    with pytest.raises(ValueError, match="HC3 is not supported"):
        LocalProjectionsIVModel(target_variable="y1", shock_variable="y2", instrument_variable="z", horizons=2, cov_type="HC3").fit(df)

def test_hac_std_errors_differ_from_nonrobust_and_are_positive():
    df = create_synthetic_data()
    base = LocalProjectionsIVModel(target_variable="y1", shock_variable="y2", instrument_variable="z", horizons=3, date_column="date")
    base.fit(df)
    hac = LocalProjectionsIVModel(target_variable="y1", shock_variable="y2", instrument_variable="z", horizons=3, date_column="date", cov_type="HAC")
    hac.fit(df)
    se_hac = np.asarray(hac.irf_std_errors)
    se_base = np.asarray(base.irf_std_errors)
    assert np.all(se_hac > 0)
    assert np.all(se_base > 0)
    assert not np.allclose(se_hac, se_base)

def test_lags_add_expected_columns():
    df = create_synthetic_data()
    model = LocalProjectionsIVModel(target_variable="y1", shock_variable="y2", instrument_variable="z", horizons=2, date_column="date", lags=2)
    model.fit(df)
    cols = set(model._build_exog(model.df_clean).columns)
    for lag in (1, 2):
        for col in ("y1", "y2", "z"):
            assert f"{col}_lag{lag}" in cols

def test_cumulative_matches_manual_two_sls():
    df = create_synthetic_data()
    model = LocalProjectionsIVModel(target_variable="y1", shock_variable="y2", instrument_variable="z", horizons=3, date_column="date", cumulative=True)
    model.fit(df)
    for h in range(4):
        y_h = df["y1"].shift(-h) - df["y1"].shift(1)
        exog = pd.DataFrame(np.ones((len(df), 1)), columns=["const"], index=df.index)
        data = pd.concat([y_h.rename("__outcome"), exog, df[["y2", "z"]]], axis=1).dropna()
        res = IV2SLS(data["__outcome"], data[["const"]], data[["y2"]], data[["z"]]).fit(cov_type="unadjusted", debiased=True)
        assert np.isclose(model.irf_coefficients[h], res.params["y2"])
        assert np.isclose(model.irf_std_errors[h], res.std_errors["y2"])