import numpy as np
import pandas as pd

from stats_transformer.models.timeseries.identification.proxy_svar import ProxySVARModel

N = 160
NAN_FIRST = 60
LAGS = 2


def _gk_style_frame():
    rng = np.random.default_rng(7)
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    for t in range(1, N):
        a[t] = 0.5 * a[t - 1] + rng.normal(0, 1)
        b[t] = 0.3 * b[t - 1] + 0.4 * a[t - 1] + rng.normal(0, 1)
        c[t] = 0.2 * c[t - 1] + 0.3 * b[t - 1] + rng.normal(0, 1)
    z = np.zeros(N)
    for t in range(1, N):
        z[t] = 0.7 * z[t - 1] + 0.8 * a[t - 1] + rng.normal(0, 1)
    z[:NAN_FIRST] = np.nan
    df = pd.DataFrame({"a": a, "b": b, "c": c, "z": z})
    df["date"] = pd.date_range("2000-01-01", periods=N, freq="MS")
    return df


def _fit_sorted_without_date_column(df):
    sorted_df = df.sort_values("date").drop(columns=["date"]).reset_index(drop=True)
    model = ProxySVARModel(target_variables=["a", "b", "c"], instrument_variable="z", maxlags=LAGS)
    model.fit(sorted_df)
    return np.asarray(model.impact_column, dtype=float)


def test_proxy_svar_full_sample_and_row_order_invariance():
    df = _gk_style_frame()
    reference = _fit_sorted_without_date_column(df)

    # date_column set, input rows in date order
    model = ProxySVARModel(target_variables=["a", "b", "c"], instrument_variable="z", date_column="date", maxlags=LAGS)
    model.fit(df)
    # the missing instrument rows must not shrink the VAR sample
    assert model.var_result.nobs == N - LAGS
    np.testing.assert_allclose(model.impact_column, reference, rtol=1e-12)

    # shuffled input rows must give the same result on the same data
    shuffled = ProxySVARModel(target_variables=["a", "b", "c"], instrument_variable="z", date_column="date", maxlags=LAGS)
    shuffled.fit(df.sample(frac=1, random_state=3))
    assert shuffled.var_result.nobs == N - LAGS
    np.testing.assert_allclose(shuffled.impact_column, reference, rtol=1e-12)

    # non-RangeIndex input labels must not break the instrument alignment
    relabelled = df.copy()
    relabelled.index = [f"obs{i}" for i in range(N)][::-1]
    relabelled = relabelled.sample(frac=1, random_state=4)
    relabelled_model = ProxySVARModel(target_variables=["a", "b", "c"], instrument_variable="z", date_column="date", maxlags=LAGS)
    relabelled_model.fit(relabelled)
    assert relabelled_model.var_result.nobs == N - LAGS
    np.testing.assert_allclose(relabelled_model.impact_column, reference, rtol=1e-12)