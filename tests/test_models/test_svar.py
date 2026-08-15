import numpy as np
import pandas as pd
import pytest
from stats_transformer.models.timeseries.identification.svar import SVARModel


def _synthetic_data(observations=150, n_vars=2):
    generator = np.random.default_rng(42)
    shocks = generator.normal(size=(observations, n_vars))
    values = np.zeros((observations, n_vars))
    transition = 0.4 * np.eye(n_vars) + 0.1 * (np.ones((n_vars, n_vars)) - np.eye(n_vars))
    for index in range(1, observations):
        values[index] = transition @ values[index - 1] + shocks[index]
    columns = ["output", "inflation", "rate"][:n_vars]
    return pd.DataFrame(values, columns=columns)


def test_svar_type_a_fits_with_auto_generated_mask():
    # statsmodels 0.14.6's SVAR has a real bug (confirmed independent of
    # this repo, see test_svar_type_a_k2_crashes_on_statsmodels_bug below):
    # svar_type='A' crashes whenever there is exactly 1 free parameter,
    # i.e. K=2 with a minimal Cholesky-style mask. K=3 has 3 free
    # parameters in the auto-generated mask and fits without issue, so it
    # is used here to exercise the real, working code path.
    data = _synthetic_data(n_vars=3)
    model = SVARModel(target_variables=["output", "inflation", "rate"], svar_type="A", maxlags=1)

    metrics = model.fit(data)

    assert model.model is not None
    assert metrics["num_observations"] is not None
    # Auto-generated A mask: diagonal fixed at 1.0, strictly-lower-triangular
    # free ('E'), strictly-upper-triangular fixed at 0.0 (recursive/Cholesky
    # style identification).
    assert model.A[0, 0] == 1.0
    assert model.A[1, 1] == 1.0
    assert model.A[1, 0] == "E"
    assert model.A[2, 0] == "E"
    assert model.A[0, 1] == 0.0


@pytest.mark.xfail(reason="statsmodels 0.14.6 SVAR bug: svar_type='A' with exactly 1 free parameter "
                          "(K=2, minimal Cholesky mask) crashes inside SVAR.loglike() with "
                          "'NumPy boolean array indexing assignment requires a 0 or 1-dimensional "
                          "input, input has 2 dimensions'. Reproduces against raw statsmodels with "
                          "no stats_transformer code involved; not something this library can fix. "
                          "K=3+ (3+ free parameters) is unaffected -- see test_svar_type_a_fits_with_auto_generated_mask.",
                    strict=True)
def test_svar_type_a_k2_crashes_on_statsmodels_bug():
    data = _synthetic_data(n_vars=2)
    model = SVARModel(target_variables=["output", "inflation"], svar_type="A", maxlags=1)
    model.fit(data)


def test_svar_type_b_fits_with_auto_generated_mask():
    data = _synthetic_data()
    model = SVARModel(target_variables=["output", "inflation"], svar_type="B", maxlags=1)

    metrics = model.fit(data)

    assert model.model is not None
    assert metrics["num_observations"] is not None
    # Auto-generated B mask: lower-triangular-and-diagonal free ('E'),
    # strictly upper triangular fixed at 0.0.
    assert model.B[0, 0] == "E"
    assert model.B[1, 0] == "E"
    assert model.B[0, 1] == 0.0


def test_svar_respects_explicit_a_matrix():
    # Uses K=3 to route around the statsmodels K=2/1-free-parameter bug
    # documented above.
    data = _synthetic_data(n_vars=3)
    A = np.array([[1.0, 0.0, 0.0], ["E", 1.0, 0.0], ["E", "E", 1.0]], dtype=object)
    model = SVARModel(target_variables=["output", "inflation", "rate"], svar_type="A", A=A, maxlags=1)

    model.fit(data)

    assert model.A[1, 0] == "E"
    assert model.A[0, 1] == 0.0


def test_svar_fit_raises_on_missing_columns():
    data = _synthetic_data()
    model = SVARModel(target_variables=["output", "inflation", "unemployment"], maxlags=1)
    with pytest.raises(ValueError, match="Missing columns"):
        model.fit(data)


def test_svar_sorts_by_date_column_when_present():
    # svar_type="B" (the default here is left implicit as "A", so pass "B"
    # explicitly) to route around the K=2 statsmodels bug documented above;
    # date-sorting is identical regardless of svar_type.
    data = _synthetic_data()
    data["date"] = pd.date_range("2020-01-01", periods=len(data), freq="MS")
    shuffled = data.sample(frac=1.0, random_state=1).reset_index(drop=True)

    model = SVARModel(target_variables=["output", "inflation"], svar_type="B", date_column="date", maxlags=1)
    model.fit(shuffled)

    assert model.df_clean["date"].is_monotonic_increasing


def test_svar_get_model_metrics_raises_before_fit():
    model = SVARModel(target_variables=["output", "inflation"])
    with pytest.raises(ValueError, match="Model not trained"):
        model.get_model_metrics()


def test_svar_get_summary_raises_before_fit():
    model = SVARModel(target_variables=["output", "inflation"])
    with pytest.raises(ValueError, match="Model not trained"):
        model.get_summary()


def test_svar_get_summary_after_fit():
    data = _synthetic_data()
    model = SVARModel(target_variables=["output", "inflation"], svar_type="B", maxlags=1)
    model.fit(data)

    summary = model.get_summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_svar_build_model_raises_without_data():
    model = SVARModel(target_variables=["output", "inflation"])
    with pytest.raises(ValueError, match="No cleaned data available"):
        model.build_model()


def test_svar_get_required_columns_includes_date_column():
    model = SVARModel(target_variables=["output", "inflation"], date_column="date")
    assert model.get_model_metadata  # sanity: model constructed without error
    assert model._get_required_columns() == ["output", "inflation", "date"]


def test_svar_get_required_columns_without_date_column():
    model = SVARModel(target_variables=["output", "inflation"])
    assert model._get_required_columns() == ["output", "inflation"]
