import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.vector_ar.vecm import VECM
from stats_transformer.models.timeseries.structural.svec import SVEC, SVECModel


def test_svec_estimation():
    np.random.seed(42)
    y1 = np.cumsum(np.random.randn(120))
    y2 = y1 * 0.5 + np.random.randn(120)
    data = pd.DataFrame({"y1": y1, "y2": y2})

    vecm_model = VECM(data, k_ar_diff=1, coint_rank=1).fit()

    SR = np.array([[np.nan, 0.0], [np.nan, np.nan]])
    LR = np.array([[np.nan, np.nan], [np.nan, np.nan]])

    svec_fit = SVEC(vecm_model, SR=SR, LR=LR)
    matrices = svec_fit.get_structural_matrices()

    assert "SR" in matrices
    assert "LR" in matrices
    assert matrices["SR"].shape == (2, 2)
    assert np.isclose(matrices["SR"][0, 1], 0.0, atol=1e-5)
    assert np.isfinite(svec_fit.llf)


def test_svec_model_pipeline_interface():
    np.random.seed(42)
    y1 = np.cumsum(np.random.randn(100))
    y2 = y1 * 0.7 + np.random.randn(100)
    data = pd.DataFrame({"y1": y1, "y2": y2})

    SR = np.array([[np.nan, 0.0], [np.nan, np.nan]])
    model = SVECModel(target_variables=["y1", "y2"], k_ar_diff=1, coint_rank=1, SR=SR)
    metrics = model.fit(data)

    assert "num_observations" in metrics
    assert "log_likelihood" in metrics
    assert model.B_0.shape == (2, 2)
    assert "SVEC" in model.get_summary()


def test_svec_raises_when_all_starts_fail(monkeypatch):
    import stats_transformer.models.timeseries.structural.svec as svec_mod

    np.random.seed(42)
    y1 = np.cumsum(np.random.randn(120))
    y2 = y1 * 0.5 + np.random.randn(120)
    data = pd.DataFrame({"y1": y1, "y2": y2})
    vecm_result = VECM(data, k_ar_diff=1, coint_rank=1).fit()
    SR = np.array([[np.nan, 0.0], [np.nan, np.nan]])

    class _Failed:
        success = False
        fun = np.inf
        x = np.zeros(3)

    monkeypatch.setattr(svec_mod, "minimize", lambda *a, **k: _Failed())
    monkeypatch.setattr(svec_mod, "root", lambda *a, **k: _Failed())
    with pytest.raises(RuntimeError):
        SVEC(vecm_result, SR=SR)


def test_svec_raises_on_restriction_violation(monkeypatch):
    import stats_transformer.models.timeseries.structural.svec as svec_mod

    np.random.seed(42)
    y1 = np.cumsum(np.random.randn(120))
    y2 = y1 * 0.5 + np.random.randn(120)
    data = pd.DataFrame({"y1": y1, "y2": y2})
    vecm_result = VECM(data, k_ar_diff=1, coint_rank=1).fit()
    SR = np.array([[np.nan, 0.0], [np.nan, np.nan]])

    class _Violating:  # claims success but violates the SR[0, 1] = 0 restriction
        success = True
        fun = 1.0
        x = np.ones(4)  # root() sees full vec(B); violates SR[0, 1] = 0

    monkeypatch.setattr(svec_mod, "root", lambda *a, **k: _Violating())

    class _Failed:  # the free-direction fallback finds nothing either
        success = False
        fun = np.inf
        x = np.zeros(3)

    monkeypatch.setattr(svec_mod, "minimize", lambda *a, **k: _Failed())
    with pytest.raises(RuntimeError, match="restrictions") as excinfo:
        SVEC(vecm_result, SR=SR)
    assert excinfo.value.max_restriction_violation > 0.0


def test_lr_restriction_rows_match_columns_of_b():
    import stats_transformer.models.timeseries.structural.svec as svec_mod

    rng = np.random.default_rng(7)
    Xi = rng.normal(size=(4, 4))
    sr = np.full((4, 4), np.nan)
    sr[3, 1] = 0.0
    lr = np.full((4, 4), np.nan)
    lr[0, 1:4] = 0.0
    lr[1:4, 3] = 0.0
    E = svec_mod._restriction_matrix(Xi, sr, lr)
    B = rng.normal(size=(4, 4))
    out = E @ B.ravel()  # row-major vec
    idx = 0
    for i in range(4):
        for j in range(4):
            if not np.isnan(lr[i, j]):
                np.testing.assert_allclose(out[idx], (Xi @ B)[i, j])
                idx += 1
            if not np.isnan(sr[i, j]):
                np.testing.assert_allclose(out[idx], B[i, j])
                idx += 1


def test_free_directions_respect_restrictions():
    import stats_transformer.models.timeseries.structural.svec as svec_mod
    from scipy.linalg import null_space

    rng = np.random.default_rng(7)
    Xi = rng.normal(size=(4, 4))
    sr = np.full((4, 4), np.nan)
    sr[3, 1] = 0.0
    lr = np.full((4, 4), np.nan)
    lr[0, 1:4] = 0.0
    lr[1:4, 3] = 0.0
    E = svec_mod._restriction_matrix(Xi, sr, lr)
    Sb = null_space(E)
    for _ in range(5):
        g = rng.normal(size=Sb.shape[1])
        assert svec_mod._restriction_violation(Sb @ g, Xi, sr, lr) < 1e-12


def test_svec_over_identified_spec_fits():
    from pathlib import Path

    df = pd.read_csv(Path(__file__).parents[2] / "tests/integration/var/canada.csv")
    cols = ["prod", "e", "U", "rw"]
    nan = np.nan

    def fit_spec(sr_extra):
        SR = np.full((4, 4), nan)
        SR[3, 1] = 0.0
        SR[0, 0] = sr_extra
        LR = np.full((4, 4), nan)
        LR[0, 1:4] = 0.0
        LR[1:4, 3] = 0.0
        m = SVECModel(target_variables=cols, k_ar_diff=2, coint_rank=1, deterministic="coli", SR=SR, LR=LR)
        m.fit(df[cols].reset_index(drop=True))
        return m.svec_fit

    over = fit_spec(0.0)
    just = fit_spec(np.nan)

    def concentrated(fit):
        B = np.asarray(fit.SR_est)
        sig = np.asarray(fit.sigma_u)
        return 0.5 * fit.nobs * (np.linalg.slogdet(B @ B.T)[1] + np.trace(np.linalg.inv(B @ B.T) @ sig))

    assert over.max_restriction_violation <= over.restr_tol
    assert just.max_restriction_violation <= just.restr_tol
    # More restrictions cannot raise the likelihood (concentrated objective is
    # lower-is-better, so the over-identified value must be no better).
    assert concentrated(over) >= concentrated(just) - 1e-9
