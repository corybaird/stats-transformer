import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.api import VAR
from stats_transformer.models.timeseries.reduced_form.bvar import BVARModel, minnesota_prior_moments
from examples.academic.jarocinski_karadi_2020 import JarocinskiKaradi2020Replication

def _simulate_var(seed=42, n=300, n_vars=3, p=2):
    np.random.seed(seed)
    A1 = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.05, 0.0, 0.3]])
    A2 = np.array([[0.1, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]])
    Y = np.zeros((n, n_vars))
    for t in range(2, n):
        Y[t] = A1 @ Y[t - 1] + A2 @ Y[t - 2] + np.random.multivariate_normal(mean=[0] * n_vars, cov=np.eye(n_vars) * 0.3)
    df = pd.DataFrame(Y, columns=["y1", "y2", "y3"])
    df["date"] = pd.date_range("2000-01-01", periods=n, freq="D")
    return df, Y

def test_bvar_posterior_mean_converges_to_ols_under_loose_prior():
    df, Y = _simulate_var()
    model = BVARModel(target_variables=["y1", "y2", "y3"], date_column="date", lags=2, lambda1=1e6, lambda2=1.0, lambda3=1.0, lambda4=1e6, n_draws=200)
    model.fit(df)

    ols = VAR(Y).fit(2)
    np.testing.assert_allclose(model.posterior_B_mean, ols.params, atol=1e-6)

def test_bvar_irf_credible_bands_bracket_ols_point_estimate():
    df, Y = _simulate_var()
    model = BVARModel(target_variables=["y1", "y2", "y3"], date_column="date", lags=2, lambda1=1e6, lambda2=1.0, lambda3=1.0, lambda4=1e6, n_draws=1000)
    model.fit(df)
    irf = model.compute_irf(horizon=8, response="y1", shock="y1")

    ols = VAR(Y).fit(2)
    ols_irf = ols.irf(8).orth_irfs[:, 0, 0]

    brackets = (irf["lower"].to_numpy() <= ols_irf) & (ols_irf <= irf["upper"].to_numpy())
    assert brackets.all()

def test_bvar_tighter_prior_shrinks_toward_prior_mean():
    # The Minnesota prior mean is a random walk (own-lag-1 coefficient = 1,
    # everything else = 0), so a tighter prior pulls the posterior toward
    # that prior mean, not toward zero -- own-lag-1 coefficients should move
    # closer to 1 and all other coefficients closer to 0 as lambda1 shrinks.
    df, _ = _simulate_var()
    loose = BVARModel(target_variables=["y1", "y2", "y3"], date_column="date", lags=2, lambda1=10.0, lambda2=1.0, lambda3=1.0, lambda4=100.0, n_draws=50)
    loose.fit(df)
    tight = BVARModel(target_variables=["y1", "y2", "y3"], date_column="date", lags=2, lambda1=0.01, lambda2=0.1, lambda3=1.0, lambda4=100.0, n_draws=50)
    tight.fit(df)

    prior_mean = loose.posterior_B_mean * 0
    for i in range(3):
        prior_mean[1 + i, i] = 1.0

    loose_dist = np.linalg.norm(loose.posterior_B_mean - prior_mean)
    tight_dist = np.linalg.norm(tight.posterior_B_mean - prior_mean)
    assert tight_dist < loose_dist

def test_bvar_posterior_draws_have_expected_shape():
    df, _ = _simulate_var()
    n_draws = 50
    model = BVARModel(target_variables=["y1", "y2", "y3"], date_column="date", lags=2, n_draws=n_draws)
    model.fit(df)
    k = 1 + 3 * 2
    assert model.B_draws.shape == (n_draws, k, 3)
    assert model.Sigma_draws.shape == (n_draws, 3, 3)
    for sigma in model.Sigma_draws:
        eigvals = np.linalg.eigvalsh(sigma)
        assert (eigvals > 0).all()

def test_bvar_minnesota_prior_variance_matches_hand_computation():
    # Kadiyala & Karlsson (1997) natural-conjugate form: each equation's
    # coefficient prior covariance is sigma_i^2 * V, so the Minnesota
    # scaling sigma_i^2/sigma_j^2 requires V entries tightness/sigma_j^2,
    # not tightness*sigma_j^2. Expected V computed by hand for
    # sigma_hat = [2, 0.5].
    sigma_hat = np.array([2.0, 0.5])
    Lambda_prior, V_prior = minnesota_prior_moments(2, 1, sigma_hat, lambda1=0.2, lambda2=0.5, lambda3=1.0, lambda4=100.0)

    expected = np.zeros(3)
    expected[0] = (sigma_hat.mean() * 100.0) ** 2
    for j in range(2):
        own = 0.2 ** 2
        cross = own * 0.5 ** 2
        weighted = (own + cross) / 2.0
        expected[1 + j] = weighted / sigma_hat[j] ** 2
    np.testing.assert_allclose(np.diag(V_prior), expected, rtol=1e-12)
    assert Lambda_prior[1, 0] == 1.0 and Lambda_prior[2, 1] == 1.0
    np.testing.assert_allclose(Lambda_prior, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))

def test_bvar_posterior_mean_is_scale_equivariant():
    # Rescaling variable y1 by c must scale the coefficients on y1 by 1/c
    # and the equation for y1 by c: the posterior is equivariant to units.
    # With sigma_j^2 multiplied in, the prior itself changes under rescaling
    # and the equivariance breaks.
    df, _ = _simulate_var()
    c = 10.0
    df_c = df.copy()
    df_c["y1"] = df_c["y1"] * c

    model = BVARModel(target_variables=["y1", "y2", "y3"], date_column="date", lags=2, lambda4=1e6, n_draws=10)
    model.fit(df)
    model_c = BVARModel(target_variables=["y1", "y2", "y3"], date_column="date", lags=2, lambda4=1e6, n_draws=10)
    model_c.fit(df_c)

    base, scaled = model.posterior_B_mean, model_c.posterior_B_mean
    rows_y1 = [1, 4]  # lag-1 and lag-2 rows of y1's own lags
    mask = np.ones(base.shape[0], dtype=bool)
    mask[rows_y1] = False
    np.testing.assert_allclose(scaled[rows_y1][:, 1:], base[rows_y1][:, 1:] / c, rtol=1e-8)
    np.testing.assert_allclose(scaled[rows_y1][:, 0], base[rows_y1][:, 0], rtol=1e-8)
    np.testing.assert_allclose(scaled[mask][:, 0], base[mask][:, 0] * c, rtol=1e-8)

def test_bvar_run_returns_metadata():
    df, _ = _simulate_var(n=150)
    model = BVARModel(target_variables=["y1", "y2", "y3"], date_column="date", lags=1, n_draws=50)
    metadata = model.run(df)
    assert "metrics" in metadata
    assert metadata["metrics"]["lags"] == 1

def test_jarocinski_karadi_2020_replication():
    result = JarocinskiKaradi2020Replication().run()
    assert "metrics" in result
    assert result["classification"] in ("information shock (mp1 and sp500 co-move)", "monetary policy shock (mp1 and sp500 move oppositely)")
    assert not result["irf_sp500_to_mp1"].empty
