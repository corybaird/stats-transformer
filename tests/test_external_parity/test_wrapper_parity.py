import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stats_transformer.models.discrete.logit import LogitModel
from stats_transformer.models.discrete.probit import ProbitModel
from stats_transformer.models.regression.iv import IV2SLSModel
from stats_transformer.models.regression.panel import PanelRegressionModel
from stats_transformer.models.regression.regression import RegressionModel
from stats_transformer.models.regression.robust_ols import RobustOLSModel
from stats_transformer.models.timeseries.arima import ARIMAModel
from stats_transformer.models.timeseries.identification.blanchard_quah import BlanchardQuahModel
from stats_transformer.models.timeseries.identification.proxy_svar import ProxySVARModel
from stats_transformer.models.timeseries.identification.svar import SVARModel
from stats_transformer.models.timeseries.reduced_form.local_projections import LocalProjectionsModel
from stats_transformer.models.timeseries.reduced_form.local_projections_iv import LocalProjectionsIVModel
from stats_transformer.models.timeseries.reduced_form.var import VARModel
from stats_transformer.models.timeseries.reduced_form.vecm import VECMModel
from stats_transformer.models.timeseries.structural.svec import SVECModel
from stats_transformer.models.unsupervised.unsupervised import PCAModel

ROOT = Path(__file__).resolve().parents[2]
REFS = ROOT / "tests" / "external_benchmarks"
LONGLEY_X = ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"]
MACRO = ["realgdp", "realcons", "realinv"]

pytestmark = pytest.mark.filterwarnings("ignore")


def _ref(software, case):
    return json.loads((REFS / software / (case + ".json")).read_text())["outputs"]


def _inputs(case):
    return json.loads((REFS / "matlab" / (case + ".json")).read_text())["inputs"]


def _vec(x):
    return np.asarray(x, dtype=float).ravel()


def _const_last_to_first(x):
    v = _vec(x)
    return np.r_[v[-1], v[:-1]]


def _csv(rel):
    return pd.read_csv(ROOT / rel)


def _row_sign_normalise(a):
    a = np.asarray(a, dtype=float)
    return a * np.sign(np.diag(a))[:, None]


def _col_sign_normalise(a):
    a = np.asarray(a, dtype=float).copy()
    for j in range(a.shape[1]):
        i = np.argmax(np.abs(a[:, j]))
        a[:, j] *= np.sign(a[i, j])
    return a


def _fit_ols(cls, **kwargs):
    m = cls(target="TOTEMP", independent_variables=LONGLEY_X, **kwargs)
    m.fit(_csv("data/examples/regression/longley.csv"))
    return m.model


def test_ols_longley_matches_r_stata_matlab():
    res = _fit_ols(RegressionModel)
    beta = res.params.to_numpy()
    np.testing.assert_allclose(beta, _vec(_ref("r", "ols_longley")["beta"]), rtol=1e-7)
    np.testing.assert_allclose(beta, _const_last_to_first(_ref("stata", "ols_longley")["beta"]), rtol=1e-7)
    np.testing.assert_allclose(beta, _vec(_ref("matlab", "ols_longley")["beta"]), rtol=1e-7)
    np.testing.assert_allclose(res.bse.to_numpy(), _vec(_ref("r", "ols_longley")["se_classical"]), rtol=1e-7)


@pytest.mark.parametrize("cov_type", ["HC1", "HC2", "HC3"])
def test_robust_ols_longley_matches_r_and_stata(cov_type):
    res = _fit_ols(RobustOLSModel, cov_type=cov_type)
    key = "se_" + cov_type.lower()
    np.testing.assert_allclose(res.bse.to_numpy(), _vec(_ref("r", "ols_longley")[key]), rtol=1e-6)
    np.testing.assert_allclose(res.bse.to_numpy(), _const_last_to_first(_ref("stata", "ols_longley")[key]), rtol=1e-6)


def test_iv2sls_mroz_matches_r_and_stata():
    df = _csv("data/examples/regression/mroz.csv")
    df = df[df["inlf"] == 1].copy()
    df["lwage"] = pd.to_numeric(df["lwage"], errors="coerce")
    m = IV2SLSModel(target="lwage", independent_variables=["exper", "expersq"], endogenous=["educ"], instruments=["motheduc", "fatheduc"], cov_type="unadjusted")
    m.fit(df[["lwage", "exper", "expersq", "educ", "motheduc", "fatheduc"]])
    res = m.model
    stata_order = ["educ", "exper", "expersq", "const"]
    np.testing.assert_allclose(res.params.to_numpy(), _vec(_ref("r", "iv_mroz")["beta"]), rtol=1e-7)
    np.testing.assert_allclose(res.params[stata_order].to_numpy(), _vec(_ref("stata", "iv_mroz")["beta"]), rtol=1e-7)
    np.testing.assert_allclose(res.std_errors[stata_order].to_numpy(), _vec(_ref("stata", "iv_mroz")["se"]), rtol=1e-6)


def test_panel_fe_grunfeld_matches_r_and_stata():
    df = _csv("data/examples/regression/grunfeld.csv")
    m = PanelRegressionModel(target="invest", independent_variables=["value", "capital"], entity_column="firm", time_column="year")
    m.fit(df)
    res = m.model
    np.testing.assert_allclose(res.params[["value", "capital"]].to_numpy(), _vec(_ref("r", "panel_grunfeld")["beta_fe"]), rtol=1e-7)
    np.testing.assert_allclose(res.params[["value", "capital", "const"]].to_numpy(), _vec(_ref("stata", "panel_grunfeld")["beta_fe"]), rtol=1e-7)
    np.testing.assert_allclose(res.std_errors[["value", "capital", "const"]].to_numpy(), _vec(_ref("stata", "panel_grunfeld")["se_fe"]), rtol=1e-6)


@pytest.mark.parametrize("cls,case", [(LogitModel, "logit_spector"), (ProbitModel, "probit_spector")])
def test_binary_choice_spector_matches_r_and_stata(cls, case):
    m = cls(target="GRADE", independent_variables=["GPA", "TUCE", "PSI"])
    m.fit(_csv("data/examples/discrete/spector.csv"))
    res = m.model
    np.testing.assert_allclose(res.params.to_numpy(), _vec(_ref("r", case)["beta"]), rtol=1e-6)
    np.testing.assert_allclose(res.params.to_numpy(), _const_last_to_first(_ref("stata", case)["beta"]), rtol=1e-6)
    np.testing.assert_allclose(res.bse.to_numpy(), _const_last_to_first(_ref("stata", case)["se"]), rtol=1e-5)
    np.testing.assert_allclose(res.llf, _vec(_ref("stata", case)["loglik"])[0], rtol=1e-8)


def test_pca_longley_matches_r_and_stata():
    m = PCAModel(features=LONGLEY_X, n_components=6)
    m.fit(_csv("data/examples/regression/longley.csv"))
    ratio = m.model.explained_variance_ratio_
    stata_ev = _vec(_ref("stata", "pca_longley")["eigenvalues"])
    np.testing.assert_allclose(ratio, stata_ev / stata_ev.sum(), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(ratio, _vec(_ref("r", "pca_longley")["eigenvalues"]) / 6.0, rtol=1e-8, atol=1e-12)
    loadings = np.abs(m.model.components_.T)
    np.testing.assert_allclose(loadings, np.abs(np.asarray(_ref("stata", "pca_longley")["loadings"])), atol=1e-8)


def test_var_macro_matches_r_and_stata():
    m = VARModel(target_variables=MACRO, maxlags=2)
    m.fit(_csv("data/examples/timeseries/macrodata.csv"))
    res = m.model
    r_coef = np.asarray(_ref("r", "var_macro")["coef"])
    r_coef = np.vstack([r_coef[-1:], r_coef[:-1]])
    np.testing.assert_allclose(res.params.to_numpy(), r_coef, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(np.asarray(res.sigma_u), np.asarray(_ref("r", "var_macro")["Sigma_dof"]), rtol=1e-8)
    np.testing.assert_allclose(np.asarray(res.sigma_u_mle), np.asarray(_ref("stata", "var_macro")["Sigma_ml"]), rtol=1e-8)


@pytest.mark.parametrize("deterministic,mcase,scase", [("ci", "h1star", "vecm_macro_rconst"), ("co", "h1", "vecm_macro")])
def test_vecm_macro_matches_matlab_and_stata(deterministic, mcase, scase):
    m = VECMModel(target_variables=MACRO, k_ar_diff=1, deterministic=deterministic)
    m.fit(_csv("data/examples/timeseries/macrodata.csv"))
    res = m.model
    beta = _vec(res.beta)[:3]
    alpha = _vec(res.alpha)
    stata_beta = _vec(_ref("stata", scase)["beta"])[:3]
    stata_alpha = _vec(_ref("stata", scase)["alpha"])
    np.testing.assert_allclose(beta / beta[0], stata_beta, rtol=1e-6)
    np.testing.assert_allclose(alpha * beta[0], stata_alpha, rtol=1e-6)
    mat = _ref("matlab", "vecm_macro")
    mb = _vec(mat[mcase + "_beta"])
    np.testing.assert_allclose(beta / beta[0], mb / mb[0], rtol=1e-6)


def test_blanchard_quah_matches_matlab_and_r():
    endo = np.asarray(_inputs("bq1989_long")["ENDO"], dtype=float)
    m = BlanchardQuahModel(target_variables=["y", "u"], maxlags=8)
    m.fit(pd.DataFrame(endo, columns=["y", "u"]))
    np.testing.assert_allclose(m.B_0, np.asarray(_ref("matlab", "bq1989_long")["B"]), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(m.B_0, np.asarray(_ref("r", "bq1989_long")["B_impact"]), rtol=1e-8, atol=1e-12)


def test_svar_a_model_canada_matches_r():
    df = _csv("tests/integration/var/canada.csv")
    mask = np.array([["E", 0, 0, 0], ["E", "E", 0, 0], [0, 0, "E", 0], ["E", 0, 0, "E"]], dtype=object)
    m = SVARModel(target_variables=["e", "prod", "rw", "U"], svar_type="A", A=mask, maxlags=2)
    m.fit(df)
    np.testing.assert_allclose(_row_sign_normalise(m.model.A), _row_sign_normalise(_ref("r", "svar_canada_a")["A"]), rtol=1e-4)


def test_arima_realgdp_matches_r_and_stata():
    m = ARIMAModel(target="realgdp", order=(1, 0, 1), date_column=None, trend="c")
    m.fit(_csv("data/examples/timeseries/macrodata.csv")[["realgdp"]])
    p = m.model.params
    r = _vec(_ref("r", "arima_realgdp")["coef"])
    got = np.array([p["const"], p["ar.L1"], p["ma.L1"]])
    # R and Stata differ from each other by about 1e-4 in the MA term; the log-likelihood is the sharp check
    np.testing.assert_allclose(got, [r[2], r[0], r[1]], rtol=1e-3)
    np.testing.assert_allclose(m.model.llf, _ref("r", "arima_realgdp")["loglik"], rtol=1e-7)
    np.testing.assert_allclose(m.model.llf, _vec(_ref("stata", "arima_realgdp")["scalars"])[1], rtol=1e-7)


def test_local_projections_plain_jt2025_matches_stata():
    inp = _inputs("lp_jt2025")
    ctrl = np.asarray(inp["CTRL"], dtype=float)
    df = pd.DataFrame({"lcpi": _vec(inp["ENDO"]), "rr_shock": _vec(inp["TREAT"]), "dlrgdp": ctrl[:, 0], "dlcpi": ctrl[:, 1], "dstir": ctrl[:, 2]})
    m = LocalProjectionsModel(target="lcpi", shock_var="rr_shock", horizon=17, controls=["dlrgdp", "dlcpi", "dstir"])
    m.fit(df)
    irf = np.asarray(_ref("stata", "lp_plain_jt2025")["irf"])
    np.testing.assert_allclose([r["effect"] for r in m.irf_results], irf[:, 1], rtol=1e-7)
    np.testing.assert_allclose([r["stderr"] for r in m.irf_results], irf[:, 2], rtol=1e-6)


def test_local_projections_iv_plain_jt2025_matches_stata():
    inp = _inputs("lpiv_jt2025")
    ctrl = np.asarray(inp["CTRL"], dtype=float)
    df = pd.DataFrame({"urate": _vec(inp["ENDO"]), "ffr": _vec(inp["TREAT"]), "infl": ctrl[:, 1], "rrcgshock": _vec(inp["IV"])})
    m = LocalProjectionsIVModel(target_variable="urate", shock_variable="ffr", instrument_variable="rrcgshock", control_variables=["infl"], horizons=48)
    m.fit(df)
    irf = np.asarray(_ref("stata", "lpiv_plain_jt2025")["irf"])
    np.testing.assert_allclose(m.irf_coefficients, irf[:, 1], rtol=1e-7)
    np.testing.assert_allclose(m.irf_std_errors, irf[:, 2], rtol=1e-6)


def test_proxy_svar_gk2015_matches_matlab():
    inp = _inputs("gk2015_iv")
    endo = np.asarray(inp["ENDO"], dtype=float)
    df = pd.DataFrame(endo, columns=["gs1", "logcpi", "logip", "ebp"])
    df["ff4_tc"] = _vec(inp["IV"])
    m = ProxySVARModel(target_variables=["gs1", "logcpi", "logip", "ebp"], instrument_variable="ff4_tc", maxlags=12)
    m.fit(df)
    np.testing.assert_allclose(m.impact_column, np.asarray(_ref("matlab", "gk2015_iv")["B"])[:, 0], rtol=1e-6)


def test_proxy_svar_gk2015_irf_matches_matlab():
    inp = _inputs("gk2015_iv")
    df = pd.DataFrame(np.asarray(inp["ENDO"], dtype=float), columns=["gs1", "logcpi", "logip", "ebp"])
    df["ff4_tc"] = _vec(inp["IV"])
    m = ProxySVARModel(target_variables=["gs1", "logcpi", "logip", "ebp"], instrument_variable="ff4_tc", maxlags=12)
    m.fit(df)
    ir = np.asarray(_ref("matlab", "gk2015_iv")["IR"], dtype=float)
    np.testing.assert_allclose(m.irf(horizons=ir.shape[0]).to_numpy(), ir[:, :, 0], rtol=1e-6, atol=1e-12)


GHYSELS_CH7 = [
    ("sim", "data/examples/timeseries/ghysels_ch7/simulated_cointegration.csv", ["x", "y"], slice(0, 400), 1),
    ("uk_r1", "data/examples/timeseries/ghysels_ch7/dataonly_uktermstructure.csv", ["r1", "r3", "r6", "r24"], slice(0, 216), 2),
    ("lei", "data/examples/timeseries/ghysels_ch7/dataonly_uslei.csv", ["cci", "cli"], slice(312, 540), 1),
]


@pytest.mark.parametrize("key,path,cols,rows,k_ar_diff", GHYSELS_CH7)
def test_vecm_ghysels_ch7_matches_r_and_stata(key, path, cols, rows, k_ar_diff):
    m = VECMModel(target_variables=cols, k_ar_diff=k_ar_diff, deterministic="ci")
    m.fit(_csv(path).iloc[rows][cols].reset_index(drop=True))
    beta = _vec(m.model.beta)
    beta_const = _vec(m.model.det_coef_coint)
    got = np.r_[beta, beta_const] / beta[0]
    np.testing.assert_allclose(got, _vec(_ref("r", "vecm_ghysels_ch7")[key + "_beta"]), rtol=1e-6)
    np.testing.assert_allclose(got, _vec(_ref("stata", "vecm_ghysels_ch7")[key + "_beta"]), rtol=1e-6)
    np.testing.assert_allclose(_vec(m.model.alpha) * beta[0], _vec(_ref("stata", "vecm_ghysels_ch7")[key + "_alpha"]), rtol=1e-6)


def test_vecm_ghysels_ch7_uk_rank3_matches_r():
    cols = ["r1", "r3", "r6", "r24"]
    m = VECMModel(target_variables=cols, k_ar_diff=2, deterministic="ci", coint_rank=3)
    m.fit(_csv("data/examples/timeseries/ghysels_ch7/dataonly_uktermstructure.csv").iloc[:216][cols].reset_index(drop=True))
    beta = np.asarray(m.model.beta, dtype=float)
    beta_const = np.asarray(m.model.det_coef_coint, dtype=float)
    got = np.vstack([beta, beta_const])
    # R fixes the top rows of each cointegrating vector to the identity (same as statsmodels)
    ref = np.asarray(_ref("r", "vecm_ghysels_ch7")["uk_r3_beta"])
    np.testing.assert_allclose(got, ref, atol=1e-8)
    np.testing.assert_allclose(np.asarray(m.model.alpha, dtype=float).T, np.asarray(_ref("r", "vecm_ghysels_ch7")["uk_r3_alpha"]), atol=1e-8)
    assert got.shape == (5, 3)




def test_svec_canada_matches_r():
    cols = ["prod", "e", "U", "rw"]
    nan = np.nan
    SR = np.full((4, 4), nan)
    SR[3, 1] = 0
    LR = np.full((4, 4), nan)
    LR[0, 1:4] = 0
    LR[1:4, 3] = 0
    m = SVECModel(target_variables=cols, k_ar_diff=2, coint_rank=1, deterministic="coli", SR=SR, LR=LR)
    m.fit(_csv("tests/integration/var/canada.csv")[cols].reset_index(drop=True))
    ref = _ref("r", "svec_canada")
    # Compare raw matrices; the scale of each column is identified here (the
    # ML optimum pins B B' to Sigma_u), so only the free sign is aligned by
    # flipping a column whose diagonal sign differs.
    got_sr = np.asarray(m.svec_fit.SR_est, dtype=float).copy()
    got_lr = np.asarray(m.svec_fit.LR_est, dtype=float).copy()
    ref_sr = np.asarray(ref["SR"], dtype=float)
    for j in range(4):
        if np.sign(got_sr[j, j]) != np.sign(ref_sr[j, j]):
            got_sr[:, j] *= -1
            got_lr[:, j] *= -1
    np.testing.assert_allclose(got_sr, ref_sr, atol=1e-6)
    np.testing.assert_allclose(got_lr, np.asarray(ref["LR"], dtype=float), atol=1e-6)
    assert m.svec_fit.max_restriction_violation <= m.svec_fit.restr_tol
