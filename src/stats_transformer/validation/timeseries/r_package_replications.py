from dataclasses import dataclass

@dataclass
class VarsCanadaBenchmark:
    """R vars::Canada dataset, Pfaff JSS vignette, 1980Q1–2000Q4."""
    # VARselect results (per vignette example on page X)
    varselect_aic_lag = 3
    varselect_hq_lag = 2
    varselect_sc_lag = 1
    varselect_fpe_lag = 3
    # VAR(p=1, type="both") log-likelihood and companion roots
    var_p1_loglevel = -207.525
    companion_roots = (0.95, 0.95, 0.904, 0.751)
    # Diagnostic test statistics and p-values
    portmanteau_q16_stat = 233.5
    portmanteau_q16_pvalue = 0.61
    jarque_bera_stat = 9.92
    jarque_bera_pvalue = 0.27
    arch_lm_stat = 570.1
    arch_lm_pvalue = 0.02
    # Johansen cointegration tests
    johansen_trace_r0_p3_stat = 84.92
    # SVEC long-run restricted model
    svec_loglevel = -161.838

@dataclass
class SvarsUsaBenchmark:
    """R svars::USA dataset, svars JSS vignette, changes-in-volatility SVAR."""
    # VAR(p=6, type="const") baseline
    var_p6_loglevel = -564.30
    # id.cv(SB=c(1979,3)) structural identification
    lambda_diag = (0.393, 0.192, 1.244)
    # Wald test: λ1 = λ3
    wald_lambda1_eq_lambda3_stat = 7.66
    wald_lambda1_eq_lambda3_pvalue = 0.01
    # Restricted (over-identified) model
    restricted_lr_stat = 8.734
    restricted_lr_pvalue = 0.033
