# Software Benchmarks & Paper Replications Guide

This document details the cross-language numerical verification procedures against external econometrics software (Stata, MATLAB, R) and catalogs all runnable academic paper replications in `stats-transformer`.

---

## 1. Verification Levels

Every example script and model implementation in `stats-transformer` is categorized under one of four verification levels:

1. **Cross-Language Verified (MATLAB / R / Stata)**: A wrapper test in `tests/test_external_parity/` passes against at least one recorded external output at the tolerance listed in Section 2.1.
2. **Direct Python Verified (`statsmodels` / `linearmodels`)**: Estimated numerical outputs match underlying Python packages on identical input data.
3. **Paper Replication Example**: Translates a published paper's econometric specification, data transformations, and model structure on real research data.
4. **Data & Feature Pipeline Demo**: Demonstrates automated data ingestion, frequency resampling, and featurization pipelines.

---

## 2. Software Parity Benchmarks (Stata, MATLAB, R)

This section records which `stats-transformer` model wrappers match external software (StataNow 19.5 MP, MATLAB R2025b, R 4.6.1) and lays out the work needed for the rest.

- **Recorded outputs**: `tests/external_benchmarks/{matlab,r,stata}/` hold JSON outputs recorded from MATLAB R2025b, R 4.6.1 (Docker image `st-ref-r:1`) and StataNow 19.5 MP, together with the driver scripts and recorders that produced them. None of the recorders import `stats-transformer`.
- **Wrapper tests**: `tests/test_external_parity/test_wrapper_parity.py` fits each wrapper on the recorded inputs and asserts against the JSON. These tests need no external software and run in CI.
- **Rule**: a model is **Verified** only when a wrapper test passes against at least one recorded external output at a stated tolerance. A known mismatch is recorded as a strict `xfail` that names the bug, so a fix turns it into a pass.
- **Rerun recordings**: `uv run python -m tests.external_benchmarks.<matlab.record_matlab_refs | r.record_r_refs | stata.record_stata_refs>`; see `tests/external_benchmarks/README.md` and Running the Benchmarks below.

### 2.1 Verified Models (MATLAB, Stata, R)

| Family | Model | Target Software / Function | Benchmark Dataset | Verification Status & Max Discrepancy |
| --- | --- | --- | --- | --- |
| **Regression** | `RegressionModel` | R (`stats::lm`), Stata (`regress`), MATLAB (`mldivide`) | Longley (`data/examples/regression/longley.csv`) | **Verified** (relative $10^{-7}$) |
| **Regression** | `RobustOLSModel` | R (`sandwich::vcovHC`), Stata (`regress, vce(hc1/hc2/hc3)`) | Longley (`data/examples/regression/longley.csv`) | **Verified** (HC1, HC2, HC3, $10^{-6}$) |
| **Regression** | `IV2SLSModel` | R (`AER::ivreg`), Stata (`ivregress 2sls`) | Mroz (`data/examples/regression/mroz.csv`) | **Verified** ($10^{-6}$) |
| **Regression** | `PanelRegressionModel` | R (`plm::plm`), Stata (`xtreg, fe`) | Grunfeld (`data/examples/regression/grunfeld.csv`) | **Verified** (fixed effects, $10^{-6}$) |
| **Discrete** | `LogitModel` | R (`stats::glm`), Stata (`logit`) | Spector & Mazzeo (`data/examples/discrete/spector.csv`) | **Verified** ($10^{-5}$) |
| **Discrete** | `ProbitModel` | R (`stats::glm`), Stata (`probit`) | Spector & Mazzeo (`data/examples/discrete/spector.csv`) | **Verified** ($10^{-5}$) |
| **Unsupervised** | `PCAModel` | R (`stats::prcomp`), Stata (`pca`) | Longley (`data/examples/regression/longley.csv`) | **Verified** ($10^{-8}$) |
| **Time Series** | `VARModel` | R (`vars::VAR`), Stata (`var`) | US macro growth rates (`data/examples/timeseries/macrodata.csv`) | **Verified** ($10^{-8}$) |
| **Time Series** | `VECMModel` | R (`urca::ca.jo`), Stata (`vec`), MATLAB (`vecm`) | Ghysels & Marcellino Ch 7 (`data/examples/timeseries/ghysels_ch7/`), US macro | **Verified** (rank 1 and rank 3, `ci` and `co`, $10^{-6}$) |
| **Time Series** | `ARIMAModel` | R (`stats::arima`), Stata (`arima`) | US macro growth rates (`data/examples/timeseries/macrodata.csv`) | **Verified** (log-likelihood $10^{-7}$) |
| **Time Series** | `LocalProjectionsModel` | Stata (`regress` per horizon, HC3) | Jorda & Taylor (`data/examples/matlab_examples/JT2025_Data.xlsx`) | **Verified** (level-lead specification, $10^{-6}$) |
| **Time Series** | `LocalProjectionsIVModel` | Stata (`ivregress 2sls` per horizon) | Jorda & Taylor (`data/examples/matlab_examples/JT2025_Data.xlsx`) | **Verified** (level-lead specification, $10^{-6}$) |
| **Time Series** | `BlanchardQuahModel` | MATLAB (`VAR-Toolbox 4.0`), R (`vars::BQ`) | Blanchard & Quah (`data/examples/matlab_examples/BQ1989_Data.xlsx`) | **Verified** ($2.22 \times 10^{-16}$ vs MATLAB) |
| **Time Series** | `SVARModel` | R (`vars::SVAR`) | Canada (`tests/integration/var/canada.csv`) | **Verified** (A-model, $10^{-4}$) |
| **Time Series** | `SVECModel` | R (`vars::SVEC`) | Canada (`tests/integration/var/canada.csv`) | **Verified** ($10^{-6}$; column signs aligned) |

### 2.2 Non-Verified Models

#### 2.2.1 Known Mismatches (strict `xfail`)

| Model | Cause | Reference |
| --- | --- | --- |
| `ProxySVARModel` | impact column scaled by sqrt(Sigma_11) instead of the Gertler-Karadi / Mertens-Ravn shock-size formula; rows with a missing instrument are dropped before the VAR is estimated | MATLAB `gk2015_iv` |

#### 2.2.2 Recorded References Not Yet Usable

| Reference | Why not |
| --- | --- |
| MATLAB `lp_jt2025`, `lpiv_jt2025` | paper spec uses long differences and lagged controls, which the LP wrappers do not support |
| MATLAB `bvar_kl_minnesota` | Litterman fixed-Sigma prior; `BVARModel` uses a natural-conjugate prior |
| MATLAB `svar_keating` | Keating (1992) model with an equality restriction; MATLAB only, Stata and R did not converge |
| MATLAB `gk2015_short`, `sw2001_short` | Cholesky SVAR outputs; no wrapper test written yet |

#### 2.2.3 Library Bug Fixes (separate branch)

All bugs below are in `stats-transformer`, not in R, Stata or MATLAB. Fix on a dedicated branch, then add or flip the matching parity test.

| Pri | Bug | Location | Verification hook after the fix |
| --- | --- | --- | --- |
| 1 | Cointegration rank fixed at 1 | `vecm.py:32` | flips `test_vecm_ghysels_ch7_uk_rank3_matches_r` |
| 1 | Proxy SVAR scaling and instrument sample | `proxy_svar.py:49-63` | flips `test_proxy_svar_gk2015_matches_matlab` |
| 1 | Minnesota prior multiplies by sigma_j squared — FIXED: divides by sigma_j^2 | `bvar.py` (`minnesota_prior_moments`, shared with `pandemic_bvar.py`, `bayesian_sign_zero.py`) | `test_bvar_minnesota_prior_variance_matches_hand_computation`, `test_bvar_posterior_mean_is_scale_equivariant` (both fail on main); exact parity with `bvar_kl_minnesota` NOT expected (Litterman fixed-Sigma prior, different family) |
| 1 | Sign-zero "exact zeros" use a 1e-10 tolerance | `sign_zero.py:158`, `bayesian_sign_zero.py:121` | deterministic test with an injected rotation |
| 1 | TVECM error-correction term lag | `tvecm.py:63` | R `tsDyn::TVECM` |
| 1 | TVAR / TVECM empty-regime fallback | `tvar.py:140`, `tvecm.py:148` | R `tsDyn::TVAR` |
| 2 | DiD controls unused, standard errors | `did.py:14` | R `did::att_gt`, Stata `csdid` |
| 2 | Ghysels Ch 7 example uses `co` on 500 rows; book uses `ci` on rows 1 to 400 | `src/examples/timeseries/ghysels_chap7.py` | existing `vecm_ghysels_ch7` references |

#### 2.2.4 Future Cross-Checks

Legend: **Code** new driver or test, **Data** dataset to add or move, **Lib** wrapper fix or option, **Infra** package to install.

| Model | Code | Data | Lib | Infra |
| --- | --- | --- | --- | --- |
| `VolatilitySVARModel` | R `svars::id.cv` driver | move `tests/integration/svar/data_usa.csv` | fix regime split | `svars` |
| `IndependenceSVARModel`, `CVMSVARModel`, `NonGaussianSVARModel` | R `id.dc`, `id.cvm`, `id.ngml` drivers | `data_usa.csv` | fixed start or seed option | `svars` |
| `TVARModel` | R `tsDyn::TVAR` driver | US macro | empty-regime fix | `tsDyn` |
| `TVECMModel` | R `tsDyn::TVECM` driver | Ghysels Ch 7 | ECT lag fix | `tsDyn` |
| `BVARModel` | R `BVAR` driver with fixed hyperparameters | US macro | prior fix; optional Litterman variant | `BVAR` |
| `PandemicBVARModel` | MATLAB driver | Cascaldi-Garcia (2022) code and data | none | download |
| `SignZeroSVARModel`, `BayesianSignZeroSVARModel` | fixed-rotation tests; MATLAB `Replic/Uhlig2005`, `ADRR2018` for medians | present | rotation or seed injection; zero tolerance fix | none |
| `LocalProjectionsModel`, `LocalProjectionsIVModel` (paper spec) | wrapper tests against MATLAB `lp_jt2025`, `lpiv_jt2025` | present | long-difference and lagged-control options | none |
| `KMeansModel` | MATLAB and R drivers with fixed starts | Longley or Spector | accept initial centres | none |
| `PanelIV2SLSModel` | Stata `xtivreg, fe`, R `plm` drivers | panel IV dataset | none | none |
| `GMMModel` | Stata `gmm`, R `gmm` drivers | Mroz or Coibion-Gorodnichenko | clarify weight matrix | R `gmm` |
| `DiDModel` | R `did`, Stata `csdid` drivers | staggered-adoption dataset | controls fix | `did`, `ssc install csdid` |
| `DynamicFactorModel` | MATLAB `ssm` or R `dfms` driver | real panel (current input is synthetic) | tolerance and start control | `dfms` |

#### 2.2.5 Housekeeping

- **R image pin**: `tests/external_benchmarks/r/Dockerfile` uses `r-base:4.4.2`, but apt pulls Debian sid and the image reports R 4.6.1. Pin by digest or switch to a versioned base.
- **Data location**: move `canada.csv` (`vars::Canada`) and `data_usa.csv` (`svars::USA`) to `data/examples/timeseries/` and document them in `docs/library/data.md`.
- **Data documentation**: `macrodata.csv` is 100 x log-difference of statsmodels macrodata (quarterly growth rates); say so in `docs/library/data.md` and use Ghysels Ch 7 for cointegration examples.
- **Legacy suites**: retire or migrate `tests/integration/*.R` (8-digit JSON, raw-library tests) and treat `src/examples/software_benchmarks/` as an optional live tool, not as evidence.
- **Docs and paper**: `docs/library/models.md` now follows Section 2.1. Table 1 of the paper (separate Overleaf repo) still needs `SVARModel`, `ARIMAModel`, `ProbitModel`, `RobustOLSModel`, `IV2SLSModel` and the two LP models updated, and `VolatilitySVARModel` kept out of the verified tier.

### Running the Benchmarks

Run the wrapper parity tests (no external software needed):

```bash
/opt/homebrew/bin/uv run python -m pytest tests/test_external_parity
```

Re-record the external outputs (needs MATLAB with a shared engine, Docker image `st-ref-r:1`, or Stata):

```bash
/opt/homebrew/bin/uv run python -m tests.external_benchmarks.matlab.record_matlab_refs
/opt/homebrew/bin/uv run python -m tests.external_benchmarks.r.record_r_refs
/opt/homebrew/bin/uv run python -m tests.external_benchmarks.stata.record_stata_refs
```

The older live suite in `src/examples/software_benchmarks/` remains available as an optional tool but is not used as evidence:

```bash
/opt/homebrew/bin/uv run python -m src.examples.software_benchmarks.benchmark_suite
```

---

## 3. Cross-Language MATLAB Comparator

The MATLAB comparator in `src/examples/software_benchmarks/matlab_comparator.py` provides an opt-in verification tool against Ambrogio Cesa-Bianchi's MATLAB VAR-Toolbox 4.0.

- **Benchmark Dataset**: Blanchard & Quah (1989) quarterly macro data in `data/examples/matlab_examples/BQ1989_Data.xlsx`.
- **System Variables**: GDP growth ($\Delta y_t$) and unemployment rate ($u_t$).
- **Lag Structure**: VAR(8) with an intercept.
- **Identification Scheme**: Blanchard & Quah (1989) long-run structural restrictions ($C(1)$ lower-triangular).
- **Observed Discrepancy**: Maximum absolute difference: **$2.22 \times 10^{-16}$** (machine precision), re-confirmed against the recorded MATLAB output in `tests/external_benchmarks/matlab/bq1989_long.json`.

This comparator predates the recorded benchmarks; the `bq1989_long` case in Section 2 supersedes it as evidence. Execute it directly:

```bash
/opt/homebrew/bin/uv run python -m src.examples.software_benchmarks.matlab_comparator
```

---

## 4. Academic Paper Replications & Example Scripts

This section catalogs all 27 runnable example modules in `src/examples/`, detailing their academic source papers, module paths, data source files, and empirical verification role.

| Domain / Method | Script Module Path | Academic Paper / Benchmark Target | Data Source File | Verification Status | Target Verification / Compared Object |
| --- | --- | --- | --- | --- | --- |
| **MATLAB Comparator** | `src.examples.software_benchmarks.matlab_comparator` | Blanchard & Quah (1989) / MATLAB VAR-Toolbox 4.0 | `data/examples/matlab_examples/BQ1989_Data.xlsx` | **Cross-Language Verified (MATLAB)** | Structural impact matrix $C(1)$ ($2.22 \times 10^{-16}$ max diff) |
| **Structural VAR** | `src.examples.academic.var.blanchard_quah_1989` | Blanchard & Quah (1989) | `data/examples/matlab_examples/BQ1989_Data.xlsx` | **Cross-Language Verified (MATLAB, R)** | Long-run structural supply & demand shock identification |
| **Proxy SVAR / SVAR-IV** | `src.examples.academic.var.gertler_karadi_2015` | Gertler & Karadi (2015) | `data/examples/matlab_examples/GK2015_Data.xlsx` | **Paper Replication Example** | External-instrument monetary policy shock identification |
| **LP-IV Local Projections** | `src.examples.academic.var.jorda_taylor_2025` | Jordà & Taylor (2025) / Stock & Watson (2018) | `data/examples/matlab_examples/JT2025_Data.xlsx` | **Paper Replication Example** | Instrumental-variable impulse response functions |
| **Reduced-Form VAR** | `src.examples.academic.var.stock_watson_2001` | Stock & Watson (2001) | `data/examples/matlab_examples/SW2001_Data.xlsx` | **Paper Replication Example** | 3-variable macro VAR (Inflation, Unemployment, Fed Funds) |
| **Reduced-Form VAR** | `src.examples.timeseries.macro_var` | `statsmodels.tsa.vector_ar.var_model` | `data/examples/timeseries/macrodata.csv` | **Cross-Language Verified (R, Stata)** | VAR(2) coefficient matrices & standard error parity |
| **SVAR Identification** | `src.examples.timeseries.kilian_svar` | Kilian & Lütkepohl (2017) | `data/examples/timeseries/macrodata.csv` | **Paper Replication Example** | Short-run Cholesky & A-model structural identification |
| **Johansen VECM** | `src.examples.timeseries.kilian_vecm` | Johansen (1991) / Kilian & Lütkepohl (2017) | `data/examples/timeseries/macrodata.csv` | **Paper Replication Example** | Cointegration rank test & error-correction dynamics |
| **VAR & Forecasting** | `src.examples.timeseries.ghysels_chap6` | Ghysels & Marcellino (2018) Chapter 6 | `data/examples/timeseries/ghysels_ch6/var_simulated_ch6_sec9.csv` | **Textbook Replication Example** | Multi-step forecasting & simulated VAR impulse responses |
| **VECM & Cointegration** | `src.examples.timeseries.ghysels_chap7` | Ghysels & Marcellino (2018) Chapter 7 | `data/examples/timeseries/ghysels_ch7/simulated_cointegration.csv` | **Textbook Replication Example** | UK term structure cointegration & vector error correction |
| **High-Frequency Shock** | `src.examples.academic.nakamura_steinsson` | Nakamura & Steinsson (2018) | `data/examples/academic/nakamura_steinsson_2018/NominalYields.csv` | **Paper Replication Example** | Daily first difference of Fed Funds futures surprise series |
| **PCA Shock Extraction** | `src.examples.academic.nakamura_steinsson_pca` | Nakamura & Steinsson (2018) | `data/examples/academic/nakamura_steinsson_2018/master.dta` | **Paper Replication Example** | First principal component extraction from monetary futures |
| **High-Frequency Shock** | `src.examples.academic.bauer_swanson` | Bauer & Swanson (2023) | `data/examples/academic/bauer_swanson_2023/cpi.txt` | **Paper Replication Example** | Monthly orthogonalized monetary surprise transformations |
| **High-Frequency Shock** | `src.examples.academic.bauer_bernanke_milstein` | Bauer, Bernanke, & Milstein (2023) | `data/examples/academic/bbm_2023/feds_subset.csv` | **Paper Replication Example** | Daily difference and percentage-change transformation logic |
| **Bayesian VAR / Sign Identification** | `src.examples.academic.jarocinski_karadi_2020` | Jarociński & Karadi (2020) | `data/examples/academic/jarocinski_karadi_2020/fomc_surprises.parquet.gzip` | **Paper Replication Example** | Conjugate BVAR with sign-restriction shock classification |
| **Collinear OLS** | `src.examples.regression.longley` | Longley (1967) | `data/examples/regression/longley.csv` | **Cross-Language Verified (R, Stata, MATLAB)** | OLS & Robust OLS numerical stability under collinearity |
| **Panel Regression** | `src.examples.regression.grunfeld` | Grunfeld (1958) | `data/examples/regression/grunfeld.csv` | **Cross-Language Verified (R, Stata)** | Fixed-effects corporate investment panel regression |
| **Instrumental Variables** | `src.examples.regression.mroz_iv` | Mroz (1987) | `data/examples/regression/mroz.csv` | **Cross-Language Verified (R, Stata)** | 2SLS female labor supply hours equation |
| **Mincer Wage Equation** | `src.examples.regression.mincer_wage` | Mincer (1974) | `data/examples/regression/mroz.csv` | **Textbook Replication Example** | Semi-logarithmic human capital wage regression |
| **Okun's Law** | `src.examples.regression.okuns_law` | Okun (1962) | `data/examples/timeseries/macrodata.csv` | **Textbook Replication Example** | GDP growth vs unemployment rate change regression |
| **Applied Regression** | `src.examples.regression.ghysels_chap1` | Ghysels & Marcellino (2018) Chapter 1 | `data/examples/regression/ghysels_ch1/ex2_regress_gdp.csv` | **Textbook Replication Example** | Linear trend & seasonal dummy regression models |
| **Applied Regression** | `src.examples.regression.ghysels_chap2` | Ghysels & Marcellino (2018) Chapter 2 | `data/examples/regression/ghysels_ch2/ex1_regress_gdp.csv` | **Textbook Replication Example** | Autoregressive distributed lag (ARDL) forecasting |
| **Binary Discrete Choice** | `src.examples.discrete.spector_logit` | Spector & Mazzeo (1980) | `data/examples/discrete/spector.csv` | **Cross-Language Verified (R, Stata)** | Binary Logit educational choice estimation |
| **High-Frequency SOFR** | `src.examples.academic.acosta_brennan_jacobson_2024` | Acosta, Brennan, & Jacobson (2024) | `data/examples/academic/acosta_brennan_jacobson_2024/sofr_surprises.parquet.gzip` | **Paper Replication Example** | SOFR futures surprise VAR & Robust OLS estimation |
| **Policy Uncertainty** | `src.examples.academic.cieslak_hansen_mcmahon_xiao_2024` | Cieslak, Hansen, McMahon, & Xiao (2024) | `data/examples/academic/cieslak_hansen_mcmahon_xiao_2024/pmu_data.parquet.gzip` | **Paper Replication Example** | Policymakers' Uncertainty OLS & VAR estimation |
| **News Sentiment** | `src.examples.academic.shapiro_sudhof_wilson_2022` | Shapiro, Sudhof, & Wilson (2022) | `data/examples/academic/shapiro_sudhof_wilson_2022/news_sentiment.parquet.gzip` | **Paper Replication Example** | Daily news sentiment featurization & Robust OLS |
| **Industrial Policy** | `src.examples.academic.lane_2025` | Lane (2025) | `data/examples/academic/lane_2025/policy_loans.parquet.gzip` | **Paper Replication Example** | Targeted policy lending Robust OLS regression |
| **Dynamic Factor Model** | `src.examples.academic.miranda_agrippino_rey_2020` | Miranda-Agrippino & Rey (2020) | `data/examples/academic/miranda_agrippino_rey_2020/global_factor.parquet.gzip` | **Illustrative Method Demo (Synthetic Input)** | EM-estimated dynamic factor extraction from risky-asset panel |
| **Survey Forecast Rigidity** | `src.examples.academic.coibion_gorodnichenko_2012` | Coibion & Gorodnichenko (2012) | `data/examples/academic/coibion_gorodnichenko_2012/greenbook_forecast_errors.parquet.gzip` | **Paper Replication Example** | Two-step GMM forecast-error orthogonality test |
| **Provider Pipeline** | `src.examples.featurization.fred` | St. Louis Fed FRED API | `data/raw/fred_cache.parquet` | **Data & Feature Pipeline Demo** | Automated multi-series FRED data download & alignment |
| **Provider Pipeline** | `src.examples.featurization.monetary` | Federal Reserve Macro Series | `data/macrodb_gdp_inflation.parquet` | **Data & Feature Pipeline Demo** | Monetary policy indicator featurization pipeline |

---

## 5. Verification Protocol

When publishing or reporting empirical validation results, always document:
1. Exact dataset name, version, and date of download.
2. Feature transformations and sample adjustment rules.
3. Estimator specification (lags, deterministic terms, identification constraints).
4. Compared object (coefficients, standard errors, structural impact matrix, IRF paths).
5. Numerical tolerance and observed maximum absolute discrepancy.
