# Model Extension Roadmap & Cross-Language Verification Guide

This document outlines the master roadmap for model extensions in `stats-transformer`. It pairs every planned and implemented model with its target source folder, mathematical specification, reference software target (R, Stata, MATLAB), and design principles.

---

## 1. Master Model Extension & Cross-Language Matrix

The table below provides a complete index of all models, target source folder locations under `src/stats_transformer/models/`, reference software benchmarks, and active verification status via our [`software_benchmarks/`](../../src/examples/software_benchmarks/benchmark_suite.py) framework.

| Domain / Family | Model Class | Target Source Folder | Method / Specification | R Target | Stata Target | MATLAB Target | Verification Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Regression** | `RegressionModel` | `models/regression/` | Ordinary Least Squares (OLS) | `stats::lm` | `regress` | `mldivide` | **Verified** |
| **Regression** | `RobustOLSModel` | `models/regression/` | Robust Covariance (HC1-HC3, HAC) | `sandwich::vcovHC` | `regress, robust` | `hac` | **Implemented** |
| **Regression** | `IV2SLSModel` | `models/regression/` | Two-Stage Least Squares (2SLS) | `AER::ivreg` | `ivregress 2sls` | Econ Toolbox | **Implemented** |
| **Regression** | `PanelRegressionModel` | `models/regression/` | Fixed Effects & Random Effects | `plm::plm` | `xtreg` | `fitlmem` | **Implemented** |
| **Regression** | `PanelIV2SLSModel` | `models/regression/` | 2SLS + Fixed Effects & Random Effects | `fixest::feols` | `xtivreg`, `fe` | Econ Toolbox | **Implemented** |
| **Regression** | `GMMModel` | `models/regression/` | Generalized Method of Moments (2-step, iterated, CUE) | `gmm::gmm` | `gmm` | Econ Toolbox | **Implemented** |
| **Regression** | `DiDModel` | `models/regression/` | Callaway-Sant'Anna Staggered Difference-in-Differences | `did::att_gt` | `csdid` | Custom | **Implemented** |
| **Discrete Choice** | `LogitModel` | `models/discrete/` | Binary Logit MLE | `stats::glm` | `logit` | `fitglm` | **Verified** |
| **Discrete Choice** | `ProbitModel` | `models/discrete/` | Binary Probit MLE | `stats::glm(probit)` | `probit` | `fitglm` | **Implemented** |
| **Unsupervised** | `PCAModel` | `models/unsupervised/` | Principal Component Analysis | `stats::prcomp` | `pca` | `pca` | **Verified** |
| **Unsupervised** | `KMeansModel` | `models/unsupervised/` | K-Means Clustering | `stats::kmeans` | `cluster kmeans` | `kmeans` | **Implemented** |
| **Time Series (RF)** | `VARModel` | `models/timeseries/reduced_form/` | Reduced-Form VAR ($p$) | `vars::VAR` | `var` | `varm` | **Implemented** |
| **Time Series (RF)** | `VECMModel` | `models/timeseries/reduced_form/` | Vector Error Correction Model | `urca::ca.jo` | `vec` | `vecm` | **Implemented** |
| **Time Series (RF)** | `RestrictedVAR` | `models/timeseries/reduced_form/` | Zero-Restricted Coefficient VAR | `vars::restrict` | `var (constrained)` | `varm` | **Implemented** |
| **Time Series (RF)** | `ARIMAModel` | `models/timeseries/reduced_form/` | Autoregressive Integrated Moving Average | `forecast::auto.arima` | `arima` | `arima` | **Implemented** |
| **Time Series (RF)** | `LocalProjectionsModel` | `models/timeseries/reduced_form/` | Local Projections (Jordà 2005) | `lpirfs::lp_lin` | `jorda` / `lproj` | Custom | **Implemented** |
| **Time Series (RF)** | `LocalProjectionsIVModel` | `models/timeseries/reduced_form/` | LP Instrumental Variables (Stock & Watson 2018) | `lpirfs::lp_lin_iv` | `lproj (iv)` | Custom | **Implemented** |
| **Time Series (RF)** | `DynamicFactorModel` | `models/timeseries/reduced_form/` | Dynamic Factor Model (Kalman Filter/Smoother, EM) | `MARSS` / `dfms` | `dfactor` | Custom | **Implemented** |
| **Time Series (Bayesian)** | `BVARModel` | `models/timeseries/reduced_form/` | Conjugate Normal-Inverse-Wishart BVAR (Minnesota prior) | `BVAR::bvar` | `bayes: var` | Custom | **Implemented** |
| **SVAR** | `SVARModel` | `models/timeseries/identification/` | Short-Run Cholesky / AB Restrictions | `vars::SVAR` | `svar` | VAR-Toolbox | **Implemented** |
| **SVAR** | `BlanchardQuahModel` | `models/timeseries/identification/` | Long-Run Impact Restrictions $C(1)$ | `vars::BQ` | `svar, lreq` | `VARmodel.m` (VAR-Toolbox) | **Verified** |
| **SVAR** | `ProxySVARModel` | `models/timeseries/identification/` | External Instrument SVAR-IV | `svars` | SVAR-IV | VAR-Toolbox | **Implemented** |
| **SVAR** | `SignZeroSVARModel` | `models/timeseries/identification/` | Sign & Zero Restrictions (Rubio-Ramírez et al.) | `BMR` / `VARsignR` | Custom | VAR-Toolbox | **Implemented** |
| **SVAR** | `VolatilitySVARModel` | `models/timeseries/identification/` | Identification by Volatility Breaks | `svars` | Custom | Custom | **Implemented** |
| **SVAR** | `IndependenceSVARModel` | `models/timeseries/identification/` | Non-Gaussian ICA Identification | `svars` | Custom | Custom | **Implemented** |
| **SVAR** | `CVMSVARModel` | `models/timeseries/identification/` | Cramér-von Mises Distance SVAR | `svars::id.cvm` | Custom | Custom | **Implemented** |
| **SVAR** | `NonGaussianSVARModel` | `models/timeseries/identification/` | Non-Gaussian Maximum Likelihood SVAR | `svars::id.ng` | Custom | Custom | **Implemented** |
| **SVAR** | `SVECModel` | `models/timeseries/structural/` | Structural VECM (King et al. 1991) | `vars::SVEC` | `svar` / `vec` | VAR-Toolbox | **Implemented** |
| **Non-Linear** | `TVARModel` | `models/timeseries/nonlinear/` | Two-Regime Threshold VAR | `tsDyn::TVAR` | `tvar` | Custom | **Implemented** |
| **Non-Linear** | `TVECMModel` | `models/timeseries/nonlinear/` | Threshold VECM | `tsDyn::TVECM` | `tvecm` | Custom | **Implemented** |
| **Non-Linear** | `STVARModel` | `models/timeseries/nonlinear/` | Smooth Transition VAR & GIRFs | `sstvars` | Custom | Custom | **Implemented** |

---

## 2. Cross-Language Validation Framework

All model implementations are systematically validated against native econometrics engines using our modular suite under [`src/examples/software_benchmarks/`](../../src/examples/software_benchmarks/benchmark_suite.py):

```
src/examples/software_benchmarks/
├── modules/                         # Engine adapters (r_engine.py, stata_engine.py, matlab_engine.py)
├── regression/                      # regression_benchmark.py (OLS vs R lm, Stata regress, MATLAB mldivide)
├── discrete/                        # logit_benchmark.py (Logit vs R glm, Stata logit)
├── unsupervised/                    # pca_benchmark.py (PCA vs R prcomp, Stata pca)
├── timeseries/                      # blanchard_quah_benchmark.py (SVAR vs MATLAB VAR-Toolbox)
└── benchmark_suite.py               # Unified execution entrypoint
```

### Execution Strategy
- **Host Engines**: StataNow 19.5 and MATLAB 2025b run natively via `pystata` and `matlab.engine`.
- **Containerized Engines**: R benchmarks execute inside Docker (`docker compose run r-benchmarks`) to maintain environment isolation.
- **Tolerances**: Point estimates and covariance matrices must achieve `rtol=1e-5` / `atol=1e-5` parity.

---

## 3. Core Architecture & System Design Principles

### 3.1 Separation of Responsibilities
1. **Estimator Layer (`models/`)**: Focuses purely on mathematical estimation, returning clean result objects without inline plotting or disk writing.
2. **Result Contracts (`models/base.py`)**: All models wrap estimates into normalized DataFrames and dictionary metadata.
3. **Visualization Layer (`visualization/`)**: Consumes normalized result objects to generate publication-ready figures.
4. **Pipeline Orchestration (`pipeline/`)**: Handles YAML parsing (`params.yaml`), frequency alignment, and artifact persistence.

### 3.2 Portable Data Contracts
- Outputs are exported strictly as portable, machine-readable artifacts (CSV, Parquet, JSON, LaTeX) rather than Python pickles.
- All coefficient matrices, impulse responses (IRF), forecast error variance decompositions (FEVD), and historical decompositions (HD) maintain explicit labeled dimensions (`horizon`, `variable`, `shock`).

---

## 4. Diagnostics & Failure Handling Policy

Every model implementation must enforce deterministic failure modes rather than swallowing exceptions:
- **Stationarity & Stability**: Companion matrix roots ($A_c$) checked for unit circle boundary violations.
- **Multivariate Normality & Autocorrelation**: Portmanteau $Q$-tests, Jarque-Bera, and ARCH-LM heteroskedasticity diagnostics included in standard output.
- **Identification Failures**: Infeasible sign/zero restrictions, zero accepted rotation draws, or weak external instruments ($F < 10$) raise explicit diagnostic warnings.

---

## 5. Implementation Tiers & Deferred Scope

- **Tier 1 (Linear Reduced-Form Baseline)**: OLS VAR, VECM, Lag Selection (`VARselect`), ARIMAModel, Restricted VAR, and diagnostics.
- **Tier 2 (Structural Identification)**: Short-Run SVAR, Long-Run Blanchard-Quah, SVEC, Sign/Zero Restrictions, and frequentist bootstrap confidence bounds.
- **Tier 3 (Data-Driven Identification)**: Rigobon Changes in Volatility, Non-Gaussian ICA independence, and distance covariance.
- **Tier 4 (Non-Linear Dynamics)**: Threshold VAR (TVAR), Threshold VECM (TVECM), Smooth Transition VAR (STVAR), and Generalized IRFs (GIRFs).
- **Tier 5 (Analytical Bayesian VAR)**: Conjugate Normal-Inverse-Wishart BVAR with a Minnesota prior (`BVARModel`), estimated via a closed-form posterior and direct posterior draws for IRF credible bands.
- **Deferred Scope**: MCMC samplers (Gibbs sampling, Metropolis-Hastings), hierarchical/non-conjugate priors, and general state-space Bayesian estimation remain intentionally deferred to maintain lightweight core dependencies. Analytical conjugate BVAR is in scope precisely because it requires no sampler and no new dependencies.
