# Frequentist multivariate time-series extension roadmap

## Quick reference

- **Sections 1-3**: Objectives, current baseline, and reference projects (including note on adjacent `tsecon` library).
- **Section 4**: Design principles including §4.6 on avoiding duplicated procedural logic.
- **Section 5**: Package structure with explicit subfolder creation rule (create at 4+ files, not sooner).
- **Sections 6-10**: Architecture, result contracts, model extensions, figures and tables to reproduce.
- **Section 11**: Validation strategy, including §11.4 on R vignette golden number fixtures (tier 1: frozen Python constants; tier 2: optional Rscript re-derivation).
- **Sections 12-18**: Phases, testing, dependencies (R validation via subprocess, not rpy2), risks, milestones, definition of done, and deferred Bayesian scope.

## Contents

- [1. Objectives and boundaries](#1-objectives-and-boundaries)
- [2. Current baseline](#2-current-baseline)
- [3. Reference projects](#3-reference-projects)
- [4. Design principles](#4-design-principles)
- [5. Proposed package structure](#5-proposed-package-structure)
- [6. Execution and reporting architecture](#6-execution-and-reporting-architecture)
- [7. Normalized result contracts](#7-normalized-result-contracts)
- [8. Model extensions](#8-model-extensions)
- [9. Figures to reproduce](#9-figures-to-reproduce)
- [10. Tables to reproduce](#10-tables-to-reproduce)
- [11. Validation datasets and comparators](#11-validation-datasets-and-comparators)
- [12. Implementation phases](#12-implementation-phases)
- [13. Testing and acceptance criteria](#13-testing-and-acceptance-criteria)
- [14. Dependencies and compatibility](#14-dependencies-and-compatibility)
- [15. Risks and controls](#15-risks-and-controls)
- [16. Issue and milestone breakdown](#16-issue-and-milestone-breakdown)
- [17. Definition of done](#17-definition-of-done)
- [18. Deferred Bayesian scope](#18-deferred-bayesian-scope)

---

## 1. Objectives and boundaries

### 1.1 Objectives

The primary goal of this roadmap is to establish `stats-transformer` as a reference-grade Python package for frequentist multivariate time-series analysis.

- Provide a reliable, documented, tested implementation of reduced-form VAR and VECM estimation.
- Provide a flexible structural identification engine covering exact, over-identified, sign, zero, narrative, and data-driven restrictions.
- Support nonlinear multivariate time-series models including Threshold VAR (TVAR), Threshold VECM (TVECM), and Smooth Transition VAR (STVAR).
- Standardize estimation results into normalized tabular DataFrames and structured JSON representations.
- Connect all estimation models directly to visualization components and LaTeX/Overleaf table exporters.

### 1.2 User-facing position

- `stats-transformer` offers high-level YAML-configured pipelines alongside low-level modular OOP components.
- The library targets empirical economists, financial analysts, and quantitative researchers who require reproducible time-series workflows.
- All numerical outputs are benchmarked against canonical R packages (`vars`, `tsDyn`, `svars`, `sstvars`) and MATLAB toolboxes (VAR-Toolbox).

### 1.3 Non-goals for this roadmap

- **Bayesian methods:** Bayesian VAR (BVAR), Bayesian SVAR, MCMC sampling, and posterior inference are explicitly out of scope for this roadmap to maintain lean dependencies and clear system boundaries.
- **High-dimensional regularized VARs:** Penalized VAR models (LASSO, Ridge, Elastic Net) are deferred to a separate high-dimensional module proposal.
- **Direct GUI or web interfaces:** The core package focuses strictly on pythonic APIs, CLI execution, and reporting artifacts.

---

## 2. Current baseline

The existing repository features baseline implementations across single-equation regression, panel models, reduced-form VAR, local projections, and initial structural identification modules.

| Feature area | Status | Current module location | Target reference |
| --- | --- | --- | --- |
| Reduced-form OLS VAR | Implemented | `src/stats_transformer/models/timeseries/reduced_form/var.py` | R `vars::VAR` |
| Cointegration & VECM | Implemented | `src/stats_transformer/models/timeseries/reduced_form/vecm.py` | R `vars::ca.jo` / `vec2var` |
| Local Projections | Implemented | `src/stats_transformer/models/timeseries/reduced_form/local_projections.py` | Jordà (2005) |
| Local Projections IV | Implemented | `src/stats_transformer/models/timeseries/reduced_form/local_projections_iv.py` | Stock & Watson (2018) |
| Structural VAR (Short/Long-run) | Implemented | `src/stats_transformer/models/timeseries/identification/svar.py` | R `vars::SVAR` |
| Blanchard-Quah Identification | Implemented | `src/stats_transformer/models/timeseries/identification/blanchard_quah.py` | Blanchard & Quah (1989) |
| Proxy SVAR / SVAR-IV | Implemented | `src/stats_transformer/models/timeseries/identification/proxy_svar.py` | Mertens & Ravn (2013) |
| Sign & Zero Restrictions | Implemented | `src/stats_transformer/models/timeseries/identification/sign_zero.py` | Rubio-Ramírez et al. (2010) |
| Data-Driven Volatility SVAR | Implemented | `src/stats_transformer/models/timeseries/identification/volatility.py` | Rigobon (2003) |
| Data-Driven Independence SVAR | Implemented | `src/stats_transformer/models/timeseries/identification/independence.py` | Matteson & Tsay (2017) |
| Structural VECM (SVEC) | Implemented | `src/stats_transformer/models/timeseries/structural/svec.py` | R `vars::SVEC` |

---

## 3. Reference projects

### 3.1 Licensing rule

- Algorithms must be implemented independently from published mathematical literature and primary source papers.
- GPL-licensed source code from R or MATLAB packages must not be translated line-by-line into this MIT-licensed codebase.
- Output arrays and benchmark statistics from external packages may be used as independent validation targets.

### 3.2 Dataset rule

- Benchmark datasets embedded in tests must either use public domain macro data (e.g. FRED-QD, US macro data) or synthetic data generated by deterministic scripts.
- Benchmark targets and golden values are stored as frozen JSON or Python constants.

| Reference project | Location / Package | Domain / Purpose | Usage in stats-transformer |
| --- | --- | --- | --- |
| R `vars` | CRAN package `vars` | Reduced-form VAR, SVAR, SVEC, lag selection, diagnostic tests | Primary numerical benchmark for linear VAR workflows. |
| R `tsDyn` | CRAN package `tsDyn` | Threshold VAR (TVAR) and Threshold VECM (TVECM) | Primary benchmark for threshold regime-switching models. |
| R `svars` | CRAN package `svars` | Data-driven SVAR identification (volatility, independence, non-Gaussianity) | Primary benchmark for statistical identification methods. |
| R `sstvars` | CRAN package `sstvars` | Smooth Transition VAR (STVAR) and Generalized IRF (GIRF) | Primary benchmark for smooth transition dynamics. |
| MATLAB VAR-Toolbox | Ambito / Ambrogio Cesa-Bianchi | VAR, SVAR, Sign Restrictions, Historical Decompositions | Secondary cross-language numerical comparator. |
| Python `statsmodels` | `statsmodels.tsa.vector_ar` | Reduced-form VAR baseline | Comparison target for basic Python VAR utilities. |
| Python `tsecon` | GitHub repository `tsecon` | Rust/PyO3 macroeconomic package | Ecosystem monitoring only. Not an active parity target. |

---

## 4. Design principles

### 4.1 Separation of responsibilities

- **Models (`src/stats_transformer/models/`)**: Responsible solely for estimation, mathematical transformations, diagnostic calculations, and returning structured result dictionaries.
- **Visualization (`src/stats_transformer/visualization/`)**: Responsible solely for consuming normalized DataFrames or result dicts and rendering high-quality plots.
- **Reporting (`src/stats_transformer/reporting/`)**: Responsible solely for formatting tabular outputs, generating LaTeX/Overleaf tables, and persisting JSON summaries.
- **Pipeline (`src/stats_transformer/pipeline.py`)**: Responsible for orchestrating stage execution based on configuration YAMLs.

### 4.2 Composition over inheritance

- Maintain flat class hierarchies (maximum one parent base class).
- Subclass `ModelBase` for estimator contracts while composing helper objects for diagnostics, bootstrap, and structural rotations.

### 4.3 Explicit assumptions

- Models must never silently fill missing values or alter lag structures without user configuration.
- Input data requirements, deterministic terms, and sample adjustments must be declared explicitly in parameters or model initialization.

### 4.4 Stable result dimensions

- Numerical output objects must expose standardized attribute names: `coefficients`, `residuals`, `covariance_matrix`, `irf`, `fevd`, `hd`.
- Column names in result DataFrames follow consistent snake_case conventions (`response`, `shock`, `horizon`, `mean`, `lower`, `upper`).

### 4.5 Incremental migration

- New estimation models and features must integrate into existing `Pipeline` stages without breaking existing public API signatures.

### 4.6 Avoid duplicated procedural logic

- Adapters, table builders, and chart components repeatedly iterate over result dimensions (response, shock, horizon, bound) to generate long-format rows or panel indices.
- Shared iteration logic should be extracted into reusable generator helpers (e.g. `_iterate_response_shock_horizon`) to eliminate inline loop duplication.
- Chart components should delegate panel geometry layout math to shared visual utility functions.

---

## 5. Proposed package structure

The package layout adheres to Cookiecutter Data Science and the Rule of 3 subfolder creation policy (subfolders are introduced when 3 or more related files share a functional domain).

```
src/stats_transformer/models/timeseries/
├── __init__.py
├── base_irf.py
├── utilities.py
├── arima.py
├── reduced_form/
│   ├── __init__.py
│   ├── var.py
│   ├── vecm.py
│   ├── local_projections.py
│   ├── local_projections_iv.py
│   ├── lag_selection.py
│   ├── restrictions.py
│   └── forecasting.py
├── identification/
│   ├── __init__.py
│   ├── svar.py
│   ├── blanchard_quah.py
│   ├── proxy_svar.py
│   ├── sign_zero.py
│   ├── volatility.py
│   ├── independence.py
│   ├── alignment.py
│   └── bootstrap.py
├── structural/
│   ├── __init__.py
│   └── svec.py
├── nonlinear/
│   ├── __init__.py
│   ├── tvar.py
│   ├── tvecm.py
│   ├── stvar.py
│   └── girf.py
└── diagnostics/
    ├── __init__.py
    ├── residuals.py
    ├── stability.py
    └── stationarity.py
```

---

## 6. Execution and reporting architecture

### 6.1 End-to-end workflow

```
  [ YAML Config / Params ]
            │
            ▼
   [ Pipeline Runner ] ──► [ FeatureEngineer / DataMerger ]
            │
            ▼
    [ Model Estimator ] ──► [ Fitted Model Object ]
            │
            ├───────────────────────┬──────────────────────┐
            ▼                       ▼                      ▼
  [ Tabular Exporter ]     [ Visualizers ]      [ JSON Metadata ]
  (LaTeX / CSV / Parquet)   (Plots / IRF / HD)   (Reports / DVC)
```

### 6.2 Model-family structure

- Reduced-form models estimate system equations and provide log-likelihood, lag selection criteria, and residual covariance matrices.
- Structural identification modules accept fitted reduced-form models and compute impact matrices ($B$), structural shocks ($\epsilon$), and structural impulse responses.
- Nonlinear models estimate regime-specific parameter matrices and compute generalized impulse response paths via simulation.

### 6.3 Configuration boundary

- System options are configured via YAML files in `references/configs/`.
- YAML configurations specify dataset sources, target and independent variables, lag orders, identification method, restriction matrices, bootstrap replications, and plot output formatting.

---

## 7. Normalized result contracts

### 7.1 Specification

Every fitted time-series model exposes metadata containing:

- `model_type`: String identifier (e.g. `var`, `svar`, `vecm`, `tvar`).
- `variables`: List of variable names in system order.
- `lag_order`: Integer lag length $p$.
- `sample_size`: Number of effective observations $T$.
- `deterministic`: Included deterministic terms (`const`, `trend`, `both`, `none`).

### 7.2 Core tabular results

Tabular outputs are standardized into long-format DataFrames:

- **Coefficient DataFrame**: Columns `[equation, variable, lag, coefficient, std_error, t_stat, p_value]`.
- **IRF DataFrame**: Columns `[response, shock, horizon, mean, lower, upper]`.
- **FEVD DataFrame**: Columns `[response, shock, horizon, variance_share]`.
- **Historical Decomposition DataFrame**: Columns `[date, variable, shock, contribution]`.

### 7.3 Internal Data Structures

- Matrices are stored internally as contiguous NumPy arrays with explicit shape checks.
- Dimension labels are tracked via Python dictionaries mapping index positions to variable names.

### 7.4 Invariants

- Structural variance decompositions across all shocks must sum to 1.0 (or 100%) at every horizon for each response variable.
- Structural shock covariance matrices must equal the identity matrix $I_K$ under standard unit-variance normalization.

---

## 8. Model extensions

### 8.1 Linear frequentist VAR parity

- Implemented reduced-form VAR estimation via OLS and SUR.
- Lag selection criteria ($AIC$, $HQ$, $SC/BIC$, $FPE$).
- Restricted VAR coefficient masking matrices.
- Analytic point forecasting and forecast error variance bounds.
- Comprehensive residual diagnostics: Portmanteau autocorrelation, Jarque-Bera multivariate normality, ARCH-LM heteroskedasticity.
- Companion matrix roots and stability checks.

### 8.2 Structural restriction engine

- Short-run linear restrictions ($A u_t = B \epsilon_t$).
- Long-run cumulative restrictions via Blanchard-Quah decomposition ($C(1) = A(1)^{-1} B$).
- Structural VECM (SVEC) combining cointegration rank constraints with short/long-run restrictions.
- Sign and Zero restriction engine using QR random orthogonal rotation algorithms.
- Narrative restrictions placing sign/magnitude bounds on structural shocks during specific historical dates.
- Wild bootstrap, residual bootstrap, and moving-block bootstrap inference engines.

### 8.3 Data-driven structural identification

- Identification via Changes in Volatility across discrete regime breaks (Rigobon 2003).
- Identification via Distance Covariance and Independent Component Analysis (Matteson & Tsay 2017).
- Identification via Cramér-von Mises (CVM) distance minimization.
- Identification via Non-Gaussian Maximum Likelihood estimation.
- Column permutation alignment and sign consistency algorithms across bootstrap draws.

### 8.4 Nonlinear multivariate models

- Two-regime Threshold VAR (TVAR) with grid-search optimization for threshold variable and delay parameter.
- Threshold VECM (TVECM) combining cointegration relations with threshold regime dynamics.
- Smooth Transition VAR (STVAR) with logistic or exponential transition functions.
- Generalized Impulse Response Functions (GIRF) computed via Monte Carlo history-dependent simulation.

### 8.5 Counterfactual analysis

- Historical decomposition of historical series into structural shock contributions.
- Counterfactual conditional forecasting (evaluating system trajectories under hypothetical shock paths or suppressed shocks).

---

## 9. Figures to reproduce

The visualization suite must reproduce standard empirical figures found in macroeconomic literature:

- Multiple-panel Impulse Response Function (IRF) plots with shaded confidence bands.
- Forecast Error Variance Decomposition (FEVD) stacked area and bar charts.
- Historical Decomposition (HD) stacked bar charts showing historical variable drivers.
- Threshold transition profile and regime classification timeline figures.
- Generalized IRF comparisons across initial historical states.

---

## 10. Tables to reproduce

The reporting suite must export publication-ready LaTeX tables:

- VAR system coefficient summary tables with standard errors and t-statistics.
- Model selection and lag criteria comparison tables.
- Johansen cointegration test summary tables (Trace and Maximum Eigenvalue statistics).
- Residual diagnostic test summary tables (Portmanteau, Jarque-Bera, ARCH-LM).
- Structural identification restriction setup and acceptance summary tables.

---

## 11. Validation datasets and comparators

### 11.1 Proposed benchmark matrix

- **Canada Macro Dataset**: Standard benchmark for VAR and SVAR estimation (used in R `vars`).
- **US Macro Dataset (FRED-QD)**: Standard benchmark for monetary policy SVARs and sign restrictions (Kilian & Lütkepohl 2017).
- **German Money Market Dataset**: Benchmark for VECM and SVEC estimation.

### 11.2 Comparator interface

- Test suites execute comparison scripts against stored golden JSON fixtures containing reference outputs from R and MATLAB.
- Differences are measured using explicit relative numerical tolerance thresholds.

### 11.3 Optional software and containerization

- R package benchmarks can optionally be re-derived via subprocess calls to local `Rscript` installations when available.
- Core CI test suites rely strictly on frozen golden JSON fixtures to avoid external environment dependencies.

### 11.4 R validation runner: published vignette numbers as golden references

- Tier 1 validation: Fast, offline Python unit tests matching frozen golden constants.
- Tier 2 validation: Subprocess `Rscript` execution validating exact numerical alignment against published vignette outputs.

---

## 12. Implementation phases

### Phase 0: stabilize the reporting foundation

- Finalize normalized DataFrame schemas for IRF, FEVD, HD, and coefficients.
- Standardize JSON output contracts across all existing model classes.

### Phase 1: complete the linear frequentist workflow

- Ensure full parity for reduced-form VAR, VECM, lag selection, and residual diagnostics.
- Implement restricted VAR masking matrices and analytic forecasting bounds.

### Phase 2: implement the structural restriction engine

- Solidify short-run, long-run, sign/zero, narrative, and SVEC restriction modules.
- Complete wild and residual bootstrap confidence interval generators.

### Phase 3: add data-driven SVAR identification

- Solidify volatility break and distance covariance estimators.
- Implement permutation and sign alignment algorithms across bootstrap iterations.

### Phase 4: add nonlinear frequentist models

- Implement TVAR, TVECM, and STVAR estimation classes.
- Implement Monte Carlo GIRF simulation engine.

### Phase 5: release hardening

- Verify documentation, tutorial notebooks, and LaTeX table export tools.
- Complete full test suite execution and validation benchmarks.

---

## 13. Testing and acceptance criteria

### 13.1 Test layers

- **Unit tests**: Individual matrix operations, lag construction, transformation functions.
- **Integration tests**: End-to-end pipeline execution from YAML configuration to report generation.
- **Verification tests**: Numerical parity checks against R `vars`, `tsDyn`, `svars`, `sstvars` golden fixtures.

### 13.2 Numerical tolerance policy

- Reduced-form point estimates and lag selection criteria: Relative tolerance $10^{-5}$.
- Structural impact matrices and IRF point paths: Relative tolerance $10^{-4}$.
- Bootstrap quantile confidence bands and nonlinear simulations: Absolute tolerance $10^{-2}$.

### 13.3 Required failure tests

- Assert clear `ValueError` exceptions for missing variables, invalid lag bounds, non-stationary companion matrices, or non-invertible restriction matrices.

---

## 14. Dependencies and compatibility

### 14.1 Core dependencies

- `numpy`: Numerical array operations.
- `pandas`: Data manipulation and DataFrame result containers.
- `scipy`: Optimization, linear algebra decompositions, and statistical distributions.
- `statsmodels`: Baseline time-series and regression reference routines.

### 14.2 Optional dependencies

- `matplotlib`: Core plotting engine.
- `seaborn`: Visualization styling utilities.
- `pyyaml`: YAML configuration parsing.

### 14.3 Performance policy

- In-memory array operations must avoid unnecessary deep copies.
- Reusable loops in bootstrap and rotation sampling should leverage vectorized matrix routines.

---

## 15. Risks and controls

| Risk | Impact | Mitigating control |
| --- | --- | --- |
| Numerical divergence across platforms | Inconsistent test results | Strict relative tolerance checks and frozen golden JSON fixtures. |
| Convergence failure in nonlinear optimization | Model fit failure | Multi-start optimization initializers and explicit optimization diagnostic returns. |
| Bootstrap sample instability | Slow or invalid confidence intervals | Configurable random seed handling and pre-allocated array memory. |

---

## 16. Issue and milestone breakdown

### Milestone A: common reporting

- Finalize normalized result schemas and persistence modules.
- Integrate LaTeX/Overleaf reporting tables.

### Milestone B: linear VAR completeness

- Complete lag selection, restricted VAR, and diagnostic testing suites.
- Benchmark against R `vars` golden fixtures.

### Milestone C: structural restrictions

- Complete sign/zero, narrative, and SVEC restriction engines.
- Add wild and residual bootstrap confidence intervals.

### Milestone D: data-driven identification

- Complete changes-in-volatility and distance covariance identification classes.
- Implement permutation and sign alignment across draws.

### Milestone E: nonlinear dynamics

- Implement TVAR, TVECM, and STVAR estimators.
- Implement GIRF simulation routine and state-dependent figures.

---

## 17. Definition of done

A model family is considered complete when:

- Estimator logic is fully implemented and tested.
- Fitted output produces normalized DataFrame schemas.
- Comprehensive unit, integration, and verification tests pass.
- Example YAML configurations and tutorial scripts run successfully.
- LaTeX table exporters and visualization charts support the model outputs.

---

## 18. Deferred Bayesian scope

- Bayesian VAR (BVAR), Bayesian SVAR, Gibbs sampling, and MCMC posterior inference are intentionally deferred to maintain a lightweight, frequentist-focused codebase.
- Restriction languages and result interfaces are designed to accommodate potential future Bayesian extensions without structural breaking changes.
