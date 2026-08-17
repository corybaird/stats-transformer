# Academic & Numerical Validation Guide

This document details the academic benchmarks, numerical verification procedures, paper transformation examples, cross-language MATLAB comparator tools, and the cross-language verification roadmap for `stats-transformer`.

## Table of Contents

1. [1. Intuitive Verification Status Levels](#1-intuitive-verification-status-levels)
2. [2. Master Comparison & Example Catalog](#2-master-comparison--example-catalog)
3. [3. Cross-Language MATLAB Comparator](#3-cross-language-matlab-comparator)
   - [3.1 Software Environment & Computational Provenance](#31-software-environment--computational-provenance)
   - [3.2 Verified Numerical Specification & Discrepancy](#32-verified-numerical-specification--discrepancy)
   - [3.3 Execution Protocol & Programmatic Reuse](#33-execution-protocol--programmatic-reuse)
4. [4. Cross-Language Verification Roadmap (R, Stata, & MATLAB)](#4-cross-language-verification-roadmap-r-stata--matlab)
5. [5. Detailed Breakdown by Functional Area](#5-detailed-breakdown-by-functional-area)
   - [5.1 Structural VAR, Reduced-Form, & Local Projections](#51-structural-var-reduced-form--local-projections)
   - [5.2 High-Frequency Monetary & Policy Surprise Replications](#52-high-frequency-monetary--policy-surprise-replications)
   - [5.3 Applied Micro & Macro Econometric Regressions](#53-applied-micro--macro-econometric-regressions)
   - [5.4 Discrete Choice & Classification Models](#54-discrete-choice--classification-models)
   - [5.5 Featurization & External Provider Integration](#55-featurization--external-provider-integration)
6. [6. Protocol for Reporting Validation Claims](#6-protocol-for-reporting-validation-claims)

---

## 1. Intuitive Verification Status Levels

To avoid confusion, every example script and model implementation in `stats-transformer` is categorized by one of four intuitive verification statuses:

1. **Cross-Language Verified (MATLAB / R / Stata)**: The estimated numerical outputs (coefficients, structural impact matrices, or IRFs) are explicitly cross-checked and matched to machine precision against external software routines (e.g. MATLAB VAR-Toolbox 4.0, R `vars`, Stata `svar`).
2. **Direct Python Verified (`statsmodels` / `linearmodels`)**: The estimated numerical outputs are cross-checked and matched against underlying Python packages on identical input data.
3. **Paper Replication Example (Pending R/Stata/MATLAB Check)**: An executable Python script translating an academic paper's econometric specification, data transformations, and model structure. It runs end-to-end on real research data, and is queued for formal cross-language numerical comparison against R, Stata, or MATLAB published code.
4. **Data & Feature Pipeline Demo**: Demonstrates automated data ingestion, frequency resampling, and featurization pipelines without estimating an econometric model.
5. **Illustrative Method Demo (Synthetic Input)**: Demonstrates a model's estimation mechanics on a synthetic dataset with a known ground truth, because the paper's actual input data is not available in this repository. A published reference series may be loaded and displayed alongside for orientation, but it is not used as model input and is not a formal numerical comparison.

> [!NOTE]
> A script labeled as a **Paper Replication Example** is a functional Python implementation of a published paper. It becomes **Cross-Language Verified** once its numerical outputs are formally benchmarked against R, Stata, or MATLAB outputs within a documented tolerance.

---

## 2. Master Comparison & Example Catalog

The table below catalogs all 26 example modules in `src/examples/`, detailing their academic citations, script paths, data sources, intuitive verification statuses, and benchmark targets.

| Domain / Method | Script Module Path | Academic Paper / Benchmark Target | Data Source / Location | Intuitive Verification Status | Target Verification / Compared Object |
| --- | --- | --- | --- | --- | --- |
| **MATLAB Comparator** | `src.examples.software_benchmarks.matlab_comparator` | Blanchard & Quah (1989) / MATLAB VAR-Toolbox 4.0 | `data/examples/matlab_examples/BQ1989_Data.xlsx` | **Cross-Language Verified (MATLAB)** | Structural impact matrix $C(1)$ ($2.22 \times 10^{-16}$ max diff) |
| **Structural VAR** | `src.examples.academic.var.blanchard_quah_1989` | Blanchard & Quah (1989) | `data/examples/matlab_examples/BQ1989_Data.xlsx` | **Cross-Language Verified (MATLAB)** | Long-run structural supply & demand shock identification |
| **Proxy SVAR / SVAR-IV** | `src.examples.academic.var.gertler_karadi_2015` | Gertler & Karadi (2015) | `data/examples/academic/gertler_karadi/` | **Paper Replication Example** (Pending MATLAB/R check) | External-instrument monetary policy shock identification |
| **LP-IV Local Projections** | `src.examples.academic.var.jorda_taylor_2025` | Jordà & Taylor (2025) / Stock & Watson (2018) | `data/examples/academic/` | **Paper Replication Example** (Pending R `lpirfs` check) | Instrumental-variable impulse response functions |
| **Reduced-Form VAR** | `src.examples.academic.var.stock_watson_2001` | Stock & Watson (2001) | `data/examples/academic/stock_watson/` | **Paper Replication Example** (Pending Stata check) | 3-variable macro VAR (Inflation, Unemployment, Fed Funds) |
| **Reduced-Form VAR** | `src.examples.timeseries.macro_var` | `statsmodels.tsa.vector_ar.var_model` | `data/examples/timeseries/macrodata.csv` | **Direct Python Verified (`statsmodels`)** | VAR(2) coefficient matrices & standard error parity |
| **SVAR Identification** | `src.examples.timeseries.kilian_svar` | Kilian & Lütkepohl (2017) | `data/examples/timeseries/` | **Paper Replication Example** (Pending R `svars` check) | Short-run Cholesky & A-model structural identification |
| **Johansen VECM** | `src.examples.timeseries.kilian_vecm` | Johansen (1991) / Kilian & Lütkepohl (2017) | `data/examples/timeseries/` | **Paper Replication Example** (Pending R `urca` check) | Cointegration rank test & error-correction dynamics |
| **VAR & Forecasting** | `src.examples.timeseries.ghysels_chap6` | Ghysels & Marcellino (2018) Chapter 6 | `data/examples/timeseries/ghysels_ch6/` | **Textbook Replication Example** (Pending R check) | Multi-step forecasting & simulated VAR impulse responses |
| **VECM & Cointegration** | `src.examples.timeseries.ghysels_chap7` | Ghysels & Marcellino (2018) Chapter 7 | `data/examples/timeseries/ghysels_ch7/` | **Textbook Replication Example** (Pending R `tsDyn` check) | UK term structure cointegration & vector error correction |
| **High-Frequency Shock** | `src.examples.academic.nakamura_steinsson` | Nakamura & Steinsson (2018) | Supplied Stata `master.dta` | **Paper Replication Example** (Stata parity check) | Daily first difference of Fed Funds futures surprise series |
| **PCA Shock Extraction** | `src.examples.academic.nakamura_steinsson_pca` | Nakamura & Steinsson (2018) | Supplied Stata `master.dta` | **Paper Replication Example** (Pending Stata check) | First principal component extraction from monetary futures |
| **High-Frequency Shock** | `src.examples.academic.bauer_swanson` | Bauer & Swanson (2023) | MATLAB reference dataset | **Paper Replication Example** (MATLAB parity check) | Monthly orthogonalized monetary surprise transformations |
| **High-Frequency Shock** | `src.examples.academic.bauer_bernanke_milstein` | Bauer, Bernanke, & Milstein (2023) | Daily financial series | **Paper Replication Example** (Python reference check) | Daily difference and percentage-change transformation logic |
| **Collinear OLS** | `src.examples.regression.longley` | Longley (1967) | `data/examples/regression/longley.csv` | **Direct Python Verified (`statsmodels`)** | OLS & Robust OLS numerical stability under collinearity |
| **Panel Regression** | `src.examples.regression.grunfeld` | Grunfeld (1958) | `data/examples/regression/grunfeld.csv` | **Direct Python Verified (`linearmodels`)** | Fixed-effects corporate investment panel regression |
| **Instrumental Variables** | `src.examples.regression.mroz_iv` | Mroz (1987) | `data/examples/regression/mroz.csv` | **Direct Python Verified (`linearmodels`)** | 2SLS female labor supply hours equation |
| **Mincer Wage Equation** | `src.examples.regression.mincer_wage` | Mincer (1974) | `data/examples/regression/` | **Textbook Replication Example** | Semi-logarithmic human capital wage regression |
| **Okun's Law** | `src.examples.regression.okuns_law` | Okun (1962) | `data/examples/timeseries/` | **Textbook Replication Example** | GDP growth vs unemployment rate change regression |
| **Applied Regression** | `src.examples.regression.ghysels_chap1` | Ghysels & Marcellino (2018) Chapter 1 | `data/examples/regression/` | **Textbook Replication Example** | Linear trend & seasonal dummy regression models |
| **Applied Regression** | `src.examples.regression.ghysels_chap2` | Ghysels & Marcellino (2018) Chapter 2 | `data/examples/regression/` | **Textbook Replication Example** | Autoregressive distributed lag (ARDL) forecasting |
| **Binary Discrete Choice** | `src.examples.discrete.spector_logit` | Spector & Mazzeo (1980) | `data/examples/discrete/spector.csv` | **Direct Python Verified (`statsmodels`)** | Binary Logit educational choice estimation |
| **High-Frequency SOFR** | `src.examples.academic.acosta_brennan_jacobson_2024` | Acosta, Brennan, & Jacobson (2024) | `data/examples/academic/acosta_brennan_jacobson_2024/` | **Paper Replication Example** | SOFR futures surprise VAR & Robust OLS estimation |
| **Policy Uncertainty** | `src.examples.academic.cieslak_hansen_mcmahon_xiao_2024` | Cieslak, Hansen, McMahon, & Xiao (2024) | `data/examples/academic/cieslak_hansen_mcmahon_xiao_2024/` | **Paper Replication Example** | Policymakers' Uncertainty OLS & VAR estimation |
| **News Sentiment** | `src.examples.academic.shapiro_sudhof_wilson_2022` | Shapiro, Sudhof, & Wilson (2022) | `data/examples/academic/shapiro_sudhof_wilson_2022/` | **Paper Replication Example** | Daily news sentiment featurization & Robust OLS |
| **Industrial Policy** | `src.examples.academic.lane_2025` | Lane (2025) | `data/examples/academic/lane_2025/` | **Paper Replication Example** (DiD cohort is derived from tariff timing, not the paper's treatment definition) | Targeted policy lending Robust OLS regression; illustrative Callaway-Sant'Anna DiD on tariff liberalization |
| **Dynamic Factor Model** | `src.examples.academic.miranda_agrippino_rey_2020` | Miranda-Agrippino & Rey (2020) | `data/examples/academic/miranda_agrippino_rey_2020/` (published factor series only; no asset-price input panel available) | **Illustrative Method Demo (Synthetic Input)** | EM-estimated dynamic factor extraction from a synthetic risky-asset panel |
| **Survey Forecast Rigidity** | `src.examples.academic.coibion_gorodnichenko_2012` | Coibion & Gorodnichenko (2012) | `data/examples/academic/coibion_gorodnichenko_2012/` (Greenbook extract; simplified single-series specification) | **Paper Replication Example** | Two-step GMM forecast-error / revision orthogonality test |
| **Provider Pipeline** | `src.examples.featurization.fred` | St. Louis Fed FRED API | Remote FRED API / Local cache | **Data & Feature Pipeline Demo** | Automated multi-series FRED data download & alignment |
| **Provider Pipeline** | `src.examples.featurization.monetary` | Federal Reserve Macro Series | `data/raw/` | **Data & Feature Pipeline Demo** | Monetary policy indicator featurization pipeline |

---

## 3. Cross-Language MATLAB Comparator

The MATLAB comparator in `src/examples/software_benchmarks/matlab_comparator.py` (and test wrapper `tests/verification/matlab_comparator.py`) provides an opt-in cross-language verification tool against Ambrogio Cesa-Bianchi's MATLAB VAR-Toolbox.

### 3.1 Software Environment & Computational Provenance

| Component | Verified Version | Role in Verification Suite |
| --- | --- | --- |
| **MATLAB** | R2025b Update 4 (v25.2.0) | Local MATLAB Engine execution runtime |
| **VAR-Toolbox** | 4.0 | Benchmark MATLAB implementation of structural VAR methods |
| **Dynare** | 7.1 (Apple Silicon) | Recorded for environment provenance (not called by tests) |

### 3.2 Verified Numerical Specification & Discrepancy

The comparator uses the bundled `data/examples/matlab_examples/BQ1989_Data.xlsx` dataset and estimates:
- **System Variables**: GDP growth ($\Delta y_t$) and unemployment rate ($u_t$).
- **Lag Structure**: VAR(8) with an intercept.
- **Identification Scheme**: Blanchard & Quah (1989) long-run structural restrictions ($C(1)$ lower-triangular).

#### Discrepancy Verification
- **Tolerance Criterion**: Absolute and relative tolerance set to $1.0 \times 10^{-10}$.
- **Observed Discrepancy**: Maximum absolute difference between Python and MATLAB structural impact matrices: **$2.22 \times 10^{-16}$** (machine precision).

### 3.3 Execution Protocol & Programmatic Reuse

Executing the MATLAB comparator requires:
1. Licensed local MATLAB installation.
2. `matlabengine` installed in the active Python virtual environment (`.venv`).
3. Local checkout of [VAR-Toolbox](https://github.com/ambropo/VAR-Toolbox) placed in `data/temp/VAR-Toolbox`.

Execute the comparison directly:

```bash
/opt/homebrew/bin/uv run python -m src.examples.software_benchmarks.matlab_comparator
```

Alternatively, if your VAR-Toolbox is located elsewhere, set the `VAR_TOOLBOX_DIR` environment variable:

```bash
VAR_TOOLBOX_DIR=/path/to/VAR-Toolbox /opt/homebrew/bin/uv run python -m src.examples.software_benchmarks.matlab_comparator
```

Alternatively, invoke `MATLABComparator` programmatically:

```python
from src.examples.software_benchmarks.matlab_comparator import MATLABComparator

result = MATLABComparator("/path/to/VAR-Toolbox").run()
print("Max absolute difference:", result["max_abs_diff"])
```

---

## 4. Complete Cross-Language Model Subsystem Matrix

To expand numerical validation across econometrics software, the following table maps every core `stats-transformer` model to its corresponding benchmark routine in R, Stata, and MATLAB. We are actively building test comparators in `src/examples/software_benchmarks/` to formally verify parity against these targets.

| Family           | Model                     | Target Software / Function                                | Benchmark Script File                                                                                                                                                                                                                                                  | Verification Status & Max Diff                         |
| ---------------- | ------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **Regression** | `RegressionModel` | R (`stats::lm`), Stata (`regress`), MATLAB (`mldivide`) | [`regression_benchmark.py`](../../src/examples/software_benchmarks/regression/regression_benchmark.py) | **Verified** (R: $2.07 \times 10^{-6}$, Stata: $0.00$, MATLAB: $3.52 \times 10^{-6}$) |
| **Regression**   | `RobustOLSModel`          | R (`sandwich::vcovHC`), Stata (`regress, robust`)         | `src/examples/software_benchmarks/`                                                                                                                                                                                                                                    | *Planned*                                              |
| **Regression**   | `PanelRegressionModel`    | R (`plm::plm`), Stata (`xtreg`)                           | `src/examples/software_benchmarks/`                                                                                                                                                                                                                                    | *Planned*                                              |
| **Regression**   | `IV2SLSModel`             | R (`AER::ivreg`), Stata (`ivregress 2sls`)                | `src/examples/software_benchmarks/`                                                                                                                                                                                                                                    | *Planned*                                              |
| **Regression**   | `SpecificationRunner`     | *N/A (Utility wrapper)*                                   | *N/A*                                                                                                                                                                                                                                                                  | *N/A*                                                  |
| **Time Series**  | `VARModel`                | R (`vars::VAR`), Stata (`var`), MATLAB (`varm`)           | `src/examples/software_benchmarks/`                                                                                                                                                                                                                                    | *Planned*                                              |
| **Time Series**  | `VECMModel`               | R (`urca::ca.jo`), Stata (`vec`), MATLAB (`vecm`)         | `src/examples/software_benchmarks/`                                                                                                                                                                                                                                    | *Planned*                                              |
| **Time Series**  | `RestrictedVAR`           | R (`vars::restrict`), Stata (`var`)                       | `src/examples/software_benchmarks/`                                                                                                                                                                                                                                    | *Planned*                                              |
| **Time Series**  | `ARIMAModel`              | R (`forecast::auto.arima`), Stata (`arima`)               | `src/examples/software_benchmarks/`                                                                                                                                                                                                                                    | *Planned*                                              |
| **Time Series**  | `SVARModel`               | R (`vars::SVAR`), Stata (`svar`), MATLAB (VAR-Toolbox)    | `src/examples/software_benchmarks/`                                                                                                                                                                                                                                    | *Planned*                                              |
| **Time Series**  | `BlanchardQuahModel`      | MATLAB (`VAR-Toolbox 4.0`), R (`vars::BQ`)                | [`blanchard_quah_benchmark.py`](../../src/examples/software_benchmarks/timeseries/blanchard_quah_benchmark.py) | **Verified** (MATLAB: $2.22 \times 10^{-16}$)          |
| **Time Series**  | `ProxySVARModel`          | R (`svars`), MATLAB (`VAR-Toolbox 4.0`)                   | `src/examples/software_benchmarks/`                                                                                                                                                                                                                                    | *Planned*                                              |
| **Time Series**  | `SignZeroSVARModel`       | MATLAB (`VAR-Toolbox 4.0`), R (`BMR`)                     | `src/examples/software_benchmarks/`                                                                                                                                                                                                                                    | *Planned*                                              |
| **Time Series**  | `VolatilitySVARModel`     | R (`svars`)                                               | `src/examples/software_benchmarks/`                                                                                                                                                                                                                                    | *Planned*                                              |
| **Time Series**  | `IndependenceSVARModel`   | R (`svars`)                                               | `src/examples/software_benchmarks/`                                                                                                                                                                                                                                    | *Planned*                                              |
| **Time Series**  | `SVEC`                    | R (`vars::SVEC`), MATLAB (`VAR-Toolbox 4.0`)              | `src/examples/software_benchmarks/`                                                                                                                                                                                                                                    | *Planned*                                              |
| **Time Series**  | `LocalProjectionsModel`   | R (`lpirfs::lp_lin`), Stata (`jorda`)                     | `src/examples/software_benchmarks/`                                                                                                                                                                                                                                    | *Planned*                                              |
| **Time Series**  | `LocalProjectionsIVModel` | R (`lpirfs::lp_lin_iv`), Stata (`lproj`)                  | `src/examples/software_benchmarks/`                                                                                                                                                                                                                                    | *Planned*                                              |
| **Discrete**     | `LogitModel`              | R (`stats::glm`), Stata (`logit`)                         | [`logit_benchmark.py`](../../src/examples/software_benchmarks/discrete/logit_benchmark.py) | **Verified** (R: $1.81 \times 10^{-9}$, Stata: $0.00$) |
| **Unsupervised** | `PCAModel`                | R (`stats::prcomp`), Stata (`pca`)        | [`pca_benchmark.py`](../../src/examples/software_benchmarks/unsupervised/pca_benchmark.py) | **Verified** (R: $3.33 \times 10^{-16}$, Stata: $0.00$) |
| **Unsupervised** | `KMeansModel`             | R (`stats::kmeans`), Stata (`cluster`), MATLAB (`kmeans`) | `src/examples/software_benchmarks/`                                                                                                                                                                                                                                    | *Planned*                                              |

---

## 5. Detailed Breakdown by Functional Area

### 5.1 Structural VAR, Reduced-Form, & Local Projections

```bash
# Reduced-Form VAR direct parity check against statsmodels
/opt/homebrew/bin/uv run python -m src.examples.timeseries.macro_var

# Stock & Watson 3-variable VAR demonstration
/opt/homebrew/bin/uv run python -m src.examples.academic.var.stock_watson_2001

# Blanchard & Quah long-run structural identification (MATLAB verified)
/opt/homebrew/bin/uv run python -m src.examples.academic.var.blanchard_quah_1989

# Gertler & Karadi Proxy SVAR / SVAR-IV monetary policy shocks
/opt/homebrew/bin/uv run python -m src.examples.academic.var.gertler_karadi_2015

# Jordà & Taylor instrumental-variable local projections (LP-IV)
/opt/homebrew/bin/uv run python -m src.examples.academic.var.jorda_taylor_2025
```

### 5.2 High-Frequency Monetary & Policy Surprise Replications

```bash
# Nakamura & Steinsson (2018) daily monetary policy surprises
/opt/homebrew/bin/uv run python -m src.examples.academic.nakamura_steinsson

# PCA monetary shock extraction from futures
/opt/homebrew/bin/uv run python -m src.examples.academic.nakamura_steinsson_pca

# Bauer & Swanson (2023) orthogonalized monetary shocks
/opt/homebrew/bin/uv run python -m src.examples.academic.bauer_swanson

# Bauer, Bernanke, & Milstein (2023) daily transformation routines
/opt/homebrew/bin/uv run python -m src.examples.academic.bauer_bernanke_milstein
```

### 5.3 Applied Micro & Macro Econometric Regressions

```bash
# Longley (1967) OLS & Robust OLS numerical benchmark
/opt/homebrew/bin/uv run python -m src.examples.regression.longley

# Grunfeld (1958) fixed-effects corporate investment panel
/opt/homebrew/bin/uv run python -m src.examples.regression.grunfeld

# Mroz (1987) 2SLS instrumental variables female labor supply
/opt/homebrew/bin/uv run python -m src.examples.regression.mroz_iv

# Mincer (1974) human capital wage regression
/opt/homebrew/bin/uv run python -m src.examples.regression.mincer_wage
```

### 5.4 Discrete Choice & Classification Models

```bash
# Spector & Mazzeo (1980) binary Logit educational choice model
/opt/homebrew/bin/uv run python -m src.examples.discrete.spector_logit
```

### 5.5 Featurization & External Provider Integration

```bash
# Automated FRED dataset download & frequency alignment
/opt/homebrew/bin/uv run python -m src.examples.featurization.fred

# DBnomics multi-country provider pipeline
/opt/homebrew/bin/uv run python -m src.examples.featurization.dbnomics
```

---

## 6. Protocol for Reporting Validation Claims

When publishing or reporting empirical validation results, always document:
1. Exact dataset name, version, and date of download.
2. Feature transformations and sample adjustment rules.
3. Estimator specification (lags, deterministic terms, identification constraints).
4. Compared object (coefficients, standard errors, structural impact matrix, IRF paths).
5. Numerical tolerance and observed maximum absolute discrepancy.
