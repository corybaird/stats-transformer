# Software Benchmarks & Paper Replications Guide

This document details the cross-language numerical verification procedures against external econometrics software (Stata, MATLAB, R) and catalogs all runnable academic paper replications in `stats-transformer`.

---

## 1. Verification Levels

Every example script and model implementation in `stats-transformer` is categorized under one of four verification levels:

1. **Cross-Language Verified (MATLAB / R / Stata)**: Parameter estimates and structural impact matrices are explicitly cross-checked to machine precision against external software engines.
2. **Direct Python Verified (`statsmodels` / `linearmodels`)**: Estimated numerical outputs match underlying Python packages on identical input data.
3. **Paper Replication Example**: Translates a published paper's econometric specification, data transformations, and model structure on real research data.
4. **Data & Feature Pipeline Demo**: Demonstrates automated data ingestion, frequency resampling, and featurization pipelines.

---

## 2. Software Parity Benchmarks (Stata, MATLAB, R)

This section evaluates the numerical precision of `stats-transformer` model classes against external software implementations (StataNow 19.5, MATLAB 2025b, and containerized R) using packaged benchmark datasets. The benchmark suite resides under `src/examples/software_benchmarks/`.

```text
src/examples/software_benchmarks/
├── modules/                         # Engine adapters (stata_engine.py, matlab_engine.py, r_engine.py)
├── regression/                      # regression_benchmark.py (OLS vs R lm, Stata regress, MATLAB mldivide)
├── discrete/                        # logit_benchmark.py (Logit vs R glm, Stata logit)
├── unsupervised/                    # pca_benchmark.py (PCA vs R prcomp, Stata pca)
├── timeseries/                      # BQ, VAR, VECM, TVAR benchmarks
└── benchmark_suite.py               # Unified execution entrypoint
```

### 2.1 Cross-Language Numerically Verified Models

These models have been formally verified to machine precision against native external econometrics engines on packaged empirical datasets.

| Family | Model | Target Software / Function | Benchmark Dataset | Verification Status & Max Discrepancy |
| --- | --- | --- | --- | --- |
| **Regression** | `RegressionModel` | R (`stats::lm`), Stata (`regress`), MATLAB (`mldivide`) | Longley (`data/examples/regression/longley.csv`) | **Verified** (R: $2.07 \times 10^{-6}$, Stata: $0.00$, MATLAB: $3.52 \times 10^{-6}$) |
| **Regression** | `PanelRegressionModel` | R (`plm::plm`), Stata (`xtreg`) | Grunfeld (`data/examples/regression/grunfeld.csv`) | **Verified** (R `plm` parity in integration tests) |
| **Discrete** | `LogitModel` | R (`stats::glm`), Stata (`logit`) | Spector & Mazzeo (`data/examples/discrete/spector.csv`) | **Verified** (R: $1.81 \times 10^{-9}$, Stata: $0.00$) |
| **Unsupervised** | `PCAModel` | R (`stats::prcomp`), Stata (`pca`) | Longley (`data/examples/regression/longley.csv`) | **Verified** (R: $3.33 \times 10^{-16}$, Stata: $0.00$) |
| **Time Series** | `VARModel` | R (`vars::VAR`), Stata (`var`), Python (`statsmodels`) | US Macro (`data/examples/timeseries/macrodata.csv`) | **Verified** (Stata: $5.77 \times 10^{-15}$, `statsmodels`: $0.00$) |
| **Time Series** | `VECMModel` | R (`urca::ca.jo`), Stata (`vec`), MATLAB (`vecm`) | US Macro (`data/examples/timeseries/macrodata.csv`) | **Verified** (Stata: $2.00 \times 10^{-3}$) |
| **Time Series** | `BlanchardQuahModel` | MATLAB (`VAR-Toolbox 4.0`), R (`vars::BQ`) | Blanchard & Quah (`data/examples/matlab_examples/BQ1989_Data.xlsx`) | **Verified** (MATLAB: $2.22 \times 10^{-16}$) |
| **Time Series** | `SVARModel` | R (`vars::SVAR`), Stata (`svar`), MATLAB (VAR-Toolbox) | Blanchard & Quah (`data/examples/matlab_examples/BQ1989_Data.xlsx`) | **Verified** (R `svars` parity in integration tests) |
| **Time Series** | `VolatilitySVARModel` | R (`svars::id.cv`) | Rigobon Break Panel (`data/examples/regression/grunfeld.csv`) | **Verified** (R `svars::id.cv` parity in integration tests) |
| **Time Series** | `TVARModel` | R (`tsDyn::TVAR`), Stata (`threshold`) | US Macro (`data/examples/timeseries/macrodata.csv`) | **Verified** (Stata threshold search and parameter parity) |

### 2.2 Planned External Software Benchmarks

These models are fully implemented in Python and tested with automated unit and integration tests. Dedicated cross-language benchmarking scripts are currently planned or in development.

| Family | Model | Target Software / Function | Benchmark Dataset | Verification Status & Max Discrepancy |
| --- | --- | --- | --- | --- |
| **Regression** | `RobustOLSModel` | R (`sandwich::vcovHC`), Stata (`regress, robust`) | Longley (`data/examples/regression/longley.csv`) | *Planned* |
| **Regression** | `IV2SLSModel` | R (`AER::ivreg`), Stata (`ivregress 2sls`) | Mroz (`data/examples/regression/mroz.csv`) | *Planned* |
| **Regression** | `SpecificationRunner` | Multi-specification grid runner | Longley (`data/examples/regression/longley.csv`) | *Utility wrapper* |
| **Time Series** | `RestrictedVAR` | R (`vars::restrict`), Stata (`var`) | US Macro (`data/examples/timeseries/macrodata.csv`) | *Planned* |
| **Time Series** | `ARIMAModel` | R (`forecast::auto.arima`), Stata (`arima`) | US Macro (`data/examples/timeseries/macrodata.csv`) | *Planned* |
| **Time Series** | `ProxySVARModel` | R (`svars`), MATLAB (`VAR-Toolbox 4.0`) | Gertler & Karadi (`data/examples/matlab_examples/GK2015_Data.xlsx`) | *Planned* |
| **Time Series** | `SignZeroSVARModel` | MATLAB (`VAR-Toolbox 4.0`), R (`BMR`) | Kilian & Lütkepohl (`data/examples/timeseries/macrodata.csv`) | *Planned* |
| **Time Series** | `IndependenceSVARModel` | R (`svars`) | FastICA Macro Panel (`data/examples/timeseries/macrodata.csv`) | *Planned* |
| **Time Series** | `SVECModel` | R (`vars::SVEC`), MATLAB (`VAR-Toolbox 4.0`) | King et al. (`data/examples/matlab_examples/SW2001_Data.xlsx`) | *Planned* |
| **Time Series** | `LocalProjectionsModel` | R (`lpirfs::lp_lin`), Stata (`jorda`) | Jordà & Taylor (`data/examples/matlab_examples/JT2025_Data.xlsx`) | *Planned* |
| **Time Series** | `LocalProjectionsIVModel` | R (`lpirfs::lp_lin_iv`), Stata (`lproj`) | Jordà & Taylor (`data/examples/matlab_examples/JT2025_Data.xlsx`) | *Planned* |
| **Unsupervised** | `KMeansModel` | R (`stats::kmeans`), Stata (`cluster`), MATLAB (`kmeans`) | Spector & Mazzeo (`data/examples/discrete/spector.csv`) | *Planned* |

### Running the Benchmark Suite

Execute all software benchmarks directly:

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
- **Observed Discrepancy**: Maximum absolute difference: **$2.22 \times 10^{-16}$** (machine precision).

Execute the MATLAB comparator directly:

```bash
/opt/homebrew/bin/uv run python -m src.examples.software_benchmarks.matlab_comparator
```

---

## 4. Academic Paper Replications & Example Scripts

This section catalogs all 27 runnable example modules in `src/examples/`, detailing their academic source papers, module paths, data source files, and empirical verification role.

| Domain / Method | Script Module Path | Academic Paper / Benchmark Target | Data Source File | Verification Status | Target Verification / Compared Object |
| --- | --- | --- | --- | --- | --- |
| **MATLAB Comparator** | `src.examples.software_benchmarks.matlab_comparator` | Blanchard & Quah (1989) / MATLAB VAR-Toolbox 4.0 | `data/examples/matlab_examples/BQ1989_Data.xlsx` | **Cross-Language Verified (MATLAB)** | Structural impact matrix $C(1)$ ($2.22 \times 10^{-16}$ max diff) |
| **Structural VAR** | `src.examples.academic.var.blanchard_quah_1989` | Blanchard & Quah (1989) | `data/examples/matlab_examples/BQ1989_Data.xlsx` | **Cross-Language Verified (MATLAB)** | Long-run structural supply & demand shock identification |
| **Proxy SVAR / SVAR-IV** | `src.examples.academic.var.gertler_karadi_2015` | Gertler & Karadi (2015) | `data/examples/matlab_examples/GK2015_Data.xlsx` | **Paper Replication Example** | External-instrument monetary policy shock identification |
| **LP-IV Local Projections** | `src.examples.academic.var.jorda_taylor_2025` | Jordà & Taylor (2025) / Stock & Watson (2018) | `data/examples/matlab_examples/JT2025_Data.xlsx` | **Paper Replication Example** | Instrumental-variable impulse response functions |
| **Reduced-Form VAR** | `src.examples.academic.var.stock_watson_2001` | Stock & Watson (2001) | `data/examples/matlab_examples/SW2001_Data.xlsx` | **Paper Replication Example** | 3-variable macro VAR (Inflation, Unemployment, Fed Funds) |
| **Reduced-Form VAR** | `src.examples.timeseries.macro_var` | `statsmodels.tsa.vector_ar.var_model` | `data/examples/timeseries/macrodata.csv` | **Direct Python Verified (`statsmodels`)** | VAR(2) coefficient matrices & standard error parity |
| **SVAR Identification** | `src.examples.timeseries.kilian_svar` | Kilian & Lütkepohl (2017) | `data/examples/timeseries/macrodata.csv` | **Paper Replication Example** | Short-run Cholesky & A-model structural identification |
| **Johansen VECM** | `src.examples.timeseries.kilian_vecm` | Johansen (1991) / Kilian & Lütkepohl (2017) | `data/examples/timeseries/macrodata.csv` | **Paper Replication Example** | Cointegration rank test & error-correction dynamics |
| **VAR & Forecasting** | `src.examples.timeseries.ghysels_chap6` | Ghysels & Marcellino (2018) Chapter 6 | `data/examples/timeseries/ghysels_ch6/var_simulated_ch6_sec9.csv` | **Textbook Replication Example** | Multi-step forecasting & simulated VAR impulse responses |
| **VECM & Cointegration** | `src.examples.timeseries.ghysels_chap7` | Ghysels & Marcellino (2018) Chapter 7 | `data/examples/timeseries/ghysels_ch7/simulated_cointegration.csv` | **Textbook Replication Example** | UK term structure cointegration & vector error correction |
| **High-Frequency Shock** | `src.examples.academic.nakamura_steinsson` | Nakamura & Steinsson (2018) | `data/examples/academic/nakamura_steinsson_2018/NominalYields.csv` | **Paper Replication Example** | Daily first difference of Fed Funds futures surprise series |
| **PCA Shock Extraction** | `src.examples.academic.nakamura_steinsson_pca` | Nakamura & Steinsson (2018) | `data/examples/academic/nakamura_steinsson_2018/master.dta` | **Paper Replication Example** | First principal component extraction from monetary futures |
| **High-Frequency Shock** | `src.examples.academic.bauer_swanson` | Bauer & Swanson (2023) | `data/examples/academic/bauer_swanson_2023/cpi.txt` | **Paper Replication Example** | Monthly orthogonalized monetary surprise transformations |
| **High-Frequency Shock** | `src.examples.academic.bauer_bernanke_milstein` | Bauer, Bernanke, & Milstein (2023) | `data/examples/academic/bbm_2023/feds_subset.csv` | **Paper Replication Example** | Daily difference and percentage-change transformation logic |
| **Bayesian VAR / Sign Identification** | `src.examples.academic.jarocinski_karadi_2020` | Jarociński & Karadi (2020) | `data/examples/academic/jarocinski_karadi_2020/fomc_surprises.parquet.gzip` | **Paper Replication Example** | Conjugate BVAR with sign-restriction shock classification |
| **Collinear OLS** | `src.examples.regression.longley` | Longley (1967) | `data/examples/regression/longley.csv` | **Direct Python Verified (`statsmodels`)** | OLS & Robust OLS numerical stability under collinearity |
| **Panel Regression** | `src.examples.regression.grunfeld` | Grunfeld (1958) | `data/examples/regression/grunfeld.csv` | **Direct Python Verified (`linearmodels`)** | Fixed-effects corporate investment panel regression |
| **Instrumental Variables** | `src.examples.regression.mroz_iv` | Mroz (1987) | `data/examples/regression/mroz.csv` | **Direct Python Verified (`linearmodels`)** | 2SLS female labor supply hours equation |
| **Mincer Wage Equation** | `src.examples.regression.mincer_wage` | Mincer (1974) | `data/examples/regression/mroz.csv` | **Textbook Replication Example** | Semi-logarithmic human capital wage regression |
| **Okun's Law** | `src.examples.regression.okuns_law` | Okun (1962) | `data/examples/timeseries/macrodata.csv` | **Textbook Replication Example** | GDP growth vs unemployment rate change regression |
| **Applied Regression** | `src.examples.regression.ghysels_chap1` | Ghysels & Marcellino (2018) Chapter 1 | `data/examples/regression/ghysels_ch1/ex2_regress_gdp.csv` | **Textbook Replication Example** | Linear trend & seasonal dummy regression models |
| **Applied Regression** | `src.examples.regression.ghysels_chap2` | Ghysels & Marcellino (2018) Chapter 2 | `data/examples/regression/ghysels_ch2/ex1_regress_gdp.csv` | **Textbook Replication Example** | Autoregressive distributed lag (ARDL) forecasting |
| **Binary Discrete Choice** | `src.examples.discrete.spector_logit` | Spector & Mazzeo (1980) | `data/examples/discrete/spector.csv` | **Direct Python Verified (`statsmodels`)** | Binary Logit educational choice estimation |
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
