# Documentation overview

`stats-transformer` joins two parts of empirical work that are often documented separately: feature construction and econometric estimation. The documentation is split into short, purpose-specific pages so a reader can move from the project map to implementation details or validation evidence without searching through a single long guide.

## Contents

1. [Choose a starting point](#1-choose-a-starting-point)
2. [Documentation map](#2-documentation-map)
3. [Library model inventory](#3-library-model-inventory)
4. [Example inventory](#4-example-inventory)
5. [How to read validation claims](#5-how-to-read-validation-claims)
6. [Extension planning](#6-extension-planning)

## 1. Choose a starting point

| If you want to... | Start here |
| --- | --- |
| Understand the configured workflow and its components | [Library architecture](library/architecture.md) |
| Locate code, data, configurations, outputs, and examples | [Repository structure](library/file_structure.md) |
| Run the automated checks or understand their scope | [Testing suite](validation/testing_suite.md) |
| Assess replication claims, paper examples, and MATLAB comparator | [Academic and numerical validation](validation/validation.md) |
| Review the proposed frequentist VAR, SVAR, nonlinear-model, and reporting work | [Frequentist multivariate time-series roadmap](extensions/roadmap.md) |

## 2. Documentation map

| Section | Page | Purpose | When it is useful |
| --- | --- | --- | --- |
| Project entry point | [README](../README.md) | Installation, a quick pipeline example, and project-level links. | First use of the repository. |
| Library | [Architecture](library/architecture.md) | Explains the pipeline, package boundaries, models, features, reporting, and visualizations. | Planning an analysis or extending the library. |
| Library | [File structure](library/file_structure.md) | Maps repository directories to their responsibilities. | Finding the right home for code, data, a configuration, or an output. |
| Library | [Academic citations](library/citations.md) | Catalogs literature, datasets, and benchmark software cited in the library. | Reviewing academic foundations and data provenance. |
| Library | [Data directory guide](library/data.md) | Details the data/ directory structure, dataset catalog, frequencies, and pipeline flow. | Locating raw series, final panel parquets, and example datasets. |
| Validation | [Testing suite](validation/testing_suite.md) | Defines the automated test categories and command. | Checking package behavior before a change. |
| Validation | [Academic & numerical validation](validation/validation.md) | Records direct comparisons, paper transformation examples, and MATLAB cross-language comparator tools. | Reporting results responsibly and verifying numerical parity. |
| Extensions | [Frequentist multivariate time-series roadmap](extensions/roadmap.md) | Defines the proposed model, reporting, visualization, and cross-language validation milestones while explicitly deferring Bayesian estimation. | Planning and reviewing future time-series work. |
| Notebook | `notebooks/07_chart_components.ipynb` | Demonstrates modular chart components interactively. | Learning visualization usage. |
| Notebook | `notebooks/08_structural_timeseries_models.ipynb` | Demonstrates direct-use structural VAR, local-projection, and decomposition APIs on synthetic data. | Learning the newer time-series models. |

The pages should remain separate. Architecture and file layout are stable reference material, while validation guides describe evidence and prerequisites that evolve independently. This overview is the short table of contents linking them together.

## 3. Library model inventory

Every model listed below can be instantiated directly as a Python class or configured in YAML and dispatched through `Pipeline` using `model.model_type`.

| Family | Model or utility | Primary use | Access |
| --- | --- | --- | --- |
| Regression | OLS (`RegressionModel`) | Linear regression with an intercept or entity dummies. | Pipeline: `ols`; direct |
| Regression | Robust OLS (`RobustOLSModel`) | OLS with HC or HAC covariance estimation. | Pipeline: `robust_ols`; direct |
| Regression | Panel OLS (`PanelRegressionModel`) | Entity and optional time fixed-effects panel regression. | Pipeline: `panel_ols`; direct |
| Regression | IV 2SLS (`IV2SLSModel`) | Instrumental-variables estimation with `linearmodels`. | Pipeline: `iv` (aliases: `iv_2sls`, `2sls`); direct |
| Regression | Panel IV (`PanelIV2SLSModel`) | Panel two-stage least squares with fixed effects. | Pipeline: `panel_iv`; direct |
| Regression | GMM (`GMMModel`) | Generalized Method of Moments (one-step, two-step, iterated, CUE). | Pipeline: `gmm`; direct |
| Regression | DiD (`DiDModel`) | Callaway-Sant'Anna staggered difference-in-differences. | Pipeline: `did`; direct |
| Discrete | Logit (`LogitModel`) | Binary-outcome maximum-likelihood logit model. | Pipeline: `logit`; direct |
| Discrete | Probit (`ProbitModel`) | Binary-outcome maximum-likelihood probit model. | Pipeline: `probit`; direct |
| Unsupervised | PCA (`PCAModel`) | Standardized principal components and explained variance. | Pipeline: `pca`; direct |
| Unsupervised | K-means (`KMeansModel`) | Standardized clustering. | Pipeline: `kmeans`; direct |
| Time series | VAR (`VARModel`) | Reduced-form multivariate dynamics, forecasts, and IRFs. | Pipeline: `var`; direct |
| Time series | VECM (`VECMModel`) | Cointegrated multivariate time series. | Pipeline: `vecm`; direct |
| Time series | SVEC (`SVECModel`) | Structural VECM combining cointegration rank with restrictions. | Pipeline: `svec`; direct |
| Time series | SVAR (`SVARModel`) | Structural VAR under specified short-run restrictions. | Pipeline: `svar`; direct |
| Time series | Blanchard--Quah (`BlanchardQuahModel`) | Long-run structural identification for a VAR. | Pipeline: `blanchard_quah`; direct |
| Time series | Proxy SVAR (`ProxySVARModel`) | External-instrument structural identification. | Pipeline: `proxy_svar`; direct |
| Time series | Sign & Zero SVAR (`SignZeroSVARModel`) | Structural identification via sign, zero, and narrative restrictions. | Pipeline: `sign_restrictions` (alias: `sign_zero`); direct |
| Time series | Volatility SVAR (`VolatilitySVARModel`) | Changes-in-volatility structural identification across break regimes. | Pipeline: `volatility_svar`; direct |
| Time series | Independence SVAR (`IndependenceSVARModel`) | Distance covariance and FastICA structural identification. | Pipeline: `independence_svar`; direct |
| Time series | Cramér-von Mises SVAR (`CVMSVARModel`) | Distance metric testing mutual independence across rotations. | Pipeline: `cvm_svar` (alias: `cvm`); direct |
| Time series | Non-Gaussian SVAR (`NonGaussianSVARModel`) | Maximum likelihood structural identification with Student-t shocks. | Pipeline: `non_gaussian_svar` (alias: `non_gaussian`); direct |
| Time series | Dynamic Factor (`DynamicFactorModel`) | EM-estimated dynamic factor model with Kalman filtering. | Pipeline: `dynamic_factor`; direct |
| Time series | Bayesian VAR (`BVARModel`) | Conjugate Normal-Inverse-Wishart prior analytical BVAR. | Pipeline: `bvar`; direct |
| Time series | Threshold VAR (`TVARModel`) | Two-regime threshold vector autoregression. | Pipeline: `tvar`; direct |
| Time series | Threshold VECM (`TVECMModel`) | Threshold cointegrated vector error correction. | Pipeline: `tvecm`; direct |
| Time series | Smooth Transition VAR (`STVARModel`) | Smooth transition regime-switching VAR with GIRFs. | Pipeline: `stvar`; direct |
| Time series | Local projections (`LocalProjectionsModel`) | Horizon-by-horizon impulse-response estimation. | Pipeline: `local_projections`; direct |
| Time series | LP-IV (`LocalProjectionsIVModel`) | Instrumented local projections. | Pipeline: `lp_iv`; direct |
| Time series | ARIMA (`ARIMAModel`) | Univariate autoregressive integrated moving average model. | Pipeline: `arima`; direct |
| Time series | Decompositions (`TimeSeriesDecompositions`) | Historical, forecast-error, and related decompositions. | Direct utility |
| Time series | Diagnostics and helpers | Stationarity checks, Granger tests, time-series features, and forecast evaluation. | Direct utilities |

Feature engineering is a coequal part of the library rather than a preprocessing afterthought. `FeatureEngineer` supports log levels, changes, lags, leads, rolling means, z-scores, forward log differences, and frequency resampling; `DataMerger` joins explicitly keyed data sources before estimation. See [Architecture](library/architecture.md) for the workflow contract.

For a runnable introduction to the direct-use structural time-series APIs, see [structural time-series models](../notebooks/08_structural_timeseries_models.ipynb).

## 4. Example inventory

Examples are executable demonstrations, not all formal replications. The validation page specifies the evidence available for the examples that compare against external code or data.

| Area | Example | What it demonstrates | Validation status |
| --- | --- | --- | --- |
| YAML workflow | `examples_running_from_yaml.py` | Runs selected regression and PCA configurations through `Pipeline`. | Demonstration |
| Featurization | `featurization/demo.py`, `example.py` | Core transformations and resampling. | Demonstrations |
| Featurization | `featurization/fred.py`, `dbnomics.py` | Data-provider-oriented feature workflows. | Demonstrations; external availability may vary |
| Featurization | `featurization/monetary.py` | Monetary-policy transformation comparison. | Source-specific comparison |
| Regression | `regression/longley.py`, `mincer_wage.py`, `okuns_law.py` | OLS, robust OLS, and applied regression workflows. | Demonstrations |
| Regression | `regression/grunfeld.py` | Fixed-effects panel regression. | Demonstration |
| Regression | `regression/mroz_iv.py` | Instrumental variables. | Demonstration |
| Regression | `regression/ghysels_chap1.py`, `ghysels_chap2.py` | Textbook regression applications. | Demonstrations |
| Discrete | `discrete/spector_logit.py` | Binary logit estimation. | Demonstration |
| Time series | `timeseries/macro_var.py` | Direct `statsmodels` versus `VARModel` coefficient and standard-error comparison. | Direct Python comparison |
| Time series | `timeseries/ghysels_chap6.py`, `ghysels_chap7.py` | Textbook VAR and VECM applications. | Demonstrations |
| Time series | `timeseries/kilian_svar.py`, `kilian_vecm.py` | Structural VAR and VECM workflows. | Demonstrations |
| Academic features | `academic/nakamura_steinsson.py` | First-difference comparison with supplied Stata output. | Data-dependent comparison |
| Academic features | `academic/bauer_swanson.py`, `bauer_bernanke_milstein.py` | Paper-inspired transformation logic. | Data-dependent comparisons |
| Academic features | `academic/nakamura_steinsson_pca.py` | Synthetic PCA shock extraction. | Demonstration |
| Academic reporting | `academic/stats_transformer_paper_figures.py` | Generates figures used by the Overleaf draft. | Figure-generation workflow |
| VAR-Toolbox translations | `academic/var/stock_watson_2001.py`, `blanchard_quah_1989.py`, `gertler_karadi_2015.py`, `jorda_taylor_2025.py` | Reduced-form VAR, long-run identification, proxy SVAR, and LP-IV on bundled data. | Execution checks; see validation page |
| Cross-language check | `src/examples/software_benchmarks/matlab_comparator.py` | Blanchard--Quah Python/MATLAB impact-matrix comparison. | Verified numerical comparison |

## 5. How to read validation claims

Use the strongest accurate label available:

1. **Automated test**: controlled package behavior passes.
2. **Execution check**: an example fits and returns its expected structure.
3. **Direct comparison**: a defined object is compared with another Python implementation or supplied output.
4. **Cross-language numerical comparison**: a defined object agrees with an independent implementation under matched settings.

Only the last label supports a claim of numerical parity, and only for the data, settings, and object actually compared. The currently documented MATLAB comparison is the Blanchard--Quah impact matrix, not every VAR-Toolbox translation.

## 6. Extension planning

Future work is separated from documentation of current behavior. The [model extension roadmap](extensions/roadmap.md) and [planned models catalog](extensions/models.md) detail advanced structural identification, non-linear dynamics, and missing domain extensions. Closed-form conjugate Bayesian VAR (`BVARModel`) is implemented, while general MCMC sampling and hierarchical priors are tracked in the extended research roadmap.
