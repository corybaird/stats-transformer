# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.10.0] - 2026-10-14

### Added

- **Local Projections cumulative IRFs, lag augmentation, and HAC standard errors (`LocalProjectionsModel`).**
  - Added `cumulative=True` (or `"baseline"`, `"diff"`) to estimate cumulative impulse response functions ($y_{t+h} - y_{t-1}$ or $y_{t+h} - y_t$), enabling replication of cumulative macroeconomic responses to structural shocks (e.g., Nakamura & Steinsson 2018b).
  - Added `lags=P` parameter to automatically construct and append lag blocks ($t-1, \dots, t-P$) for the dependent variable, shock variable, and control covariates.
  - Added `cov_type="HAC"` with automatic Newey-West bandwidth $h+1$, alongside `"HC0"`, `"HC1"`, `"HC2"`, `"HC3"`, and `"standard"`.
- **Demean-only / covariance PCA (`PCAModel`).**
  - Added `standardize: bool = True` and `with_mean: bool = True` (with YAML configuration support). Setting `standardize=False` skips unit-variance normalization, essential for yield curve decompositions (Diebold & Li 2006, Vanderwel et al. 2026) and macroeconomic factor models where variance scale reflects economic signal.
- **Out-of-sample prediction and high-speed estimation (`RegressionModel`).**
  - Implemented `.predict(new_data)` accepting DataFrames, Series, and 2D arrays, with automatic constant term handling.
  - Added `fit(..., compute_metrics=True)` flag allowing callers to pass `compute_metrics=False` to bypass heavy diagnostics (F-stat, AIC/BIC, condition numbers) during high-frequency rolling regressions.
  - `Pipeline.predict()` now delegates directly to `self.model.predict()` when supported.
- **Direct multi-step forecasting (`VARForecaster.forecast_direct`).**
  - Added `forecast_direct(data, target_variables, lags, steps, trend)` to estimate direct multi-step horizon autoregressions robust to lag order misspecification.
- **Academic replication datasets and benchmarks.**
  - Added `vanderwel_2026` dataset and replication example (`src/examples/academic/vanderwel_2026.py`), reproducing Table 2 Diebold-Li out-of-sample $R^2$ benchmarks from Vanderwel et al. (2026, IJF).
  - Added `nakamura_steinsson_2018_b` dataset and replication example (`src/examples/academic/nakamura_steinsson_2018b_jorda.py`), reproducing Jordà local projection monetary policy multipliers from Nakamura & Steinsson (2018b, QJE).
  - Added test suite in `tests/test_academic_replications.py`.
- **Top-level model exports in `stats_transformer`.**
  - Directly exported `VARModel`, `SVARModel`, `ProxySVARModel`, `SignZeroSVARModel`, `PCAModel`, `VARForecaster`, `BVARModel`, `DynamicFactorModel`, `TVARModel`, `VARLagSelector`, `LocalProjectionsModel`, and `LocalProjectionsIVModel` from the top-level package namespace.

## [1.9.1] - 2026-10-14

### Fixed

- **Full pipeline run on configs without `data.datasets`.** A full run (`stage=None`, or the CLI without `--stage`) always ran the data stage, which crashed with `IndexError` for any config that uses `data.raw_data_file`, including every config in `references/configs` and the paper's Mincer example. A full run now skips the data stage when no datasets are configured and feeds `raw_data_file` straight into features. The manifest lists only the stages that ran. `run(stage="data")` without datasets raises a clear `ValueError` and creates no run folder.
- README quickstart used a non-existent data file and a `frequency` key the schema does not have. It now uses the bundled Grunfeld data and current keys, checked by a test.
- `docs/validation/benchmarks.md` no longer points to `src/examples/software_benchmarks/matlab_comparator.py`, whose source was removed. It points to the recorded `bq1989_long` benchmark and its test.

### Changed

- Model validation status has one source, `docs/validation/model_status.yaml`. `scripts/generate_model_status.py` renders the benchmarks.md §2.1 table and, with `--tex-out`, the paper's Table 1 rows; a test fails if the docs drift from the YAML. The YAML adds a `PandemicBVARModel` row and uses the paper's labels (`Known issue` for ProxySVAR and DiD, `Unresolved` for TVAR).

## [1.9.0] - 2026-10-14

### Added

- **Per-run pipeline output folders with a provenance manifest.** New YAML key `output.root` (default `data/pipeline`). Each full or single-stage run writes to `<root>/<config_stem>/<YYYYMMDD_HHMMSS>/`: `01_data/merged.parquet` (copy, when `data.merge.output_path` is unset), `02_features/features.parquet` (keeps dtypes), `03_model/model_summary.json`, and `manifest.json` (config copy and sha256, input file sha256s, git SHA when available, package version, stages run, UTC timestamps, written files). Explicit output paths are still written exactly as configured, and single-stage runs chain through them as before.
- `VECMModel(coint_rank=...)` and YAML `model.coint_rank` (default 1).
- `SVEC.max_restriction_violation`: the largest violation of the identifying restrictions in the fitted model.

### Changed

- The pipeline no longer writes `reports/model_summary.json` or `reports/transformed_data_<timestamp>.csv`. Use `model.summary_output_path` or the run folder's `03_model/model_summary.json`. `Pipeline.save_results()` is unchanged for direct callers.
- Results change for the models fixed below (BVAR family, TVECM, SVEC, sign-zero SVARs with zero restrictions); re-run analyses that use them.

### Fixed

- **Minnesota prior scaling.** `BVARModel`, `PandemicBVARModel` and `BayesianSignZeroSVARModel` multiplied the prior variance by sigma_j^2 instead of dividing by it. In the natural-conjugate form the prior for equation i, variable j is now sigma_i^2/sigma_j^2 times the tightness (Kadiyala & Karlsson 1997), so the posterior is equivariant to rescaling a variable. One shared `minnesota_prior_moments` helper replaces three identical copies.
- **VECM cointegration rank.** The rank was always 1 because `coint_rank` was never passed to statsmodels. The UK term structure rank-3 case now matches R `urca::ca.jo` (beta and alpha, 1e-8).
- **SVEC estimation.** The unconstrained Nelder-Mead fallback, which ignored the identifying restrictions, is gone, and the constrained solver no longer stops at a local optimum. A just-identified restriction set is solved directly from the ML system (restrictions and BB' = Sigma_u); an over-identified set maximises the concentrated likelihood in the free directions from several deterministic starts. If no start satisfies the restrictions to 1e-6, the fit raises `RuntimeError`. The Canada example now matches R `vars::SVEC` on raw SR and LR (1e-6).
- **Cross-sectional data and single series without an entity column.** `FeatureEngineer` no longer requires unconfigured entity or date columns; transformations run on the whole frame and resampling works without an entity. `country_list` without an entity and `start_date` without a date column raise clear errors. The paper's Mincer wage example (`references/configs/regression/mincer_wage.yaml`) now runs end to end.
- **Sign-zero SVAR zero restrictions.** `SignZeroSVARModel` and `BayesianSignZeroSVARModel` impose zero restrictions by construction (Arias, Rubio-Ramírez & Waggoner 2018, Algorithm 2) instead of accepting random rotations within 1e-10 of zero, which almost never happened. Without zero restrictions, draws are unchanged for a fixed seed. Bayesian draws with zero restrictions are from the ARW proposal; importance weights are not yet applied.
- **TVECM error-correction term.** The regressor is now ECT_{t-1} and the threshold variable ECT_{t-d} (Hansen & Seo 2002). The previous code used ECT_{t-1-d} for both, so results change for every delay, including d = 1.
- **TVAR / TVECM empty regimes.** A threshold that leaves a regime empty or below the minimum size raises `ValueError` instead of silently estimating both regimes on the full sample.
- Tests no longer write into the repository's `reports/` folder.

## [1.8.0] - 2026-10-14

### Changed

- **Four research stages: `data`, `features`, `model`, `reporting`.** README, docs, notebook, and code now use one vocabulary: data (ingest, resample, merge) → features (transform) → model (estimate, diagnostics, JSON summary) → reporting (charts, tables). The canonical names are exported as `stats_transformer.pipeline.STAGES`.
- `Pipeline.run()` is split into one method per stage: `run_data()`, `run_features()`, `run_model()`, `run_reporting()`, plus `run_eda()`. The methods can be called directly without going through `run()`. A full run (`stage=None`) calls them in order instead of duplicating the resample and merge block; return values and output files are unchanged.
- CLI `--stage` validates its value and lists the canonical stages in `--help`.
- `docs/library/architecture.md`, the agent skill, and `notebooks/0-overall-pipeline.ipynb` use the new stage names.

### Deprecated

These keep working in 1.8.0 but emit a `DeprecationWarning` and will be removed in a future release:

- `pipeline.run(stage="resample")`; use `"data"`.
- `pipeline.run(stage="regression")`; use `"model"`.

## [1.7.0] - 2026-10-13

### Changed

- **Package rename: `visualization` → `reporting`.** `src/stats_transformer/visualization/` is merged with the former `reporting/` submodule (`exporters.py`, `timeseries/` reporter and adapters) into a single `src/stats_transformer/reporting/` package covering charts, tables, exports, formatters, and defaults. Import from `stats_transformer.reporting.*`.
- Pipeline stage `reporting` replaces `visualization` (`pipeline.run(stage="reporting")`), and the YAML config key `reporting:` replaces `visualization:`. `Config.get_reporting_config()` replaces `get_visualization_config()`.
- `ReportExporter` is now importable from the top-level `stats_transformer` package.
- `tests/test_visualization/` is merged into `tests/test_reporting/`; configs, README, notebooks, agent skill, and `docs/library/architecture.md` use the new name.

### Deprecated

These keep working in 1.7.0 but emit a `DeprecationWarning` and will be removed in a future release:

- Imports from `stats_transformer.visualization` and its submodules (e.g. `stats_transformer.visualization.charts.bar`). They resolve to the same module objects under `stats_transformer.reporting`.
- `pipeline.run(stage="visualization")`.
- The `visualization:` YAML config key (used only when `reporting:` is absent or empty).
- `Config.get_visualization_config()`.

### Fixed

- **The reporting stage crashed when no model summary was available.** `Pipeline.create_visualizations` tested a DataFrame for truthiness (`not self.transformed_data`), which raised `ValueError: The truth value of a DataFrame is ambiguous` whenever `model_results` was empty, e.g. running the stage on transformed data before any model had been fit.

## [1.6.0] - 2026-08-16

### ⚠️ Breaking

- **`jupyter` is no longer a hard runtime dependency.** Install `stats-transformer[notebooks]` to run the notebooks under `notebooks/`. If you relied on `jupyter` or `IPython` being pulled in transitively by this package, add it explicitly or use the new extra. `rpy2` similarly lives under `stats-transformer[benchmarks]` (moved in 1.5.1), and a new `stats-transformer[all]` extra installs both. The core install is now substantially smaller and no longer requires a working R toolchain.

### Fixed

Several of these produced wrong numbers silently rather than raising, so results generated with earlier versions may be affected.

- **`VolatilitySVARModel.fit()` was unusable.** It did not declare `regime_column` as a required column, so `ModelBase.load_data` stripped it and `build_model` then raised "A regime column must be specified" on data that did contain one.
- **Narrative restrictions were broken under the most natural column name.** `ModelBase.load_data` hardcoded the string `'date'`, consuming that column into the index; `SignZeroSVARModel._check_narrative` then raised `KeyError: 'date'`. The same bug made the date-sorting step in `BlanchardQuahModel`, `ProxySVARModel` and `SignZeroSVARModel` dead code, so those models silently depended on input row order.
- **The literal string `"dummy"` was written into saved model metadata.** Multivariate models (VAR/VECM/SVAR family) invented a fake `target`/`independent_variables` split to satisfy a single-equation contract in `ModelBase`, and it flowed into `get_model_metadata()` and onto disk. They now report a symmetric `variables` list instead.
- **`coefficients` was silently empty for every VAR.** `get_model_metadata` assumed a single-equation `params` Series; for a system model it is a DataFrame, the extraction raised, and a bare `except Exception` swallowed it. System models now emit a nested `{equation: {term: ...}}` structure.
- **`TimeSeriesDecompositions` did not work with restricted VARs.** `RestrictedVARResults` had no `ma_rep`, which `decompositions.py` and the reporting adapters both call.
- **`RestrictedVARResults.k_trend` was always 0**, even with an intercept, because it tested against fabricated placeholder `exog_names`. Any lag-coefficient slice keyed on `k_trend` would have treated the intercept row as a lag coefficient.
- **Blanchard-Quah produced a wrong long-run impact matrix under `trend="ct"`.** The deterministic-row offset was derived by checking for `'const'`, which undercounts when a trend term is also present; all four elements of `B_0` were wrong in a 2-variable test case.
- **`DataMerger.standardize_entity` never converted country codes.** `ISO2_TO_ISO3` was an empty dict, so `"GB"` stayed `"GB"` rather than becoming `"GBR"` — the bundled country converter was never wired in.
- **Component-plus-residual plots raised on the first regressor and skipped the last.** `create_component_plus_residual_plots` passed positional indices to `sm.graphics.plot_ccpr`, but a fitted model with an intercept carries `const` at index 0. Affected every model with an intercept.
- `SpecificationRunner` reported `n_obs` as `None` for plain regression models, reading a metrics key that only panel models emit.
- `BaseFeatureEngineer` raised `AttributeError` instead of the real error when a config file was missing, because `_load_params` logged before the logger existed.

### Added

- **`ProbitModel`** for binary probit MLE, dispatchable as `model_type: probit`, with an example config at `references/configs/spector_probit.yaml`. Verified against the canonical Spector dataset.
- **Model dispatch registry** (`stats_transformer.models.registry.MODEL_REGISTRY`) replacing two duplicated if/elif chains, with an alias map so config typos such as `iv_2sls` resolve rather than silently falling back to OLS. Six previously-unreachable models became dispatchable: `logit`, `var`, `vecm`, `svar`, `volatility_svar`, `independence_svar`.
- **Config validation.** `Pipeline` now validates a `params_path` config — `model_type` is a known registry key, required fields are present — before building a model, via a revived `Config.validate()`.
- **New public exports**: `ProbitModel`, `LogitModel`, `UnsupervisedModel`, `SVARBootstrap`, `RestrictedVAR`, `RestrictedVARResults`, `VARLagSelector`, `VARForecaster`, `SVEC`. Several were implemented and tested but importable only by full module path.
- `RestrictedVARResults.ma_rep`, sharing one MA-recursion implementation with `VARForecaster`.
- **CI gates**: `ruff` (blocking), a coverage floor, and `mypy` on a growing include list. A nightly R-parity workflow runs the golden-value integration suite against R's `vars`/`svars`/`plm`/`AER`.
- **Doc-vs-code consistency tests** asserting that every model documented as **Verified**/**Implemented** is importable, and that nothing marked *Planned* already exists.
- Type annotations on `ModelBase`, `Pipeline`, `RegressionModel` and `UnsupervisedModel`.
- `tests/conftest.py` with shared fixtures. Test coverage rose from 64% to **80.35%** (78 tests to 274).

### Changed

- `Pipeline.predict()` now raises `NotImplementedError` with a clear message instead of `AttributeError`; no model in the library implements `predict()`.
- The rolling-origin `ForecastEvaluator` in `models/timeseries/analysis/` is renamed `RollingOriginEvaluator`, resolving a name collision with the instance-based `ForecastEvaluator` in `models/timeseries/utilities.py`. Only the latter was ever exported.
- `docs/extensions/models.md` corrects five class names that referred to classes which do not exist (`RestrictedVARModel`, `LagSelection`, `BootstrapInference`, `HeteroskedasticSVARModel`, `AlignmentEngine`) and now marks every model **Implemented** or *Planned*.

### Removed

- `models/timeseries/base_irf.py` (`BaseIRFEstimator`) — imported by nothing, never exported, 0% coverage.
- Stale empty test directories and dead `tests/verification/` stubs that shadowed the real benchmark harness.

## [1.5.1] - 2026-08-14

### Fixed
- `sign_restrictions` model_type dispatched to an undefined class name, raising `NameError`; now dispatches to `SignZeroSVARModel`.
- `from stats_transformer.models import *` raised `AttributeError` due to a stale entry in `__all__`.
- The `mroz_iv.yaml` example config silently ran OLS instead of the declared 2SLS replication; `IV2SLSModel` now reads `endogenous`/`instruments` from `params_path` configs.
- Unknown `model_type` and unknown pipeline `stage` values now raise `ValueError` instead of silently falling back to OLS or a full pipeline run.
- `SVEC` and `StabilityDiagnostics.ols_cusum` previously returned fabricated placeholder results; both now raise `NotImplementedError` until estimation is implemented.
- Fixed a broken import in `visualization/charts/timeseries/structural.py` that made `RestrictionHeatmap` and `SwathePlot` unusable.
- `IndependenceSVARModel` and `SignZeroSVARModel` no longer mutate global numpy RNG state; seed is now configurable per instance.
- `docker-compose.yml`'s `r-benchmarks` service no longer mutates the host's `pyproject.toml`/`uv.lock`.

## [1.5.0] - 2026-07-27

### Added
- **Structural Vector Error Correction Models (SVEC)**: Added `VECMModel` wrapper with Long-Run and Short-Run restrictions support.
- **Data-Driven SVAR Identification**: Exploiting heteroskedasticity and non-Gaussianity for shock identification.
- `ARIMAModel` for univariate ARIMA estimation, scalar fit metrics, tidy forecasts, and reporting-friendly metadata.
- ARIMA tests covering simulated AR(1)-style data, forecast output shape, and fitted parameter metadata.
- Nonlinear model features and tests.

## [1.4.0] - 2026-07-27

### Added
- **Frequentist VAR Engine**: Massive update adding Vector Autoregression capabilities.
- Advanced forecasting, residual diagnostics, and stability checks for VAR models.
- Support for restricted VARs and lag selection criteria.

## [1.3.1] - 2026-07-27

### Added
- Documented the optional `stats-transformer-architecture` agent skill.
- Minor cleanups and R integration testing wrappers.

## [1.3.0] - 2026-07-24

### Added
- Comprehensive macro time series and SVAR replication suite.
- Five new time-series models: `BlanchardQuahModel`, `ProxySVARModel`, `SignRestrictionsSVARModel`, `LocalProjectionsIVModel`, and `TimeSeriesDecompositions`.
- Four empirical paper replications: Stock and Watson (2001), Blanchard and Quah (1989), Gertler and Karadi (2015), and Jorda and Taylor (2025).
- Integrated MATLAB Engine cross-verification comparator (`tests/verification/matlab_comparator.py`) against the canonical VAR-Toolbox.
- Added optional `matlabengine` dependency for testing.
- `ReportExporter` for markdown model cards, JSON run manifests, and markdown figure indexes.

### Changed
- Reorganized `src/examples` structure: moved replications to `src/examples/academic/var`, and configuration YAML files to `references/configs`.
- Updated `svar.py` to auto-construct missing $A$ and $B$ constraint masks.

### Fixed
- Fixed Pandas object dtype initialization crash inside `BlanchardQuahModel`.

## [1.2.0] - 2026-06-16

### Added
- `PanelBuilder` for panel data construction and validation workflows.
- `PanelDiagnostics` helpers for regression diagnostics on panel model outputs.
- `SpecRunner` for running reusable regression specifications.
- `BaseIRF` foundation for impulse-response style time-series models.
- `TableGenerator` and descriptive-statistics table helpers under `stats_transformer.visualization.tables`.
- Updated heatmap behavior and bundled matplotlib styles for bar-chart and time-series visualizations.
- Packaged example dataset registry with `list_examples()`, `describe_example()`, and `load_example()`.
- `GrangerCausalityTester` for single-series and panel Granger causality tests under `stats_transformer.models.timeseries`.
- `TimeSeriesFeatureBuilder` for panel-safe lag, lead, and horizon feature construction.
- `ForecastEvaluator` for MAE, MSE, RMSE, mean error, MAPE, and observation-count metrics.
- `StationarityDiagnostics` for ADF and KPSS stationarity checks.

### Changed
- Updated README quickstart guidance to use installed-package imports and document packaged example loading.

## [1.0.1] - 2026-05-26

### Fixed
- Fixed `Pipeline` constructor-driven model selection by reading `model_type` from stored keyword arguments.

## [1.0.0] - 2026-05-12

### Added
- **Modular Visualization Framework**: Introduced a three-level architecture for highly reusable chart components.
- **Standalone Chart Components**: 9 new atomic classes for standard econometric plots (`CoefficientBarChart`, `IRFPlot`, `BinnedScatterPlot`, etc.).
- **Library-Bundled Aesthetics**: Built-in color palettes, significance stars, and matplotlib styles (`.mplstyle`) available within the package.
- **Walkthrough Documentation**: New `07_visualization.ipynb` tutorial demonstrating the abstract component API.
- **Enhanced Test Suite**: Comprehensive unit tests for all new visualization components.
- Initial release of stats-transformer.
- Core `FeatureEngineer` for robust data transformation.
- Core `RegressionModel` for standard and robust OLS.
- Core `Pipeline` for YAML-driven execution.
- Modular visualization tools (`DataVisualizer`, `ModelVisualizer`, `RegressionVisualizer`).

### Changed
- **Refactored Visualizers**: `DataVisualizer` and `RegressionVisualizer` now delegate to the underlying `charts/` module while maintaining backward compatibility.
- Updated `.gitignore` rules for more efficient tracking of academic datasets.
