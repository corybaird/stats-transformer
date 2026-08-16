# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
