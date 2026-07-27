# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
