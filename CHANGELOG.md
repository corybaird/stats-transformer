# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - Unreleased

### Added
- `ReportExporter` for markdown model cards, JSON run manifests, and markdown figure indexes.
- Reporting exporter tests covering model-card rendering, manifest writing, and figure-index writing.
- Planned-extension documentation for report export helpers and remaining follow-up work.

## [1.1.0] - Unreleased

### Added
- Packaged example dataset registry with `list_examples()`, `describe_example()`, and `load_example()`.
- Tests covering packaged example discovery, metadata, loading, and unknown-dataset errors.
- Planned extension roadmap covering release hardening, diagnostics, time-series utilities, causal inference, macro time-series models, and provenance auditing.
- `GrangerCausalityTester` for single-series and panel Granger causality tests under `stats_transformer.models.timeseries`.
- Tests covering deterministic single-series and panel Granger causality workflows.
- `TimeSeriesFeatureBuilder` for panel-safe lag, lead, and horizon feature construction.
- `ForecastEvaluator` for MAE, MSE, RMSE, mean error, MAPE, and observation-count metrics.
- `StationarityDiagnostics` for ADF and KPSS stationarity checks.
- Time-series utility tests covering panel feature construction, forecast metrics, and stationarity diagnostics.
- Planned-extension documentation for implemented time-series utilities and remaining follow-up work.

### Changed
- Updated README quickstart guidance to use installed-package imports and document packaged example loading.
- Reworked the publishing plan into a readiness checklist with release blockers, verification steps, and PyPI smoke-test guidance.
- Refined planned econometric extension docs with proposed APIs, validation requirements, and current-vs-future capability notes.
- Updated planned-extension docs to mark the Granger causality tester as implemented and identify remaining follow-up work.

## [1.0.1] - Unreleased

### Fixed
- Fixed `Pipeline` constructor-driven model selection by reading `model_type` from stored keyword arguments.

## [1.0.0] - 2026-05-12

### Added
- **Modular Visualization Framework**: Introduced a three-level architecture for highly reusable chart components.
- **Standalone Chart Components**: 9 new atomic classes for standard econometric plots (`CoefficientBarChart`, `IRFPlot`, `BinnedScatterPlot`, etc.).
- **Library-Bundled Aesthetics**: Built-in color palettes, significance stars, and matplotlib styles (`.mplstyle`) available within the package.
- **Walkthrough Documentation**: New `07_visualization.ipynb` tutorial demonstrating the abstract component API.
- **Enhanced Test Suite**: Comprehensive unit tests for all new visualization components.

### Changed
- **Refactored Visualizers**: `DataVisualizer` and `RegressionVisualizer` now delegate to the underlying `charts/` module while maintaining backward compatibility.
- Updated `.gitignore` rules for more efficient tracking of academic datasets.


- Initial release of stats-transformer.
- Core `FeatureEngineer` for robust data transformation.
- Core `RegressionModel` for standard and robust OLS.
- Core `Pipeline` for YAML-driven execution.
- Modular visualization tools (`DataVisualizer`, `ModelVisualizer`, `RegressionVisualizer`).
- Comprehensive documentation and examples.
