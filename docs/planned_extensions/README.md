# Planned Extensions

This directory tracks future work that is useful but not required for the current package release. Planned extensions should be specific enough to implement, but honest about what is stable, experimental, or only a research idea.

## Prioritization

### Near Term

- **Release hardening:** installed-package smoke tests, package data audit, CI version alignment, and README examples that work outside the repository.
- **Diagnostics layer:** reusable model diagnostics for stationarity, multicollinearity, residual checks, weak instruments, and panel structure validation.
- **Time-series utilities:** general lag builders, horizon builders, Granger causality tests, and forecast evaluation helpers.
- **Example registry:** a documented way to locate bundled examples, load small teaching datasets, and run known-good model configurations.

Implemented first passes:

- `GrangerCausalityTester` under `stats_transformer.models.timeseries`.
- `TimeSeriesFeatureBuilder`, `ForecastEvaluator`, and `StationarityDiagnostics` under `stats_transformer.models.timeseries`.

### Medium Term

- **Causal inference suite:** IV diagnostics, event-study DiD, RDD helpers, propensity score weighting, and doubly robust estimators.
- **Macroeconomic time-series suite:** ARIMA, BVAR, MIDAS, Markov switching, time-varying parameter models, and GARCH-family volatility models.
- **Visualization extensions:** event-study plots, forecast evaluation plots, structural-break plots, and richer table exports.
- **Provenance and audit logs:** machine-readable records of specifications, transformations, model runs, and generated outputs.

### Later

- **Model registry and plugin interfaces:** a stable extension point for third-party estimators and visualization components.
- **Dataset adapters:** optional integrations for FRED, DBnomics, World Bank, OECD, and other macro data providers.
- **Research report generation:** reproducible markdown or Quarto-style summaries that combine config, diagnostics, tables, and figures.
- **Benchmark suite:** comparison against known econometrics examples with expected coefficients and diagnostics.

## Extension Template

Each planned extension should include:

- **User problem:** what workflow gets easier.
- **API surface:** proposed class, function, config keys, and output shape.
- **Dependencies:** new runtime or optional packages.
- **Validation:** datasets, tests, and expected numerical checks.
- **Documentation:** README snippet, notebook, or reference page.
- **Release status:** experimental, beta, or stable.
