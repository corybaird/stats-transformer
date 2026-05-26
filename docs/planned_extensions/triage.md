# Planned Extension Triage

`stats-transformer` should grow carefully. The package is strongest when it helps researchers move from messy macroeconomic data to reproducible transformations, model runs, diagnostics, and presentation artifacts. It should not become a loose pile of every econometric estimator that can be wrapped from `statsmodels`.

## Core Risk: Econometrics Sprawl

The main risk is turning the library into a sprawling econometrics toolbox with shallow wrappers around many models. That would create several problems:

- **Thin abstractions:** if a class only renames a `statsmodels` call, users are better served by the upstream library.
- **Maintenance drag:** every estimator adds tests, examples, docs, metadata conventions, and compatibility risk.
- **Unclear reliability:** advanced estimators can be easy to expose but hard to validate numerically.
- **API inconsistency:** too many model families can fragment result shapes, summary methods, and visualization expectations.
- **Dependency creep:** specialized models can pull in heavy optional packages that make installation slower and less predictable.
- **Documentation debt:** public model support implies examples, caveats, assumptions, and failure-mode guidance.
- **Brand dilution:** the project can lose its center if it tries to compete with full econometrics libraries instead of orchestrating reproducible research workflows.

## Selection Rule

Add an econometric model only when it satisfies most of these:

- It supports a common macroeconomic or applied-economics workflow.
- It benefits from the package's existing strengths: data alignment, panel handling, configuration, diagnostics, visualization, or reporting.
- It can return stable metadata compatible with downstream charts and reports.
- It has a deterministic test case with known behavior.
- It does not require a heavy dependency unless that dependency can be optional.
- It can be documented with a concise example that works from installed-package imports.

## Recommended Near-Term Order

### 1. ARIMA / SARIMAX

Good next candidate. It is a common forecasting workflow, has strong `statsmodels` support, and pairs naturally with the new forecast evaluator.

Risk: avoid hiding too much of the underlying `statsmodels` API. Keep the wrapper focused on fit, forecast, confidence intervals, and metadata.

### 2. Structural Break Diagnostics

High value and moderate complexity. Break tests, rolling coefficients, and recursive forecast diagnostics improve existing regression and time-series models without adding a large new model family.

Risk: tests must distinguish diagnostic convenience from formal inference promises.

### 3. Regression Discontinuity Helpers

Useful for applied work and pairs well with existing binned scatter visualization. Start with sharp RDD, local linear fits, cutoff filtering, and tidy metadata.

Risk: bandwidth selection and inference are subtle. Do not overclaim until robust methods are implemented and validated.

### 4. Event Study / Difference-in-Differences

Very useful, but more dangerous to implement casually. A first pass should focus on event-time design helpers and plotting metadata before promising modern staggered-DiD estimators.

Risk: TWFE event studies can mislead under staggered adoption. Documentation must be explicit.

### 5. MIDAS

Strategically aligned with mixed-frequency macro data. This could become a differentiator, but it requires more design care than a basic wrapper.

Risk: frequency alignment, lag weighting, and validation need careful examples.

## Defer or Keep Optional

- **BVAR:** useful, but priors and posterior summaries require careful API design.
- **Markov switching:** good for regime work, but interpretation and convergence failures need strong docs.
- **Dynamic factor models:** useful for macro panels, but should wait until metadata conventions are stable.
- **GARCH:** likely belongs behind an optional dependency extra such as `stats-transformer[volatility]`.
- **TAR/STAR and TVP models:** powerful but specialized; keep experimental until there are strong validation examples.

## Healthy Boundary

The package should prefer being an opinionated research workflow layer over being a comprehensive econometrics library. A good extension should make the complete workflow easier: load or align data, estimate the model, validate assumptions, produce tidy metadata, visualize results, and export reproducible artifacts.

If an estimator cannot connect to that workflow, it should probably stay in planned docs or be left to upstream libraries.
