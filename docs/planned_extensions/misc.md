# Miscellaneous Planned Extensions

This file captures smaller extensions that do not yet need their own design document.

## Granger Causality Tester

A generalized `GrangerCausalityTester` has a first-pass implementation in `stats_transformer.models.timeseries`. It abstracts away the paper-specific setup used for Granger causality testing (e.g., sorting, optional z-scoring, and running entity-by-entity checks).

The implementation wraps `statsmodels.tsa.stattools.grangercausalitytests` and returns tidy results that can be filtered, joined into reports, or summarized with `get_model_metadata()`.

Implemented API:

```python
from stats_transformer.models.timeseries import GrangerCausalityTester

tester = GrangerCausalityTester(
    entity_column="country",
    date_column="date",
    max_lag=4,
    standardize=True,
)
results = tester.fit(data, caused="gdp_growth", causing="interest_rate")
summary = tester.get_model_metadata()
```

Implemented:

- Support both single-series and panel workflows.
- Return tidy results with entity, lag, test statistic, p-value, and selected lag.
- Validate against a deterministic simulated dataset where the causal direction is known.

Still planned:

- Add config support under `model.model_type: granger_causality`.
- Add visualization helpers for lag-wise p-values.
- Add model-card/report export support.

## Example Dataset Registry

Add a small public API for discovering bundled examples:

```python
from stats_transformer.data import list_examples, load_example

list_examples()
df = load_example("okuns_law")
```

This would make README snippets and notebooks easier to keep stable after package installation.

## Optional Dependency Extras

Split heavier workflows into optional extras so the default install stays lean:

- `stats-transformer[notebooks]` for Jupyter tutorials.
- `stats-transformer[data]` for external data-provider clients.
- `stats-transformer[dev]` for testing, linting, and release tooling.

## Report Export Helpers

Add exporters that turn model metadata and visualization outputs into reproducible artifacts:

- Markdown model cards.
- LaTeX regression tables.
- JSON run manifests.
- Figure indexes with file paths and captions.
