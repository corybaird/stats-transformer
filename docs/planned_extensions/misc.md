# Miscellaneous Planned Extensions

This file captures smaller extensions that do not yet need their own design document.

## Granger Causality Tester

A generalized `GrangerCausalityTester` should be implemented in the future to abstract away the paper-specific setup used for Granger causality testing (e.g., resampling, z-scoring, and running region-by-region checks).

Currently, this logic lives in paper-specific pipelines (such as `notebooks.paper_econsurveys.modules.utils.granger_causality_tester.py`), but the underlying pattern of formatting panels for `statsmodels.tsa.stattools.grangercausalitytests` could be generalized as an extension to `stats_transformer.models.timeseries`.

Proposed API:

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

Implementation notes:

- Support both single-series and panel workflows.
- Return tidy results with entity, lag, test statistic, p-value, and selected lag.
- Add config support under `model.model_type: granger_causality`.
- Validate against a deterministic simulated dataset where the causal direction is known.

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
