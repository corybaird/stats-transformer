# Testing suite

The automated suite checks package behavior on controlled data. It is the first validation layer, not a substitute for a numerical comparison with another econometric implementation.

## Contents

1. [Run the suite](#run-the-suite)
2. [Test directory guide](#test-directory-guide)
3. [Coverage areas](#coverage-areas)
4. [Test categories and interpretation](#test-categories-and-interpretation)
5. [Optional local checks](#optional-local-checks)

## Run the suite

From the repository root:

```bash
/opt/homebrew/bin/uv run python -m pytest -q
```

The `pyproject.toml` test configuration adds `src/` to the Python path. The test suite does not require MATLAB, network access, or external academic replication data.

## Test directory guide

`tests/` mirrors the library by responsibility. The table is a file-level map for contributors deciding where to add or inspect a check.

| Test path | Library area | Typical evidence |
| --- | --- | --- |
| `tests/test_pipeline.py` | `Pipeline` orchestration | configured stages, paths, and persisted outputs |
| `tests/test_data/` | packaged data and `DataMerger` | loading, joins, and data contracts |
| `tests/test_feature_engineering/` | `FeatureEngineer` | transformations, resampling, entity-aware behavior |
| `tests/test_models/test_regression_models.py` | OLS, robust OLS, panel OLS, IV, diagnostics | estimator behavior and summary metrics |
| `tests/test_models/test_discrete_models.py` | logit | binary-model fitting and metrics |
| `tests/test_models/test_timeseries_models.py` | VAR, VECM, and core time-series methods | estimator and utility behavior |
| `tests/test_models/test_unsupervised_models.py` | PCA and K-means | component and cluster outputs |
| `tests/test_timeseries_extensions.py` | structural VAR extensions and VAR-Toolbox translations | synthetic-data execution and bundled-example execution |
| `tests/test_visualization/` | charts, EDA, model visualizers, event studies | plot and visualizer construction |
| `tests/test_reporting/` | exporters | report serialization |
| `tests/test_utils/` | configuration, country conversion, time-series utilities | helper behavior and error conditions |
| `tests/test_examples/` | example registry | example discovery and registration |
| `tests/data/` | stable test fixtures | small local inputs used by tests |
| `tests/verification/` | opt-in external verification | MATLAB Engine comparison; not collected by default |

Run a focused area with the same module form, for example:

```bash
/opt/homebrew/bin/uv run python -m pytest tests/test_timeseries_extensions.py -q
```

Add a regression test beside the smallest existing test that exercises the behavior. Keep MATLAB, network, and large external-data checks in an opt-in verification or example workflow rather than the default suite.

## Coverage areas

| Area | Primary tests | What is checked |
| --- | --- | --- |
| Pipeline | `tests/test_pipeline.py` | YAML configuration, stage orchestration, and persisted workflow behavior |
| Features and data | `tests/test_feature_engineering/`, `tests/test_data/` | transformations, resampling, merging, data loading, and packaged examples |
| Regression and discrete models | `tests/test_models/test_regression_models.py`, `tests/test_models/test_discrete_models.py` | OLS variants, panel models, IV, diagnostics, and logit behavior |
| Time series | `tests/test_models/test_timeseries_models.py`, `tests/test_timeseries_extensions.py` | VAR, VECM, Granger utilities, Blanchard--Quah, proxy SVAR, sign restrictions, LP-IV, and decompositions |
| Visualization and reporting | `tests/test_visualization/`, `tests/test_reporting/` | chart construction, visualizer behavior, event studies, and exporters |
| Utilities | `tests/test_utils/` | configuration and country-name conversion |

The time-series extension tests use synthetic data to verify that new structural model classes fit and return expected object shapes. They do not establish equivalence with MATLAB outputs.

## Test categories and interpretation

- A **unit test** checks a narrow function, model method, or error condition.
- An **integration test** checks that multiple components work together, such as a configured pipeline stage.
- An **execution test** checks that a model can fit a specified dataset and return an expected result structure.
- A **numerical parity check** compares results with a defined independent implementation. Those comparisons are documented in [Academic and numerical validation](academic_examples.md).

Keep these categories separate in user-facing claims. Passing tests show that the package behaves as specified by its tests. They do not independently establish that an estimator matches a published application or another software package.

## Optional local checks

The following checks have additional prerequisites and are not part of the default suite:

- Academic feature examples require the relevant research data under `data/examples/academic/`.
- The MATLAB comparator requires MATLAB Engine for Python and a local VAR-Toolbox checkout. See [MATLAB cross-language comparator](matlab_comparator.md).
