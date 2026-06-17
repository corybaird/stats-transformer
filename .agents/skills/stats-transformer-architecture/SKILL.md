---
name: stats-transformer-architecture
description: Library architecture and pipeline usage for the stats-transformer Python package. Use this skill when working in this repo on params.yaml changes, Pipeline stage additions or modifications, FeatureEngineer transformations (log, lag, lead, zscore, changepct, rollingmean), DataMerger joins, model implementations (OLS, Robust OLS, Panel OLS, PCA, KMeans, LocalProjections, VAR, VECM, SVAR, IV, Logit), visualization components (CoefficientBarChart, TimeSeriesPlot, IRFPlot, FacetedTimeSeries, BinnedScatterPlot, CorrelationHeatmap), resampling logic across annual/quarterly/monthly/daily frequencies, or any file under src/stats_transformer/, references/configs/, or dvc.yaml.
---

# stats-transformer Architecture

`stats-transformer` is a configuration-driven Python library for macroeconomic data transformation, modeling, and visualization. A single `params.yaml` describes datasets, transformations, model choice, and visualization style; the `Pipeline` class in `src/stats_transformer/pipeline.py:15` runs the configured stages end-to-end.

## Mental model

`params.yaml` → `Pipeline` orchestrator → stages in order: `resample` → `features` → `regression`/`model` → `visualization`. Each stage persists its artifact (parquet, csv, json) so stages can be run individually and re-cached.

The mermaid diagram in `docs/library/architecture.md:9` is the canonical flow. Read it before adding anything new.

## Repository layout

```
src/stats_transformer/
├── __init__.py            # Public API re-exports
├── pipeline.py            # YAML-driven orchestrator (Pipeline class)
├── featurization/
│   ├── feature_engineering.py   # FeatureEngineer: transformations + resampling
│   ├── data_merger.py           # DataMerger: multi-source panel joins
│   └── event_study.py           # EventStudyBuilder
├── models/
│   ├── base.py                  # ModelBase: fit / get_model_metadata contract
│   ├── regression/              # OLS, Robust OLS, Panel OLS, IV
│   ├── timeseries/              # VAR, VECM, SVAR, LocalProjections, Granger
│   ├── discrete/                # Logit
│   └── unsupervised/            # PCA, KMeans
├── visualization/
│   ├── base.py                  # BaseVisualizer
│   ├── charts/                  # Standalone chart components
│   ├── models/                  # RegressionVisualizer, ModelVisualizer
│   ├── eda/                     # DataVisualizer, EDAVisualizer
│   ├── defaults/                # Bundled palettes, styles
│   ├── formatters/              # Output formatters
│   ├── tables/                  # TableGenerator
│   └── utils/
├── data/                        # Packaged example datasets
├── reporting/                   # Reporting helpers
└── utils/                       # Shared utilities

references/
├── configs/                     # params.yaml files (the control center)
├── dictionaries/                # Constant-only mapping files
└── viz/                         # Generated viz configs (gitignored)
```

## Public API entry points

`src/stats_transformer/__init__.py:8` re-exports:

- `Pipeline`, `FeatureEngineer`, `EventStudyBuilder`
- `RegressionModel`
- `BaseVisualizer`, `DataVisualizer`, `ModelVisualizer`, `RegressionVisualizer`
- Standalone charts: `CoefficientBarChart`, `GroupedBarChart`, `StackedBarChart`, `TimeSeriesPlot`, `IRFPlot`, `FacetedTimeSeries`, `BinnedScatterPlot`, `ScatterWithRegression`, `CorrelationHeatmap`
- `TableGenerator`
- `LocalProjectionsModel`

## Where to make changes

| Task | Touch |
| --- | --- |
| Add a transformation (e.g. new `rollingstd`) | `src/stats_transformer/featurization/feature_engineering.py` (extend `valid_base_transformations` at line 309 and the dispatch in `fit_transform`) |
| Add a new model | New file under `src/stats_transformer/models/<family>/` subclassing `ModelBase` (`src/stats_transformer/models/base.py:10`); register in that family's `__init__.py` and in `src/stats_transformer/models/__init__.py`; wire into `Pipeline._initialize_from_params` (`src/stats_transformer/pipeline.py:50`) and `_initialize_from_args` (line 69) |
| Add a new chart component | New class in `src/stats_transformer/visualization/charts/` (mirror `bar.py` / `time_series.py`), then export from `src/stats_transformer/visualization/charts/__init__.py` and `src/stats_transformer/visualization/__init__.py` |
| Add a new YAML config | `references/configs/<name>.yaml` (template: `references/configs/test_pipeline.yaml`) |
| Add a packaged example dataset | `src/stats_transformer/data/` and expose via `list_examples` / `load_example` |
| Add a new dictionary mapping | `references/dictionaries/<NAME>.py` (constants only, per project rules) |
| Add a new pipeline stage | Extend `Pipeline.run()` in `src/stats_transformer/pipeline.py:195`; mirror the stage in `dvc.yaml` with explicit `deps` / `params` / `outs` |

## Pipeline stage contract

`Pipeline.run(stage=None)` is the dispatcher. Stage values: `resample`, `features`, `eda`, `regression`, `visualization`, or `None` for full run.

- `resample` — calls `FeatureEngineer.resample_dataset` per dataset in `data.datasets`, then `DataMerger.merge` to produce `data.merge.output_path` (default `data/pipeline/resampled_merged.parquet`).
- `features` — runs `FeatureEngineer.fit_transform` on the merged output (or `raw_data_file` if merge hasn't run) and writes `data.featurization.output_path`.
- `regression` — fits the configured model and writes `reports/model_summary.json`.
- `visualization` — reads the model summary and renders plots to `visualization.output_dir` (default `reports/visualizations`).
- `eda` — runs `EDAVisualizer` on the same data path.

The full-run branch (no `stage` arg) executes all of the above in order; see `src/stats_transformer/pipeline.py:275`.

Data-path resolution: `effective_data_path = merged_output_path if exists else raw_data_file` (`src/stats_transformer/pipeline.py:206`).

## `params.yaml` shape

```yaml
data:
  featurization:
    entity_column: country
    date_column: date
    period: annual
    transformations: [log, lag1, lag2, zscore]
    output_path: data/final/features.csv
  datasets:
    - name: macro_data
      path: data/raw/macro.csv
      frequency: Q
  merge:
    on: [country, date]
    how: outer
    output_path: data/pipeline/resampled_merged.parquet
  raw_data_file: data/raw/single_panel.csv

model:
  model_type: panel_ols          # ols | robust_ols | panel_ols | pca | kmeans | local_projections | var | vecm | svar | iv | logit
  target_variable: gdp_growth
  independent_variables: [interest_rate, inflation]
  ols:
    add_entity_fixed_effects: true
  summary_output_path: reports/tables/model_summary.json

visualization:
  output_dir: reports/visualizations
  file_format: png
  dpi: 300
  style: default
  regression:
    plots: [coefficient_plot, model_summary, residuals]
```

## Model interface contract (`ModelBase`)

Every model implements:

- `fit(data)` — loads data, calls `build_model`, returns metrics.
- `predict(data)` — transform + score on new data.
- `get_model_metadata(metrics=None)` — JSON-serializable dict with `coefficients`, `summary`, `metrics`.
- `build_model`, `get_summary`, `get_model_metrics` — abstract on `ModelBase` (`src/stats_transformer/models/base.py:107`).

YAML `model.independent_variables` and `model.target_variable` are required; the model raises `ValueError` if either is missing (`src/stats_transformer/models/base.py:31`).

## Supporting files in this skill

- [`architecture.md`](architecture.md) — full component tree and stage flow.
- [`pipeline-stages.md`](pipeline-stages.md) — per-stage contract, data flow, and how to add a new stage.
- [`models.md`](models.md) — model-by-model reference (purpose, parameters, edge cases).
- [`transformations.md`](transformations.md) — `FeatureEngineer` transformations, resampling, `DataMerger` join semantics.

## Pointers

- Canonical architecture doc: `docs/library/architecture.md`
- File-structure doc: `docs/library/file_structure.md`
- Library README: `README.md`
- Commit message conventions: `.agents/git-management-pypi.md` (NOT duplicated here)
- Test fixture config: `references/configs/test_pipeline.yaml`

## What this skill does NOT cover

- Commit message style → `.agents/git-management-pypi.md`.
- Python style rules (no docstrings, no type checks, single-line signatures, `run()` entry point) → project rules in `python-coder.md`.
- DVC pipeline YAML semantics beyond what's in `dvc.yaml` itself.
