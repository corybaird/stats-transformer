# Pipeline Stages

The pipeline has four research stages, listed in `STAGES = ("data", "features", "model", "reporting")` (`src/stats_transformer/pipeline.py:17`), plus the optional `eda` stage. Each stage is its own method on `Pipeline`; `Pipeline.run(stage=None)` (`pipeline.py:361`) is the dispatcher. `stage=None` runs the four stages in order; otherwise only the named stage runs. `run()` re-initializes from `params.yaml` on every call.

Legacy stage names still work but emit a `DeprecationWarning` via `_STAGE_ALIASES` (`pipeline.py:20`): `resample` → `data`, `regression` → `model`, `visualization` → `reporting`.

## Path resolution

Every stage method calls `Pipeline._resolve_paths()` (`pipeline.py:249`), which initializes from `params.yaml` if needed (so stage methods can be called directly) and resolves:

- `raw_data_file` → `data.raw_data_file`.
- `merged_output_path` → `data.merge.output_path` (default `data/pipeline/resampled_merged.parquet`).
- `processed_path` → `data.featurization.output_path`.
- `summary_path` → `model.summary_output_path`.
- `report_output_dir` → `reporting.output_dir` (default `reports/visualizations`; legacy `visualization:` key still read).

`effective_data_path = merged_output_path if it exists else raw_data_file` — this is the path that `features`, `model`, and `eda` use when no earlier artifact is available.

## `data` — `run_data()`

Source: `src/stats_transformer/pipeline.py:274`.

1. For each `data.datasets` entry, call `FeatureEngineer.load_data(path)` and `FeatureEngineer.resample_dataset(df, ds_cfg)`.
2. Chain the resampled frames with `DataMerger.merge(..., on, how)` (defaults `on=[country, date]`, `how=outer`).
3. Write merged output as parquet to `data.merge.output_path`.

Inputs: raw CSVs/parquets under `data/raw/`.
Output: `data/pipeline/resampled_merged.parquet`.

## `features` — `run_features(data_path=None)`

Source: `src/stats_transformer/pipeline.py:300`.

1. Read `data_path` or `effective_data_path` (CSV or parquet).
2. Call `Pipeline.fit_transform(path, fit_model=False)` → invokes `FeatureEngineer.fit_transform`.
3. If `data.featurization.output_path` is set, write the result as CSV.

Inputs: merged parquet (or raw file).
Output: featurized CSV at whatever `output_path` says.

## `model` — `run_model(data_path=None, transformed_data=None)`

Source: `src/stats_transformer/pipeline.py:312`.

1. Fit on `transformed_data` when given (the full run passes the in-memory features, so dtypes such as dates are preserved); else read `processed_path` if it exists; else call `Pipeline.fit_transform(effective_data_path)`, which fits the model in addition to transforming.
2. `Pipeline.save_results(output_dir="reports")` writes `transformed_data_<timestamp>.csv` and `model_summary.json`.
3. If `model.summary_output_path` is set, also write the summary there.

Output: `reports/model_summary.json` (stable name) and `reports/transformed_data_<timestamp>.csv`.

## `reporting` — `run_reporting()`

Source: `src/stats_transformer/pipeline.py:338`.

1. If `transformed_data` and `model_results` are not in memory, load them from `processed_path` and `summary_path`.
2. Call `Pipeline.create_visualizations(output_dir=report_output_dir)`, which builds a `RegressionVisualizer` (coefficient / model summary plots) and a `DataVisualizer` (time series), logging and continuing on exceptions.

Output: PNG files under `report_output_dir` (typically `reports/visualizations/`).

## `eda` — `run_eda(data_path=None)`

Source: `src/stats_transformer/pipeline.py:352`.

Instantiates `EDAVisualizer(params_path=..., output_dir=report_output_dir)` and delegates to `EDAVisualizer.run(data_path=effective_data_path)`.

## `None` (full run)

Calls `run_data()` → `run_features()` → `run_model(transformed_data=...)` → `run_reporting()` and returns the transformed DataFrame. Use this from a CLI invocation:

```bash
uv run python -m stats_transformer.pipeline --config params.yaml
```

For a single stage:

```bash
uv run python -m stats_transformer.pipeline --config params.yaml --stage model
```

## Adding a new stage

1. Pick a stable string key (e.g. `diagnostics`).
2. Add a `run_<key>()` method that starts with `paths = self._resolve_paths()`, then dispatch to it from `Pipeline.run()`; add the key to `STAGES` only if it belongs in the full run.
3. Add the key to the CLI `--stage` choices (they derive from `STAGES`).
4. Document the new stage in this file.
5. Update `references/configs/test_pipeline.yaml` if a fixture is needed.
