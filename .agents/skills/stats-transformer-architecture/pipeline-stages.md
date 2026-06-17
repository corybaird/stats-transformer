# Pipeline Stages

`Pipeline.run(stage=None)` is defined at `src/stats_transformer/pipeline.py:195`. `stage=None` runs the full pipeline; otherwise only the named stage runs. The dispatcher reads `params.yaml` on every call (`pipeline.py:198`).

## Path resolution

Before any stage runs, `Pipeline.run` resolves two paths from the YAML:

- `raw_data_file` → `data.raw_data_file` (or, for the `resample` stage, `data.datasets[*].path`).
- `merged_output_path` → `data.merge.output_path` (default `data/pipeline/resampled_merged.parquet`).
- `processed_path` → `data.featurization.output_path`.
- `summary_path` → `model.summary_output_path`.
- `viz_output_dir` → `visualization.output_dir` (default `reports/visualizations`).

`effective_data_path = merged_output_path if it exists else raw_data_file` (`pipeline.py:206`) — this is the path that `features`, `regression`, `eda`, and `visualization` stages will use when no earlier artifact is available.

## `resample`

Source: `src/stats_transformer/pipeline.py:214`.

1. For each `data.datasets` entry, call `FeatureEngineer.load_data(path)` and `FeatureEngineer.resample_dataset(df, ds_cfg)`.
2. Chain the resampled frames with `DataMerger.merge(..., on, how)` (defaults `on=[country, date]`, `how=outer`).
3. Write merged output as parquet to `data.merge.output_path`.

Inputs: raw CSVs/parquets under `data/raw/`.
Output: `data/pipeline/resampled_merged.parquet`.

## `features`

Source: `src/stats_transformer/pipeline.py:237`.

1. Read `effective_data_path` (CSV or parquet).
2. Call `Pipeline.fit_transform(effective_data_path, fit_model=False)` → invokes `FeatureEngineer.fit_transform` (`src/stats_transformer/featurization/feature_engineering.py:345`).
3. If `data.featurization.output_path` is set, write the result as CSV.

Inputs: merged parquet (or raw file).
Output: featurized CSV at `data/final/...` (or whatever `output_path` says).

## `eda`

Source: `src/stats_transformer/pipeline.py:245`.

Instantiates `EDAVisualizer(params_path=..., output_dir=viz_output_dir)` and delegates to `EDAVisualizer.run(data_path=effective_data_path)`.

## `regression`

Source: `src/stats_transformer/pipeline.py:251`.

1. If `processed_path` exists, read it as the featurized dataset; else call `Pipeline.fit_transform(effective_data_path)` which fits the model in addition to transforming.
2. `self.model.fit(transformed_data)` then `self.model.get_model_metadata()`.
3. `Pipeline.save_results(output_dir="reports")` writes `transformed_data_<timestamp>.csv` and `model_summary.json`.
4. If `model.summary_output_path` is set, also write the summary there.

Output: `reports/model_summary.json` (stable name) and `reports/transformed_data_<timestamp>.csv`.

## `visualization`

Source: `src/stats_transformer/pipeline.py:265`.

1. If `transformed_data` and `model_results` are not in memory, load them from `processed_path` and `summary_path`.
2. Call `Pipeline.create_visualizations(output_dir=viz_output_dir)`, which:
   - Builds a `RegressionVisualizer` (for coefficient plots / model summary plots) and a `DataVisualizer` (for time-series tracking).
   - For each plot type in `visualization.regression.plots` (default `[coefficient_plot, model_summary, residuals]`) render and save.
   - On exception, log and continue (`pipeline.py:184`).

Output: PNG files under `viz_output_dir` (typically `reports/visualizations/`).

## `None` (full run)

Source: `src/stats_transformer/pipeline.py:275`.

Runs `resample` → `features` → `regression` → `visualization` in that order, persisting each artifact. Use this from a CLI invocation:

```bash
uv run python -m stats_transformer.pipeline --config params.yaml
```

For a single stage:

```bash
uv run python -m stats_transformer.pipeline --config params.yaml --stage regression
```

## Adding a new stage

1. Pick a stable string key (e.g. `diagnostics`).
2. Add an `elif stage == "<key>":` branch to `Pipeline.run()` (`pipeline.py:245`–`272`) that reuses `effective_data_path`, `processed_path`, `summary_path`, `viz_output_dir` as appropriate.
3. Add a corresponding stage entry in `dvc.yaml` with explicit `deps`, `params`, and `outs` per project rules.
4. Document the new stage in this file.
5. Update `references/configs/test_pipeline.yaml` if a fixture is needed.
