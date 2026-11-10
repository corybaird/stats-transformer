---
name: stats-transformer-architecture
description: How to run the stats-transformer pipeline from a params.yaml config. Use when estimating a model (OLS, robust OLS, panel OLS, IV, logit/probit, PCA/KMeans, VAR/VECM/SVAR, local projections, BVAR), writing a params.yaml, or running a Pipeline stage. For modifying the library (adding a model, transformation, chart, or stage), read the supporting files in this directory.
---

# stats-transformer

Configuration-driven econometrics pipeline: one `params.yaml` declares data, transformations, model, and reporting; `Pipeline` runs it end-to-end. Each stage persists its artifact (parquet/csv/json), so stages can be re-run independently.

## Run a model

1. Write a `params.yaml`. Only `model_type`, `target_variable`, and `independent_variables` are required:

```yaml
data:
  raw_data_file: data/examples/regression/mincer_wage.csv
  featurization: {transformations: []}   # optional; see transformations.md
model:
  model_type: robust_ols
  target_variable: lwage
  independent_variables: [educ, exper, expersq, tenure]
  robust_ols: {cov_type: HC3}            # model-specific options
reporting:
  output_dir: reports/visualizations
```

2. Run it:

```python
from stats_transformer import Pipeline
Pipeline(params_path="params.yaml").run(stage="model")  # or .run() for all stages
```

Stages: `data`, `features`, `model`, `reporting`, `eda`; `None` runs the first four in order.

Common `model_type` values: `ols`, `robust_ols`, `panel_ols`, `iv`, `panel_iv`, `logit`, `probit`, `pca`, `kmeans`, `var`, `vecm`, `svar`, `blanchard_quah`, `local_projections`, `lp_iv`, `bvar`, `arima`. The full list is `MODEL_REGISTRY` in `src/stats_transformer/models/registry.py`; per-model details are in `models.md`.

## Config cheat-sheet

`model_type` (aliases in parentheses): `ols`, `robust_ols`, `panel_ols`, `iv` (`iv_2sls`, `2sls`), `panel_iv`, `gmm`, `did`, `logit`, `probit`, `pca`, `kmeans`, `var`, `vecm`, `svec`, `svar`, `blanchard_quah`, `proxy_svar`, `sign_restrictions` (`sign_zero`), `volatility_svar`, `independence_svar`, `cvm_svar` (`cvm`), `non_gaussian_svar` (`non_gaussian`), `dynamic_factor`, `bvar`, `tvar`, `tvecm`, `stvar`, `local_projections`, `lp_iv`, `bsvar_sign_zero` (`bayesian_sign_zero`), `pandemic_bvar`, `arima`.

Required model keys by kind:
- Single-equation / panel / iv / discrete (`ols`, `robust_ols`, `panel_ols`, `iv`, `logit`, `probit`, `gmm`, `did`, ...): `target_variable` + `independent_variables` (single column names).
- `iv` / `panel_iv` additionally need `endogenous` + `instruments`.
- VAR family (`var`, `vecm`, `svar`, ...): `target_variables` (plural, a list) + `maxlags`.
- Unsupervised (`pca`, `kmeans`): no target/independent required.

Model-specific options live in a sub-key named after the model type:
- `robust_ols.cov_type` (HC1 / HC3 / HAC).
- `ols.add_entity_fixed_effects` (bool), `panel_ols.entity_column`.
- `iv.endogenous`, `iv.instruments`.

Transformations append a suffix to every numeric column: `{col}_{transform}`.
- `changepct` → `{col}_changepct`, `changeraw` → `{col}_changeraw`, `log` → `{col}_log`, `zscore` → `{col}_zscore`, `rollingmean` → `{col}_rollingmean`, `lag1` → `{col}_lag1`, `lead1` → `{col}_lead1`, `changepct_yoy` → `{col}_changepct_yoy`.

Multi-source resampling appends `{col}_{period}{agg}` first (`period` ∈ D/M/Q/Y; `agg` ∈ mean/sum/first/last, only when actually resampled), then the transformation suffix. Example: monthly CPI downsampled to quarterly `last`, then `changepct`, is `CPIAUCSL_Mlast_changepct`. Reference the suffixed name in `target_variable` / `independent_variables` / `target_variables`.

## Where things live

- `src/stats_transformer/pipeline.py` — `Pipeline` orchestrator (stage dispatch, path resolution).
- `src/stats_transformer/models/` — model implementations; `base.py` is the `ModelBase` contract, `registry.py` maps `model_type` to a class.
- `src/stats_transformer/featurization/` — `FeatureEngineer` (transformations/resampling) and `DataMerger` (joins).
- `src/stats_transformer/reporting/` — visualizers and table generator.
- `references/configs/` — example `params.yaml` files.

## Supporting files (read before modifying the library)

- `architecture.md` — component tree, configuration center, data lifecycle.
- `pipeline-stages.md` — per-stage contract and data flow.
- `models.md` — model-by-model reference (parameters, edge cases).
- `transformations.md` — `FeatureEngineer` transformations, resampling, `DataMerger` joins.

## Modifying the library (summary; details in the files above)

- Add a transformation → `featurization/feature_engineering.py` (`valid_base_transformations` + `fit_transform` dispatch).
- Add a model → new file under `models/<family>/` subclassing `ModelBase`; register in `registry.py:MODEL_REGISTRY`.
- Add a chart → new class in `reporting/charts/`; export from `reporting/charts/__init__.py` and `reporting/__init__.py`.
- Add a YAML config → `references/configs/<name>.yaml`.
- Add a stage → add `run_<stage>()` and dispatch from `Pipeline.run()`; mirror in `dvc.yaml`.
