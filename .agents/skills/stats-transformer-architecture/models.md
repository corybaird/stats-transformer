# Models

All models subclass `ModelBase` (`src/stats_transformer/models/base.py:10`) and implement the same contract: `fit`, `predict`, `get_model_metadata`, plus the abstract methods `build_model`, `get_summary`, `get_model_metrics`.

## Interface contract

`ModelBase.__init__` (`base.py:12`) reads from `params.yaml` if `params_path` is given:

- `self.target = params.model.target_variable`
- `self.independent_variables = params.model.independent_variables`
- `self.add_entity_fixed_effects = params.model.ols.add_entity_fixed_effects`
- `self.entity_column = params.data.featurization.entity_column`

`ModelBase` raises `ValueError` if `target` or `independent_variables` is empty (`base.py:31`). Always set both in `params.yaml`.

`fit(data)` is the main entry point. It:

1. Calls `load_data(data)` which accepts a path or DataFrame and indexes on `[entity_column, 'date']` (or just `date`) when both columns exist.
2. Calls `build_model()` (abstract — must be implemented).
3. Returns `get_model_metrics()`.

`get_model_metadata(metrics=None)` (`base.py:119`) produces a JSON-serializable dict with:

- `model_version` (timestamp)
- `params` (raw `model` section of YAML)
- `metrics` (whatever `get_model_metrics()` returns)
- `coefficients` (variable → value, std_err, t_value, p_value, ci_lower, ci_upper)
- `summary` (dependent_variable, independent_variables, model_type, plus rsquared, fvalue, aic, bic, nobs, df_resid when available)

## Models and their files

### Regression family (`src/stats_transformer/models/regression/`)

- `RegressionModel` (`regression.py`) — plain OLS via `statsmodels.api.OLS`. Supports `add_entity_fixed_effects` for demeaning before fitting. Set `model.model_type: ols`.
- `RobustOLSModel` (`robust_ols.py`) — OLS with HC1/HC3 robust standard errors. Set `model.model_type: robust_ols`. Good default for cross-sectional or panel data with heteroskedasticity.
- `PanelRegressionModel` (`panel.py`) — uses `linearmodels.panel.PanelOLS` for proper entity/time fixed effects with cluster-robust inference. Set `model.model_type: panel_ols`. Requires a MultiIndex `[entity_column, date]`.
- `IVModel` (`iv.py`) — Instrumental Variables via `linearmodels.IV2SLS` or `statsmodels.sandbox.regression.gmm.IV2SLS`. Set `model.model_type: iv`.

### Unsupervised family (`src/stats_transformer/models/unsupervised/`)

- `PCAModel` (`unsupervised.py`) — Principal Component Analysis via `sklearn.decomposition.PCA`. `model.model_type: pca`. `target_variable` is unused; set it to a placeholder string.
- `KMeansModel` (`unsupervised.py`) — K-Means clustering. `model.model_type: kmeans`. `target_variable` is unused; set it to a placeholder string. Configure the number of clusters via `model.kmeans.n_clusters` if exposed by the config schema.

### Time series family (`src/stats_transformer/models/timeseries/`)

- `LocalProjectionsModel` (`local_projections.py`) — Jordà (2005) local projections for impulse responses. Set `model.model_type: local_projections`. Produces horizon-by-horizon coefficients, suitable for `IRFPlot`.
- `VARModel` (`var.py`), `VECMModel` (`vecm.py`), `SVARModel` (`svar.py`) — vector autoregression, VECM, and structural VAR. These are wired into `Pipeline._initialize_from_params` only in their respective `model_type` branches if they exist; check the dispatcher (`src/stats_transformer/pipeline.py:50`) before relying on a particular `model_type` string.
- `granger.py` — Granger causality tests (diagnostic, not a stand-alone fitted model).

### Discrete family (`src/stats_transformer/models/discrete/`)

- `logit.py` — `LogitModel`. Set `model.model_type: logit`. The target must be binary (0/1).

## Choosing a `model_type`

| Question | Pick |
| --- | --- |
| Cross-section with possible heteroskedasticity | `robust_ols` |
| Panel with entity (and/or time) effects | `panel_ols` |
| Need an instrumented regression | `iv` |
| Classification with a binary target | `logit` |
| Dimension reduction / factors | `pca` |
| Cluster exploration | `kmeans` |
| Impulse responses to a shock | `local_projections` |
| Multivariate forecasting with cointegration | `vecm` (or `var`) |

## Common edge cases

- `ModelBase.load_data` (`base.py:73`) drops NaNs after subsetting to required columns. If the panel is sparse, many rows will be lost — pre-impute or relax requirements.
- `ModelBase.fit` (`base.py:168`) has a NaN/inf retry: it will rebuild with NaN-masked X/y if the initial fit fails.
- `ModelBase._get_required_columns` (`base.py:99`) auto-includes `entity_column` and `time_column` if set on the subclass.
- `get_model_metadata` swallows coefficient-extraction errors and logs a warning (`base.py:136`); check `summary.model_type` and `metrics` even if `coefficients` is empty.

## Adding a new model

1. Subclass `ModelBase` in `src/stats_transformer/models/<family>/<name>.py`.
2. Implement `build_model`, `get_summary`, `get_model_metrics`.
3. Register in that family's `__init__.py` and in `src/stats_transformer/models/__init__.py`.
4. Add a branch in `Pipeline._initialize_from_params` (`src/stats_transformer/pipeline.py:50`) and `_initialize_from_args` (`pipeline.py:69`).
5. Add a row to the "Models and their files" table above.
