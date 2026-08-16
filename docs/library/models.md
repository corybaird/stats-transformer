# Model Reference

This page is organized by model family so that new estimators can be added as the library grows. Each numbered top-level section covers one family, while its numbered subsections document individual models. Future sections can add time-series, discrete-choice, and unsupervised models without changing the existing regression structure. For the complete model inventory and pipeline architecture, see [System architecture](architecture.md). For planned estimators and cross-language validation targets, see the [Model extension roadmap](../extensions/roadmap.md).

## 1. Regression Models

Regression models estimate relationships between a dependent variable and one or more regressors. This section currently covers the estimators that require panel or instrumental-variable configuration; additional regression estimators can be added as numbered subsections.

### Common Model Interface

`PanelRegressionModel`, `IV2SLSModel`, and `PanelIV2SLSModel` follow the `ModelBase` interface:

- `fit(data)` accepts a pandas `DataFrame` or a CSV/Parquet path and returns model metrics.
- `get_summary()` returns the estimator summary.
- `get_model_metadata()` returns JSON-serializable parameters, metrics, coefficients, and summary fields.
- `run(data_path, output_path=None)` fits the model and optionally saves its metadata.

For YAML-driven models, `model.target_variable` and `model.independent_variables` are required. `independent_variables` contains only exogenous regressors. Endogenous regressors belong in `endogenous`; excluded instruments belong in `instruments`.

### 1.1 Panel OLS (`PanelRegressionModel`)

`PanelRegressionModel` estimates a linear panel model with optional entity and time fixed effects using `linearmodels.panel.PanelOLS`.

#### Configuration

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `data.featurization.entity_column` | string | required | Column identifying the panel entity. |
| `data.featurization.date_column` | string | `date` | Column identifying the time period. Normalize it to `date` with `column_mapping` when using the YAML pipeline. |
| `model.target_variable` | string | required | Dependent variable. |
| `model.independent_variables` | list of strings | required | Exogenous regressors. |
| `model.panel_ols.entity_effects` | boolean | `true` | Include entity fixed effects. |
| `model.panel_ols.time_effects` | boolean | `false` | Include time fixed effects. |
| `model.panel_ols.check_rank` | boolean | `true` | Check the regressor matrix rank before estimation. |
| `model.panel_ols.cov_type` | string | `unadjusted` | Covariance estimator passed to `PanelOLS.fit`. Common choices include `unadjusted`, `robust`, and `clustered`. |
| `model.panel_ols.cluster_entity` | boolean | `false` | Cluster by entity when `cov_type: clustered`. |
| `model.panel_ols.cluster_time` | boolean | `false` | Cluster by time when `cov_type: clustered`. |

#### YAML Example

```yaml
data:
  raw_data_file: data/examples/regression/grunfeld.csv
  featurization:
    entity_column: country
    date_column: date
    column_mapping:
      firm: country
      year: date

model:
  model_type: panel_ols
  target_variable: invest
  independent_variables:
    - value
    - capital
  panel_ols:
    entity_effects: true
    time_effects: false
    cov_type: clustered
    cluster_entity: true
    cluster_time: false
```

#### Direct API

```python
from stats_transformer.models.regression.panel import PanelRegressionModel

model = PanelRegressionModel(
    target="invest",
    independent_variables=["value", "capital"],
    entity_column="firm",
    time_column="year",
    entity_effects=True,
    time_effects=False,
    cov_type="clustered",
    cluster_entity=True,
)
metrics = model.fit(data)
```

The returned metrics include overall, between, and within R-squared values, the model F-statistic and p-value, and the number of observations.

### 1.2 IV 2SLS (`IV2SLSModel`)

`IV2SLSModel` estimates a cross-sectional or pooled two-stage least-squares model using `linearmodels.iv.IV2SLS`:

$$
y = X\beta + D\gamma + \varepsilon,
$$

where `X` contains exogenous regressors, `D` contains endogenous regressors, and `Z` contains excluded instruments for `D`.

#### Direct API Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `target` | string | required | Dependent variable. |
| `independent_variables` | list of strings | required | Exogenous regressors included in both stages. |
| `endogenous` | list of strings | required for IV estimation | Endogenous regressors. |
| `instruments` | list of strings | required for IV estimation | Excluded instruments. |
| `cov_type` | string | `robust` | Covariance estimator passed to `IV2SLS.fit`. |

#### Direct API

```python
from stats_transformer.models.regression.iv import IV2SLSModel

model = IV2SLSModel(
    target="lwage",
    independent_variables=["exper", "expersq"],
    endogenous=["educ"],
    instruments=["motheduc", "fatheduc"],
    cov_type="robust",
)
metrics = model.fit(data)
```

`metrics["first_stage"]` contains the first-stage diagnostics produced by `linearmodels`. The model logs a weak-instrument warning when a reported first-stage F-statistic is below 10. The metadata also reports the model F-statistic, Wu-Hausman statistic, R-squared, and observation count.

### 1.3 Panel IV 2SLS (`PanelIV2SLSModel`)

`PanelIV2SLSModel` combines two-stage least squares with optional entity and time fixed effects. The implementation expands the requested fixed effects into dummy variables, adds them to the exogenous design matrix, and estimates the resulting specification with `linearmodels.iv.IV2SLS`.

#### Variable Roles

| YAML field | Role |
| --- | --- |
| `model.target_variable` | Dependent variable `y`. |
| `model.independent_variables` | Exogenous controls `X`. Do not repeat endogenous variables here. |
| `model.panel_iv.endogenous` | Endogenous regressors `D`. |
| `model.panel_iv.instruments` | Excluded instruments `Z`. |
| `data.featurization.entity_column` | Panel entity identifier. |
| `model.panel_iv.time_column` | Panel time identifier. Falls back to `data.featurization.date_column`. |

The number of excluded instruments must be at least the number of endogenous regressors. Rows containing missing or infinite values in any required model column are removed before estimation.

#### Panel IV Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `instruments` | list of strings | required | Excluded instruments. |
| `endogenous` | list of strings | required | Endogenous regressors. |
| `time_column` | string | `data.featurization.date_column`, then `date` | Column identifying the panel time period. |
| `entity_effects` | boolean | `true` | Add entity fixed-effect dummies. |
| `time_effects` | boolean | `false` | Add time fixed-effect dummies. |
| `cov_type` | string | `robust` | Covariance estimator passed to `IV2SLS.fit`. Use `clustered` for clustered standard errors. |
| `cluster_by` | string | `entity` | Clustering dimension when `cov_type: clustered`; must be `entity` or `time`. |

#### YAML Example

Panel IV-specific options belong under `model.panel_iv`.

```yaml
data:
  raw_data_file: data/examples/regression/panel_iv.csv
  featurization:
    entity_column: entity
    date_column: time
    period: annual
    transformations: []

model:
  model_type: panel_iv
  target_variable: y
  independent_variables:
    - w
  panel_iv:
    endogenous:
      - x
    instruments:
      - z
    time_column: time
    entity_effects: true
    time_effects: true
    cov_type: clustered
    cluster_by: entity
  summary_output_path: reports/tables/panel_iv_summary.json
```

The example assumes that `raw_data_file` points to a dataset containing the configured `entity`, `time`, `y`, `w`, `x`, and `z` columns. The YAML can also initialize the model directly:

```python
from stats_transformer.models.regression.panel_iv import PanelIV2SLSModel

model = PanelIV2SLSModel(params_path="references/configs/panel_iv.yaml")
metrics = model.fit("path/to/panel_iv.csv")
```

#### Direct API

```python
from stats_transformer.models.regression.panel_iv import PanelIV2SLSModel

model = PanelIV2SLSModel(
    target="y",
    independent_variables=["w"],
    endogenous=["x"],
    instruments=["z"],
    entity_column="entity",
    time_column="time",
    entity_effects=True,
    time_effects=True,
    cov_type="clustered",
    cluster_by="entity",
)
metrics = model.fit(data)
```

#### Diagnostics and Metadata

The Panel IV metrics contain:

- `first_stage`: first-stage partial R-squared and F-statistic diagnostics for each endogenous regressor.
- `wu_hausman_stat`: Wu-Hausman endogeneity-test statistic.
- `f_statistic` and `f_pvalue`: model F-statistic and p-value.
- `r_squared`: model R-squared.
- `num_observations`: estimation-sample size.

The model logs a warning for any endogenous regressor whose first-stage F-statistic is below 10. Treat this threshold as a screening diagnostic rather than a universal test of instrument validity. Instrument relevance does not establish the exclusion restriction; that assumption must be justified by the research design.

### 1.4 Choosing a Regression Model

| Research design | Model |
| --- | --- |
| Panel data with fixed effects and all regressors treated as exogenous | `PanelRegressionModel` |
| Cross-sectional or pooled data with endogenous regressors and excluded instruments | `IV2SLSModel` |
| Panel data requiring both fixed effects and instruments | `PanelIV2SLSModel` |

## 2. Time Series Models


## 3. Discrete Choice Models


## 4. Unsupervised Models
