# stats-transformer

## Table of Contents

- [1. Features](#1-features)
- [2. Documentation](#2-documentation)
- [3. Quickstart](#3-quickstart)
  - [3.1. Installation](#31-installation)
  - [3.2. Usage](#32-usage)
    - [3.2.1. With `params.yaml`](#321-with-paramsyaml)
    - [3.2.2. With the API directly](#322-with-the-api-directly)
  - [3.3. Testing](#33-testing)
- [4. Agent Skill](#4-agent-skill)

---

## 1. Overview 

`stats-transformer` is a Python library for data transformation, econometric modeling, and visualization. It automates feature engineering, model estimation, and reporting across time-series, panel, and cross-sectional datasets using a reproducible configuration workflow.

- **Reproducible Workflows:** All research decisions from data ingestion to model estimation are produced and tracked via single source of truth (`params.yaml`)
  - **Data stage:** Data ingestion, resampling, merging (integration resampling to align frequencies)
  - **Transformation stage:** feature construction
  - **Modeling stage:** estimation, diagnostics, and publication ready outputs
- **Broad Econometric Coverage:** Covers cross-sectional (OLS, robust OLS), panel (fixed effects), time-series (VAR, VECM, SVAR, local projections), IV (2SLS), discrete choice (Logit), and unsupervised methods (PCA, KMeans).
	- **Validated via legacy software**: Model estimation and transformation validated against MATLAB, R, and Stata

---

## 2. Documentation

- **Core Guides (`docs/`)**
  - **[Overview](docs/overview.md):** Documentation map, model domain summary, and quickstart links.
  - **[Architecture & Design](docs/library/architecture.md):** Pipeline stages, object hierarchy, and repository file structure.
  - **[Implemented Models](docs/library/models.md):** Complete catalog of all 32 implemented model classes and utilities.
  - **[Data Guide](docs/library/data.md):** Data directory organization and packaged example datasets.
  - **[Software Benchmarks](docs/validation/benchmarks.md):** Cross-language R, Stata, and MATLAB verification matrix.
  - **[Testing Suite](docs/validation/testing_suite.md):** Automated unit, integration, and verification test guide.
  - **[Academic Citations](docs/library/citations.md):** Literature sources, paper datasets, and reference software.
  - **[Future Roadmap](docs/roadmap.md):** Planned model expansions and complexity triage.

- **Interactive Notebooks (`notebooks/`)**
  - **[Overall Pipeline](notebooks/0-overall-pipeline.ipynb):** End-to-end data ingestion, transformation, and estimation.
  - **[Regression & Panel](notebooks/01_regression_examples.ipynb):** OLS, robust covariance, panel, and 2SLS examples.
  - **[Time Series & Structural VAR](notebooks/08_structural_timeseries_models.ipynb):** VAR, SVAR, local projection, and decomposition APIs.
  - **[Modular Chart Components](notebooks/07_chart_components.ipynb):** Interactive visualization components and publication plotting.

---

## 3. Quickstart

### 3.1. Installation

To use it in your project via PyPI:

```bash
pip install stats-transformer
```

Or add it with `uv`:

```bash
uv add stats-transformer
```

To run the notebooks under `notebooks/`, install the `notebooks` extra:

```bash
pip install "stats-transformer[notebooks]"
```

For local development from this repository:

```bash
uv sync
```

### 3.2. Usage

#### 3.2.1. With `params.yaml`

Define your data sources, pipeline parameters, and model specifications in a `params.yaml` file:

```yaml
data:
  featurization:
    entity_column: country
  datasets:
    - name: macro_data
      path: data/raw/macro_indicators.csv
      frequency: Q

model:
  model_type: panel_ols
  target_variable: gdp_growth
  independent_variables:
    - interest_rate
    - inflation

reporting:
  output_dir: reports/visualizations
```

Execute the full pipeline via the command line using the `Pipeline` orchestrator:

```bash
uv run python -m stats_transformer.pipeline --config params.yaml
```

Or drive the same configuration programmatically:

```python
from stats_transformer import Pipeline

pipeline = Pipeline(params_path="params.yaml")
merged_data = pipeline.run(stage="resample")
transformed_data = pipeline.run(stage="features")
model_results = pipeline.run(stage="regression")
pipeline.run(stage="reporting")
```

#### 3.2.2. With the API directly

Skip the YAML file and instantiate model classes directly. Start from a packaged example dataset:

```python
from stats_transformer.data import list_examples, load_example

print(list_examples())
df = load_example("macrodb_gdp_inflation")
```

Then fit any model class on the data:

```python
from stats_transformer.models.regression.robust_ols import RobustOLSModel
from stats_transformer.reporting.models.regression_viz import RegressionVisualizer

model = RobustOLSModel(
    target="gdp",
    independent_variables=["inflation"],
    cov_type="HC3",
)
model.fit(df)
print(model.get_summary())

viz = RegressionVisualizer(output_dir="reports/visualizations")
paths = viz.visualize_from_json(model_summary=model.get_model_metadata())
```

All 32 model classes accept direct instantiation like this; see [Implemented Models](docs/library/models.md) for each class's arguments. End-to-end examples live in the [notebooks](#2-documentation).

### 3.3. Testing

Verify the installation and library integrity by running the test suite:

```bash
uv run python -m pytest -q
```

For more details on test coverage, see the [Testing Suite](docs/validation/testing_suite.md).

---

## 4. Agent Skill

Source checkouts of this repository include an optional `stats-transformer-architecture` agent skill for AI coding tools. It gives agents a compact map of the library architecture, pipeline stages, model contracts, and feature-engineering vocabulary.

The canonical skill source lives at `.agents/skills/stats-transformer-architecture/`. From the repository root, use the local terminal to run `scripts/install-agent-skill.sh`, which installs or refreshes tool-specific copies for Claude Code, OpenAI Codex, or Kilo Code:

```bash
./scripts/install-agent-skill.sh all
```

You can also target one tool at a time:

```bash
./scripts/install-agent-skill.sh claude
./scripts/install-agent-skill.sh codex
./scripts/install-agent-skill.sh kilo
```
