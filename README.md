# stats-transformer

## Table of Contents

- [1. Overview](#1-overview)
- [2. Documentation](#2-documentation)
- [3. Quickstart](#3-quickstart)
  - [3.1. Installation](#31-installation)
  - [3.2. Usage](#32-usage)
    - [3.2.1. With `params.yaml`](#321-with-paramsyaml)
    - [3.2.2. With the API directly](#322-with-the-api-directly)
  - [3.3. Testing](#33-testing)
- [4. Reproducing the paper](#4-reproducing-the-paper)
- [5. Agent Skill](#5-agent-skill)

---

## 1. Overview

`stats-transformer` is an open-source Python pipeline for applied econometrics in which a single configuration file (YAML) records every research decision. The pipeline

1. ingests, resamples, and merges raw data,
2. constructs analysis variables,
3. estimates cross-sectional, panel, and time-series models, and
4. saves the inputs, fitted model, and LaTeX tables and figures of each run.

It offers:

- **One configuration file per analysis:** each research decision from data ingestion to reporting is recorded in `params.yaml`, so an analysis can be rerun, shared, or reviewed in version control. Each run is written to a timestamped folder with the raw sources, the data the model was fitted on (`model_data.parquet`), the fitted model (`model.joblib`), LaTeX tables, figures, and a `manifest.json` (config and input checksums, git SHA, stages run, written files).
- **Frequency-aware data preparation:** aligns, merges, and transforms data (lags, differences, growth rates) within each entity, without hand-written reshaping.
- **Verification:** estimators are tested against recorded Stata, R, and MATLAB output, and data transformations against series published by the original authors.

Models cover cross-sectional (OLS, robust OLS), panel (fixed effects), time-series (VAR, VECM, SVAR including Blanchard-Quah, proxy, and sign restrictions, local projections), IV (2SLS), discrete choice (Logit, Probit), and unsupervised methods (PCA, KMeans).

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

Define your data sources, pipeline parameters, and model specifications in a `params.yaml` file. The example below uses `data/examples/regression/grunfeld.csv` from this repository: if you installed from PyPI, point `raw_data_file` at your own CSV and set the mapping and model columns for your data.

> **Note:** the example YAML, paper configs under `references/configs/`, and their datasets ship with the GitHub repository, not with the PyPI wheel. Clone the repo to run them.

```yaml
data:
  raw_data_file: data/examples/regression/grunfeld.csv
  featurization:
    entity_column: country
    date_column: date
    column_mapping:
      year: date
      firm: country

model:
  model_type: panel_ols
  target_variable: invest
  independent_variables:
    - value
    - capital

reporting:
  output_dir: reports/visualizations
```

Execute the full pipeline via the command line using the `Pipeline` orchestrator:

```bash
python -m stats_transformer.pipeline --config params.yaml
```

(In a local checkout you can additionally `uv sync` and run `uv run python -m stats_transformer.pipeline --config params.yaml`.)

Or drive the same configuration programmatically:

```python
from stats_transformer import Pipeline

pipeline = Pipeline(params_path="params.yaml")
transformed_data = pipeline.run(stage="features")
model_results = pipeline.run(stage="model")
pipeline.run(stage="reporting")
```

With a `raw_data_file` config like the one above, the features stage reads the raw CSV directly and the data stage is skipped (it only applies to multi-dataset `data.datasets` configs; a full run with no `stage` argument handles this automatically).

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

## 4. Reproducing the paper

The four empirical applications in the paper are YAML configs under `references/configs/`
(the same list as in `scripts/generate_paper_applications.py`):

- `references/configs/regression/mincer_wage.yaml`
- `references/configs/regression/grunfeld_panel.yaml`
- `references/configs/timeseries/reduced_form/macro_var.yaml`
- `references/configs/timeseries/identification/blanchard_quah_1989.yaml`

Run each from the repository root, for example:

```bash
uv sync
uv run python -m stats_transformer.pipeline --config references/configs/regression/mincer_wage.yaml
```

Each run writes into a timestamped folder under `data/pipeline/<config name>/` with the fitted model
(`model.joblib`), the estimation data (`model_data.parquet`), a JSON summary (`model_summary.json`),
LaTeX tables and figures, and a `manifest.json`.

The paper's IRF and coefficient figures are regenerated by:

```bash
uv run python src/examples/academic/stats_transformer_paper_figures.py
```

which writes PDFs to `reports/overleaf/figures/` (the script takes its inputs from the repo, so it can
be run from any working directory).

Replication scripts for the papers in the references run with `python -m` from the repository root, e.g.

```bash
uv run python -m examples.academic.bauer_swanson_2023
```

Each script prints its key estimates. Scripts that check against published or reference values say so in their output.

---

## 5. Agent Skill

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
