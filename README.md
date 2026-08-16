# stats-transformer

## Table of Contents

- [1. Features](#1-features)
- [2. Documentation](#2-documentation)
- [3. Quickstart](#3-quickstart)
  - [3.1. Installation](#31-installation)
  - [3.2. Configuration (`params.yaml`)](#32-configuration-paramsyaml)
  - [3.3. Usage](#33-usage)
  - [3.4. Testing](#34-testing)
- [4. Agent Skill](#4-agent-skill)

---

## 1. Features

`stats-transformer` is a Python library for data transformation, econometric modeling, and visualization. It automates feature engineering, model estimation, and reporting across time-series, panel, and cross-sectional datasets using a reproducible configuration workflow.

- **Unified Empirical Workflow:** Combines feature engineering, model estimation, diagnostic checks, and publication-ready charts in one pipeline.
- **Declarative YAML Configuration:** Controls data sources, frequency resampling, transformations, model specifications, and visual outputs via `params.yaml`.
- **Broad Econometric Coverage:** Provides unified APIs for cross-sectional (OLS, Robust OLS), panel (Fixed Effects), time-series (VAR, VECM, SVAR, Local Projections), instrumental-variables (2SLS, LP-IV), discrete-choice (Logit), and unsupervised methods (PCA, KMeans).
- **Cross-Language Validation & Agent-Ready:** Numerically verified against R (`stats`, `vars`), Stata (`regress`, `logit`, `pca`), and MATLAB (`mldivide`), with an embedded architectural skill for AI agents.
- **Extensible Roadmap:** Built for active expansion into high-frequency, non-linear, and structural macroeconomic extensions (see [Roadmap](docs/extensions/roadmap.md)).

---

## 2. Documentation

- **Core Guides (`docs/`)**
  - **[Overview](docs/overview.md):** Documentation map, model inventory, and example index.
  - **[Architecture & Design](docs/library/architecture.md):** Pipeline stages, object hierarchy, and data flow.
  - **[Model Reference](docs/library/models.md):** Configuration and usage organized by model family.
  - **[Numerical Validation](docs/validation/validation.md):** Cross-language R, Stata, and MATLAB verification matrix.
  - **[Testing Suite](docs/validation/testing_suite.md):** Automated unit, integration, and verification test guide.
  - **[Academic Citations](docs/library/citations.md):** Literature sources, paper datasets, and reference software.
  - **[Roadmap & Extensions](docs/extensions/roadmap.md):** Planned model expansions and frequentist VAR milestones.
  - **[File Structure](docs/library/file_structure.md):** Cookiecutter-based research folder layout.

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

For local development from this repository:

```bash
uv sync
```

### 3.2. Configuration (`params.yaml`)

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

visualization:
  output_dir: reports/visualizations
```

### 3.3. Usage

Load a packaged example dataset:

```python
from stats_transformer.data import list_examples, load_example

print(list_examples())
df = load_example("macrodb_gdp_inflation")
```

You can execute the pipeline via the command line using the `Pipeline` orchestrator:

```bash
# Run the full end-to-end pipeline
uv run python -m stats_transformer.pipeline --config params.yaml
```

Or you can interact with the API programmatically:

```python
from stats_transformer import Pipeline

# Initialize the pipeline with your configuration
pipeline = Pipeline(params_path="params.yaml")

# Run specific stages sequentially
merged_data = pipeline.run(stage="resample")
transformed_data = pipeline.run(stage="features")
model_results = pipeline.run(stage="regression")

# Generate and save visualizations
pipeline.run(stage="visualization")
```

### 3.4. Testing

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
