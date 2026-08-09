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

`stats-transformer` is a Python library for data transformation, econometric modeling, and visualization.. It handles data ingestion, resampling, feature engineering, and econometric modeling for time-series and panel datasets using a fully traceable and transparent configuration file workflow.

- **Unified Empirical Workflow:** Combines analysis-ready feature construction with econometric estimation, diagnostic checks, and publication outputs in a single reproducible pipeline.
- **Declarative YAML Orchestration:** Records data sources, frequency alignment, feature transformations, model specifications, and visual outputs in declarative YAML files that persist intermediate stage artifacts.
- **Broad Econometric Coverage:** Exposes standardized interfaces across cross-sectional, panel, time-series, instrumental-variable, discrete-choice, and unsupervised methods, reducing setup burden relative to composing separate APIs.
- **Auditable & Agent-Ready:** Incorporates verified MATLAB VAR translations (including machine-precision Blanchard-Quah verification against VAR-Toolbox 4.0) and includes a versioned architectural skill for compatible coding agents.

---

## 2. Documentation

- **Overview:** Start with [docs/overview.md](docs/overview.md) for the documentation map, model inventory, example inventory, and validation guide.
- **Academic Citations:** For citations of literature, paper datasets, and reference software, see [docs/library/citations.md](docs/library/citations.md).
- **Academic & Numerical Validation:** For paper transformation examples, replication benchmarks, and the MATLAB comparator, see [docs/validation/academic_validation.md](docs/validation/academic_validation.md).
- **Extensions & Roadmap:** For planned frequentist VAR extensions, see [docs/extensions/roadmap.md](docs/extensions/roadmap.md) and [docs/extensions/models.md](docs/extensions/models.md).
- **Visualization Walkthrough:** For a guide on using the modular chart components, see [notebooks/07_chart_components.ipynb](notebooks/07_chart_components.ipynb).
- **System Design:** For more details on the system design, see [docs/library/architecture.md](docs/library/architecture.md).
- **File Structure:** For the standardized research folder structure, see [docs/library/file_structure.md](docs/library/file_structure.md).
- **Validation & Testing:** For details on the testing suite, see [docs/validation/testing_suite.md](docs/validation/testing_suite.md).

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
