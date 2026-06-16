# stats-transformer

`stats-transformer` is a Python library for macroeconomic data transformation, analysis, and visualization. Built around a configuration-driven architecture, it handles data ingestion, resampling, feature engineering, and econometric modeling for time-series and panel datasets.

## Features

- **Feature Engineering:** Advanced data transformations, frequency alignment, and robust merging capabilities for disparate datasets.
- **Econometric Modeling:** Built-in support for standard OLS, Robust OLS, Panel Regression, IV regression, discrete choice, time-series models, and unsupervised learning models (PCA, KMeans).
- **Visualization:** Automated generation of Exploratory Data Analysis (EDA) and regression model visual summaries (e.g., coefficient plots, residual plots, time-series tracking). Now includes a modular suite of standalone chart components for custom research plots.
- **Configuration-Driven Orchestration:** Fully integrated with YAML configuration (`params.yaml`) to enable reproducible, stage-based execution compatible with DVC pipelines.

## Quickstart

### 1. Installation

To use it in your project via PyPI:
```bash
uv add stats-transformer
```

For local development from this repository:

```bash
uv sync
```

### 2. Configuration (`params.yaml`)

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

### 3. Usage

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

### 4. Testing

Verify the installation and library integrity by running the test suite:

```bash
uv run pytest tests
```

For more details on test coverage, see the [Testing Suite](docs/validation/testing_suite.md).

## Current Status

The package has a typed `src/` layout, a PyPI Trusted Publishing workflow, bundled example data, notebooks, and a passing automated test suite. Before a broad public announcement, review the remaining release blockers in the [Publishing Plan](docs/library/publish_plan.md), especially CI version alignment, package metadata polish, and a clean installed-package smoke test.

## Documentation

- **Examples:** For examples of running the models, see [docs/validation/academic_examples.md](docs/validation/academic_examples.md).
- **Visualization Walkthrough:** For a guide on using the modular chart components, see [notebooks/07_chart_components.ipynb](notebooks/07_chart_components.ipynb).
- **System Design:** For more details on the system design, see [docs/library/architecture.md](docs/library/architecture.md).
- **File Structure:** For the standardized research folder structure, see [docs/library/file_structure.md](docs/library/file_structure.md).
- **Documentation Standards:** For standards on how to write research documentation, see [docs/research_standards/documentation.md](docs/research_standards/documentation.md).
- **Validation & Testing:** For details on the testing suite, see [docs/validation/testing_suite.md](docs/validation/testing_suite.md).
- **Publishing Plan:** For more details on the release process, please refer to [docs/library/publish_plan.md](docs/library/publish_plan.md).
