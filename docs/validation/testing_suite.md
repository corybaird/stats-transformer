# Testing Suite

This document describes the testing architecture for the `stats_transformer` library. The suite ensures that all econometric models, data transformation pipelines, and visualization tools operate correctly and produce statistically valid results.

## Running Tests

The library uses `pytest` for unit and integration testing. To run the full test suite, use the following command:

```bash
uv run pytest tests
```

## Test Coverage

The suite is divided into several logical areas:

### 1. Econometric Models (`tests/test_regression_models.py`, `tests/test_discrete_models.py`, `tests/test_timeseries_models.py`)
- **Robust OLS**: Validates that robust covariance estimators (HC0-HC3) are correctly applied and that results match expectations.
- **Panel Regression**: Tests entity and time fixed effects using `PanelOLS` from `linearmodels`.
- **IV 2SLS**: Ensures that instrumental variable models correctly handle endogenous variables and instruments.
- **Logit**: Validates binary choice modeling and associated metrics (Pseudo R²).
- **VAR / VECM**: Tests multivariate time-series modeling, including cointegration tests and lag selection.
- **Diagnostics**: Includes tests for Breusch-Pagan (heteroskedasticity), Jarque-Bera (normality), and Durbin-Watson (autocorrelation).

### 2. Pipeline Orchestration (`tests/test_pipeline.py`)
- **End-to-End Flow**: Validates the `Pipeline` class's ability to orchestrate stages from data resampling to model fitting.
- **Configuration Integration**: Ensures that parameters from `params.yaml` are correctly propagated throughout the system.

### 3. Featurization (`tests/test_feature_engineering.py`, `tests/test_data_merger.py`)
- **Data Transformation**: Tests log, lag, and percentage change transformations across panel dimensions.
- **Merging**: Validates the robust merging of disparate datasets with different frequencies and geographical coverage.

### 4. Visualization (`tests/test_visualization.py`)
- **Automated Reporting**: Ensures that visualizers for regression summaries and EDA produce correct plots and save them to the designated directories.

### 5. Utility Functions (`tests/test_config.py`, `tests/test_dict_country_converter.py`)
- **Country Standardization**: Validates the conversion of various country name formats to standardized ISO codes.
- **Config Management**: Tests YAML loading and parameter mapping.

## Mock Data

Tests utilize generated mock datasets located in `tests/data/` (e.g., `test_data.csv`) or generated on-the-fly to ensure reproducibility without requiring large external dependencies.
