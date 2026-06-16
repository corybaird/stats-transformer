# stats-transformer Repository Structure

This document outlines the file structure of the `stats-transformer` repository. As a general-purpose library, this repository is organized primarily by **capabilities** rather than paper-specific workflows.

## 1. Directory Tree

```text
.
├── data/                          # Immutable and derived data (gitignored)
│   ├── raw/                       # Original, untouched source files
│   │   └── examples/              # Data for sanity checks and unit tests
│   ├── final/                     # Feature-engineered datasets for modeling
│   ├── pipeline/                  # Intermediate merged artifacts
│   └── temp/                      # Temporary scratch space
├── docs/                          # Architectural and user documentation
├── models/                        # Serialized model artifacts (.pkl, .joblib)
├── notebooks/                     # Exploratory analysis and demonstrations
├── references/                    # CONFIGURATION CENTER
│   ├── configs/                   # YAML configuration files
│   │   └── examples/              # YAML configs for validation tests
│   ├── dictionaries/              # Mapping files (e.g., country codes)
│   └── prompts/                   # LLM templates and DSPy signatures
├── reports/                       # AUTOMATED OUTPUTS
│   ├── figures/                   # Charts and plots
│   ├── tables/                    # Regression results and summary stats
│   └── visualizations/            # EDA and tracking visualizations
├── src/                           # SOURCE CODE
│   ├── examples/                  # Validation scripts and sanity checks
│   ├── stats_transformer/         # The core Python library package
│   │   ├── data/                  # Data ingestion utilities
│   │   ├── featurization/         # Feature engineering and merging
│   │   ├── models/                # Econometric and ML implementations
│   │   ├── utils/                 # Shared helpers
│   │   ├── visualization/         # Plotting logic
│   │   └── pipeline.py            # Main YAML-driven orchestrator
│   └── temp/                      # Scratch/debug scripts
└── tests/                         # Unit and integration tests
```

## 2. Core Directory Principles

### `src/stats_transformer/` (The Tool/Library)
This is the core Python package. It is organized by *capability* (e.g., `models/`, `featurization/`, `visualization/`). Code here should be stateless, modular, and unaware of specific papers or datasets. It is driven purely by configuration.

### `data/` (The Data Warehouse)
The `data/` directory is excluded from Git (`.gitignore`). It follows a strict data lifecycle:
*   **`raw/`**: Where immutable data lives.
*   **`pipeline/`**: For intermediate, merged, or resampled data before final featurization.
*   **`final/`**: The ultimate analytical datasets fed into models.
*   **`temp/`**: For temporary scratch files.

### `references/configs/` (The Control Center)
The `stats-transformer` pipeline relies heavily on YAML configuration. `references/configs/` stores these YAML files, enabling you to change datasets, transformation lags, or model specifications without touching Python code. The `examples/` subdirectory contains configurations used to validate the library against academic benchmarks.

### `reports/` (Automated Outputs)
Running models or the main `pipeline.py` will automatically dump artifacts (JSON summaries, PNG coefficient plots) into the `reports/` directory.

## 3. Creating a New Research Project

The structure above describes the `stats-transformer` library repository. Individual research projects that use the package can keep their own paper-specific data, reports, notebooks, and manuscript files outside this package repository.
