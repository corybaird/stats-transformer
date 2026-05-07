# Standardized Research File Structure

In the "Agentic AI Age," a standardized folder structure is critical. It enables AI agents to navigate the project autonomously, maintains clear boundaries between code and data, and ensures reproducibility. This standard is based on the **Cookie Cutter Data Science** pattern, refined for macroeconomic research.

## 1. The Academic Anti-Pattern

Traditional academic research folders (like those from Nakamura & Steinsson or Jarociński & Karadi) often suffer from:
*   **Monolithic Folders**: Mixing data, scripts, and results in a single `Stata/` or `Programs/` directory.
*   **Hardcoded Paths**: Relative or absolute paths that break on other machines.
*   **Manual Steps**: "Run X then copy Y to Z" workflows that are impossible for agents to automate reliably.
*   **Flat Data Structures**: Generic folders like `Data_Orig/` or `external/` that provide no semantic context.

## 2. Modern Reproducible Project Structure

This structure separates **Configuration** (the "brain"), **Source Code** (the "muscles"), and **Data** (the "memory").

```text
.
├── data/              # Immutable and derived data (DVC Managed)
│   ├── raw/           # Original, untouched source files
│   ├── text/          # Central bank statements, narratives, scraped text
│   ├── shocks/        # Monetary policy surprises, uncertainty indices
│   ├── indicators/    # Macro fundamentals (CPI, GDP, Interest Rates)
│   ├── financial/     # Market data (Equities, CDS, Yield Curves)
│   ├── intermediate/  # Intermediate artifacts (merged panels, resampled data)
│   └── final/         # Final feature-engineered datasets for modeling
├── docs/              # Architecture, file structures, and methodology
├── models/            # Serialized model artifacts (.pkl, .joblib, .pt)
├── notebooks/         # Exploratory analysis and paper-specific pipelines
├── references/        # CONFIGURATION CENTER
│   ├── configs/       # YAML files (the Single Source of Truth)
│   ├── dictionaries/  # Mapping files (country converters, regional groups)
│   └── prompts/       # LLM templates and DSPy signatures
├── reports/           # AUTOMATED OUTPUTS
│   ├── figures/       # Charts and plots (routed to Overleaf)
│   ├── tables/        # Regression results and summary stats
│   └── visualizations/ # EDA and interactive dashboards
└── src/               # REUSABLE MODULES
    ├── data/          # Ingestion and ETL logic
    ├── featurization/ # Transformation and engineering
    ├── models/        # Econometric and ML implementations
    └── viz/           # Plotting and reporting logic
```

## 3. Core Directory Principles

### `data/` (Macro-Specific Segregation)
Avoid generic names like `external`. Use descriptive subfolders to define the data domain:
*   **`data/text/`**: Stores the unstructured "narrative" data.
*   **`data/shocks/`**: Stores the high-frequency "exogenous" surprises.
*   **`data/indicators/`**: Stores the low-frequency "fundamentals."
*   **`data/financial/`**: Stores the high-frequency "market reactions."

### `references/` (The Configuration Center)
Never hardcode parameters. The `references/configs/params.yaml` file acts as the single source of truth. If you want to change a transformation lag or a country list, you edit the YAML, not the Python.

### `src/` (The Engine)
Code in `src/` should be stateless and modular. It doesn't "know" about specific papers; it only knows how to process data given a configuration.

### `reports/` (The Overleaf Bridge)
Output logic should route directly into `reports/`. This allows for seamless integration with LaTeX documents via `git submodules` or `rclone` syncs to Overleaf.

## 4. DVC Integration

Large datasets and LLM output caches should never be in Git. Use **DVC (Data Version Control)** to track files in `data/` and `models/`.
*   **Git**: Tracks `.py`, `.yaml`, `.md`, and `.dvc` pointers.
*   **DVC**: Tracks `.parquet`, `.csv`, and `.jsonl` data files.

## 5. Workflow Summary

1.  **Initialize**: Define data sources and transformations in `references/configs/params.yaml`.
2.  **Ingest**: Use `src/data/` to pull and clean data into `data/intermediate/`.
3.  **Featurize**: Apply transformations using `src/featurization/` into `data/final/`.
4.  **Model**: Run regressions in `notebooks/` using `src/models/`.
5.  **Report**: Automatically generate outputs into `reports/` for publication.
