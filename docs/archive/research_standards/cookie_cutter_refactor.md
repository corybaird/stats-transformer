# The Cookie Cutter Refactor Roadmap

## The Problem
Currently, the `stats-transformer` repository serves a dual purpose:
1. It is a pip-installable **Python library** (`src/stats_transformer/`).
2. It attempts to act as a **reference project template** for macroeconomic research (the `data/`, `reports/`, and `references/` directories).

These two concerns conflict. A library is organized by *capability* (e.g., `models/`, `featurization/`, `visualization/`). A research project is organized by *workflow stage* (e.g., `data/`, `pipeline/`, `model/`, `viz/`). 

## The Option C Solution: A Separate Template Repo
To solve this, we will eventually implement **Option C**: extracting the project scaffolding into a dedicated template repository (e.g., `cookiecutter-econ-research`).

This new repository will be the canonical starting point for all future economic research projects. It will use the `cookiecutter` python package to generate identical, standardized directory structures that conform strictly to the Global Rules, while relying on `stats-transformer` merely as an external dependency.

## The Target Research Project Structure
When the cookiecutter template is built, it will generate the following structure for new papers:

```text
my-research-paper/
├── data/
│   ├── raw/                               # Immutable original data
│   ├── processed/                         # Cleaned data
│   ├── pipeline/                          # DVC-managed intermediate artifacts
│   └── temp/                              # Scratch space
├── docs/
│   ├── reproduction.md                    # Operations manual
│   ├── data_architecture.md               # Data lineage blueprint
│   ├── econometrics.md                    # Theoretical framework
│   └── outputs_mapping.md                 # LaTeX bridge
├── models/                                # Serialized model artifacts
├── notebooks/                             # Exploratory analysis
├── references/
│   ├── configs/
│   │   └── params.yaml                    # Pre-wired for stats-transformer
│   ├── dictionaries/                      # Mapping files
│   └── prompts/                           # LLM templates
├── reports/
│   ├── figures/                           # Publication plots
│   ├── tables/                            # Regression tables
│   └── visualizations/                    # EDA dashboards
├── src/
│   ├── data/                              # ETL and ingestion
│   ├── model/                             # Project-specific model logic (singular)
│   ├── pipeline/                          # Pipeline orchestration
│   ├── llm/                               # LLM inference
│   ├── viz/                               # Plotting utilities (short form)
│   ├── dashboard/                         # Streamlit/interactive UI
│   └── utils/                             # Shared helpers
├── tests/                                 # Unit and integration tests
├── .agents/                               # Agent workflows (commit, branch, etc.)
├── dvc.yaml                               # Pre-wired DVC pipeline
└── pyproject.toml                         # Lists stats-transformer as a dependency
```

## Why This Matters
1. **Separation of Concerns:** `stats-transformer` can focus on being an excellent, robust econometric library.
2. **Instant Compliance:** New papers instantly inherit the 4-file documentation suite, the DVC wiring, and the exact naming conventions required by our workflows.
3. **Consistency:** Avoids the organic sprawl where some papers use `data/panel/` and others use `data/final/`. The template acts as the ultimate arbiter of structure.
