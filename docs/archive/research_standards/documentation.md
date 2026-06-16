# Standardized Research Documentation

Just as code and data require strict organization, the documentation of a macroeconomic research project must follow agreed-upon design patterns. Good documentation bridges the gap between the computational pipeline (Python/DVC) and the final academic output (LaTeX/PDF).

Based on the documentation patterns from the `shocks` project, the following is the standard practice for documenting research projects in the Agentic AI age.

## 1. The Core Documentation Suite

Every research project should contain a `docs/` folder (or `notebooks/[project]/docs/`) with the following separated documents. **Do not create monolithic README files.**

### A. `reproduction.md` (The Operations Manual)
**Purpose**: The quick-start guide for anyone (or any AI agent) trying to replicate the paper from scratch.
**Required Content**:
*   **Data Retrieval**: Explicit `dvc pull` commands.
*   **Execution Commands**: The exact `uv run python -m ...` commands.
*   **Config Reference**: A brief explanation of the primary `config.yaml` or `params.yaml` that drives the pipeline.
*   *Improvement from past projects*: Keep this file strictly operational. Do not explain *why* the data is used here, only *how* to run the code.

### B. `data_architecture.md` (The Blueprint)
**Purpose**: Defines the data lineage, from raw sources to the final analytical panel.
**Required Content**:
*   **Source Inventory**: A table of raw datasets, their sources (FRED, Haver, specific papers), and frequencies.
*   **Pipeline Stages**: A high-level view of how `data/raw/` becomes `data/final/`.
*   **Feature Dictionary**: A clear mapping of academic variables to DataFrame columns (e.g., $KAOPEN_{i,t}$ $\rightarrow$ `ka_open`).
*   *Improvement from past projects*: Do not include Exploratory Data Analysis (EDA) or country-overlap deep dives here. Move those to a separate `eda_report.md` or a Jupyter notebook. Architecture docs must remain concise.

### C. `econometrics.md` (The Theoretical Framework)
**Purpose**: The bridge between the code and the academic paper's methodology section.
**Required Content**:
*   **Hypotheses**: Clear statements of what is being tested.
*   **Equations**: Explicit LaTeX formulas for all regression specifications (Baseline, Stage 1, Local Projections) and theoretical models (e.g., DSGE structure).
*   **Variable Roles**: Defining Dependent Variables, Instruments, Controls, and Fixed Effects.
*   *Why this matters*: Separating the math from the code allows an agent or researcher to verify if the `linearmodels.PanelOLS` implementation actually matches the intended theoretical equation.

### D. `outputs_mapping.md` (The LaTeX Bridge)
**Purpose**: To stop the confusion of matching generated plot files to figures in the draft.
**Required Content**:
*   A strict mapping table that connects the Python output to the Paper's figure number.

| Generated File | Source Data | LaTeX Label | Paper Figure # |
| :--- | :--- | :--- | :--- |
| `figure_4_total_statements.pdf` | `data/intermediate/statements.parquet` | `\label{fig:total_statements}` | Figure 4 |
| `figure_16_granger.pdf` | `data/final/lexical_sentiment.parquet` | `\label{fig:granger_cause}` | Figure 16 |

*   *Improvement from past projects*: This mapping is critical for automated manuscript compilation. It allows a researcher to completely rerun the pipeline and trust that Overleaf will compile correctly without manual checking.

## 2. Documentation Anti-Patterns to Avoid

*   **The "Brain Dump" README**: Putting econometric equations, DVC commands, and data source links into one massive `README.md`. It becomes unsearchable and intimidating.
*   **Code in Methodology Docs**: Do not paste Python snippets into `econometrics.md`. Keep the math and theory independent of the implementation.
*   **Stale Variable Names**: When you rename a column from `mps_cws` to `mps_baseline` in the code, the documentation must be updated simultaneously. 

## 3. The "Docs-as-Code" Philosophy

Documentation should be treated as code. If a pipeline stage changes, the corresponding `data_architecture.md` must be updated in the same commit. By adhering to this 4-file structure (`reproduction`, `data_architecture`, `econometrics`, `outputs_mapping`), both human researchers and AI agents can seamlessly parse, debug, and expand upon macroeconomic research.
