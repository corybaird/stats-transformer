# Agentic Sanity Checks

This document catalogs the automated sanity checks used to validate the `stats-transformer` library's transformations against original, paper-specific replication scripts.

## Monetary Policy Transformations (`apep_0235`)

**Script**: `src/examples/featurization/agent_sanity_monetary.py`
**Source Repo**: [SocialCatalystLab/ape-papers (apep_0235)](https://github.com/SocialCatalystLab/ape-papers/tree/70c660ea5a7de722e9a12ad965e1c665890208d8/apep_0235/v1/code)

This test ensures `stats-transformer` correctly recreates the transformations found in the `apep_0235` macroeconomic study.

### Validated Transformations
* **Inflation (CPIAUCSL)**: 12-month percentage change.
* **IP Growth (INDPRO)**: 12-month percentage change.
* **Forward Employment (PAYEMS)**: Log forward difference for local projections ($h=1$).

### Verification Logic
* **Raw Data Phase**: Fetches `CPIAUCSL`, `INDPRO`, and `PAYEMS` directly from the FRED API.
* **Original Script Phase**: Replicates the chained `.pct_change(12)` and `.shift()` operations from the `apep_0235` Python script.
* **Stats-Transformer Phase**: Uses `FeatureEngineer` building blocks (`lag1`, `lag12`, `lead1`, `log`) to process the raw data cleanly.
* **Comparison Phase**: Merges the two outputs and verifies absolute identical output.

**Result**: Output exactly aligns with the original replication code, proving `stats-transformer` can handle complex macroeconomic feature engineering natively.

---
*(Add future tests here)*
