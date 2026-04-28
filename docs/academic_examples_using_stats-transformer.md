# Academic Validation Suite

This document describes the sanity check scripts used to validate the `stats-transformer` library against established macroeconomic research. These tests ensure that our library's featurization logic perfectly replicates the data processing used in top-tier academic journals.

## 1. Nakamura & Steinsson (2018)
**Script:** `src/examples/featurization/nakamura_steinsson_sanity.py`

*   **Logic:** Validates daily first-differencing of nominal yields.
*   **Ground Truth:** Compared against official Stata replication data (`master.dta`).
*   **Key Finding:** Replicated a $10^{-8}$ precision match across 4,700+ observations. Successfully identified specific manual date overrides where researchers applied expert judgment.

## 2. Bauer & Swanson (2023)
**Script:** `src/examples/featurization/bauer_swanson_sanity.py`

*   **Logic:** Validates log-level transformations followed by differencing ($ \ln(X_t) - \ln(X_{t-1}) $) and time-series lagging.
*   **Ground Truth:** Benchmarked against the original Matlab implementation (`loadmacrodata.m`).
*   **Key Finding:** Confirmed that chaining transformations in `stats-transformer` produces a $0.000$ error parity with Matlab's numerical processing.

## 3. Bauer, Bernanke, & Milstein (2023)
**Script:** `src/examples/featurization/bauer_bernanke_milstein_sanity.py`

*   **Logic:** Validates standard raw differences (`changeraw`) and percentage changes (`changepct`) for high-frequency financial indices.
*   **Ground Truth:** Compared against the paper's native Python preprocessing scripts (`pull_and_process_data.py`).
*   **Key Finding:** Achieved perfect mathematical parity (zero error) while demonstrating that the library's declarative configuration is significantly more concise than manual Pandas manipulation.

## 4. API Sanity Checks
**Scripts:** 
*   `src/examples/featurization/fred_sanity.py`
*   `src/examples/featurization/dbnomics_sanity.py`

*   **Logic:** Validates transformations against live data from FRED and DBnomics APIs.
*   **Purpose:** Ensures the library remains compatible with real-time data providers and handles various data structures correctly.

---

### Usage
To run the full validation suite, execute the scripts from the project root:
```bash
/opt/homebrew/bin/uv run python -m src.examples.featurization.nakamura_steinsson_sanity
/opt/homebrew/bin/uv run python -m src.examples.featurization.bauer_swanson_sanity
/opt/homebrew/bin/uv run python -m src.examples.featurization.bauer_bernanke_milstein_sanity
```
