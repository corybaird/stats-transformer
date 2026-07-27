# MATLAB cross-language comparator

`src/examples/academic/var/matlab_comparator.py` compares the Blanchard--Quah long-run structural impact matrix produced by `stats-transformer` with the result from Ambrogio Cesa-Bianchi's MATLAB VAR-Toolbox. `tests/verification/matlab_comparator.py` remains a compatibility wrapper around the same class.

## Contents

1. [Software environment](#software-environment)
2. [What is compared](#what-is-compared)
3. [Prerequisites](#prerequisites)
4. [Run the comparison](#run-the-comparison)
5. [Reusing the comparator](#reusing-the-comparator)

It is an opt-in local verification utility. It is not collected by the default `pytest` suite because it requires proprietary MATLAB software and a separate MATLAB Engine installation.

## Software environment

The following local installations were inspected on 2026-07-24:

| Component | Recorded version | Role in this repository |
| --- | --- | --- |
| MATLAB | R2025b Update 4, version `25.2.0.3150157` | Runs the optional MATLAB Engine comparison. |
| VAR-Toolbox | 4.0 | External MATLAB implementation used for the Blanchard--Quah impact-matrix check. |
| Kilian & Lütkepohl (2017) | SVAR Code | External MATLAB scripts used for verifying structural VAR implementations. |
| Dynare | 7.1, Apple Silicon package | Available for DSGE work, but not currently called by an example, test, or comparator. |

Dynare 7.1 documents compatibility with MATLAB R2025b. The Kilian SVAR test suite (`tests/replication/kilian/test_svar.py`) also uses MATLAB R2025b. This records computational provenance only: Dynare availability does not imply a Dynare model has been validated by `stats-transformer`.

When adding a Dynare-based benchmark, record the `.mod` file, solver options, MATLAB or Octave runtime, Dynare release, calibration or data, the object compared, tolerance, and observed discrepancy alongside that benchmark.

## What is compared

The comparator uses the bundled `data/examples/matlab_examples/BQ1989_Data.xlsx` data and estimates:

- Two variables: GDP growth and unemployment.
- A VAR with eight lags and a constant.
- Blanchard--Quah long-run identification.
- Bootstrap inference disabled, because the compared object is the point-estimate impact matrix.

It asserts equality of the Python and MATLAB impact matrices within relative and absolute tolerance `1e-10`. On the verified local comparison, the maximum absolute difference was $2.22 \times 10^{-16}$.

This validates one numerical object for one model specification. It does not validate bootstrap intervals, figures, all structural identification schemes, proxy SVAR, sign restrictions, or local projections.

## Prerequisites

1. A licensed local MATLAB installation.
2. MATLAB Engine for Python installed into the active Python environment.
3. A local checkout of [VAR-Toolbox](https://github.com/ambropo/VAR-Toolbox).
4. The repository's development environment, including `openpyxl` for the bundled spreadsheet.

Set `VAR_TOOLBOX_DIR` to the root of the local VAR-Toolbox checkout. The comparator adds that directory and its subdirectories to the MATLAB path for the session.

## Run the comparison

From the repository root:

```bash
VAR_TOOLBOX_DIR=/absolute/path/to/VAR-Toolbox /opt/homebrew/bin/uv run python -m src.examples.academic.var.matlab_comparator
```

On success, the script reports the maximum absolute difference. A missing MATLAB Engine or toolbox path returns `matlab_connected: false`; a numerical disagreement raises an assertion error.

## Reusing the comparator

The `MATLABComparator` class can be constructed with an explicit toolbox path when calling it from another Python module:

```python
from src.examples.academic.var.matlab_comparator import MATLABComparator

result = MATLABComparator("/absolute/path/to/VAR-Toolbox").run()
```

Use a separate comparator and a documented benchmark for every additional model. Numerical equivalence should be checked only after matching the sample, transformations, deterministic terms, variable ordering, identification, normalization, and inference settings.
