# Academic & Numerical Validation Guide

This document details the academic benchmarks, numerical verification procedures, paper transformation examples, and cross-language MATLAB comparator tools in `stats-transformer`.

## Table of Contents

1. [1. Evidence Hierarchy](#1-evidence-hierarchy)
2. [2. Direct Python Estimator Comparisons](#2-direct-python-estimator-comparisons)
3. [3. Academic Feature Transformation Examples](#3-academic-feature-transformation-examples)
4. [4. Structural VAR & Local Projections Demonstrations](#4-structural-var--local-projections-demonstrations)
5. [5. Cross-Language MATLAB Comparator](#5-cross-language-matlab-comparator)
   - [5.1 Software Environment & Provenance](#51-software-environment--provenance)
   - [5.2 Numerical Specification & Comparison Object](#52-numerical-specification--comparison-object)
   - [5.3 Prerequisites & Execution](#53-prerequisites--execution)
6. [6. Protocol for Reporting Validation Claims](#6-protocol-for-reporting-validation-claims)

---

## 1. Evidence Hierarchy

Validation in `stats-transformer` distinguishes four distinct evidence tiers:

1. **Automated Unit & Integration Tests**: Controlled pytest routines verifying package APIs, transformations, and output data structures.
2. **Execution Checks**: Sample scripts confirming that specialized models fit bundled datasets without throwing errors.
3. **Direct Python Comparisons**: Array-level verification comparing library wrappers against `statsmodels` or `linearmodels` on identical data.
4. **Cross-Language Numerical Comparisons**: Machine-precision numerical validation comparing Python estimates against external implementations like Ambrogio Cesa-Bianchi's MATLAB VAR-Toolbox.

> [!IMPORTANT]
> An example script that executes successfully is a runnable demonstration. It constitutes numerical parity only when data, lag structures, ordering, restrictions, and outputs match a stated external benchmark within a documented tolerance.

---

## 2. Direct Python Estimator Comparisons

`src/examples/timeseries/macro_var.py` prepares real GDP, real consumption, and real investment from the quarterly `statsmodels` macroeconomic dataset using $100 \Delta \log(x_t)$ transformations. It estimates a VAR(2) with a constant directly through `statsmodels` and through `VARModel`.

```bash
/opt/homebrew/bin/uv run python -m src.examples.timeseries.macro_var
```

The script asserts numerical equality of coefficient matrices and standard errors across both implementations.

---

## 3. Academic Feature Transformation Examples

The following scripts verify that `FeatureEngineer` transformations match paper-specific processing routines and supplied replication data:

| Example Paper | Execution Module | Target Benchmark | Scope of Transformation |
| --- | --- | --- | --- |
| **Nakamura & Steinsson (2018)** | `src.examples.academic.nakamura_steinsson` | Supplied Stata `master.dta` output | Daily first difference of nominal yield curve shocks |
| **Bauer & Swanson (2023)** | `src.examples.academic.bauer_swanson` | MATLAB log-difference and lag logic | Monthly high-frequency monetary surprise series |
| **Bauer, Bernanke, & Milstein (2023)** | `src.examples.academic.bauer_bernanke_milstein` | Python reference calculations | Daily financial variable difference & % change logic |

Run these academic demonstrations from the repository root:

```bash
/opt/homebrew/bin/uv run python -m src.examples.academic.nakamura_steinsson
/opt/homebrew/bin/uv run python -m src.examples.academic.bauer_swanson
/opt/homebrew/bin/uv run python -m src.examples.academic.bauer_bernanke_milstein
```

---

## 4. Structural VAR & Local Projections Demonstrations

The `src/examples/academic/var/` directory contains structural time-series demonstrations based on bundled VAR-Toolbox datasets:

- **Stock & Watson (2001)** (`stock_watson_2001.py`): 3-variable reduced-form VAR demonstration.
- **Blanchard & Quah (1989)** (`blanchard_quah_1989.py`): Long-run structural identification.
- **Gertler & Karadi (2015)** (`gertler_karadi_2015.py`): External-instrument SVAR (Proxy SVAR / SVAR-IV).
- **Jordà & Taylor (2025)** (`jorda_taylor_2025.py`): Instrumental-variables local projections (LP-IV).

---

## 5. Cross-Language MATLAB Comparator

The MATLAB comparator in `src/examples/academic/var/matlab_comparator.py` (and wrapper `tests/verification/matlab_comparator.py`) provides an opt-in cross-language verification tool against Ambrogio Cesa-Bianchi's MATLAB VAR-Toolbox.

### 5.1 Software Environment & Provenance

| Component | Verified Version | Role in Verification Suite |
| --- | --- | --- |
| **MATLAB** | R2025b Update 4 (v25.2.0) | Local MATLAB Engine execution runtime |
| **VAR-Toolbox** | 4.0 | Benchmark MATLAB implementation of structural VAR methods |
| **Dynare** | 7.1 (Apple Silicon) | Recorded for environment provenance (not called by tests) |

### 5.2 Numerical Specification & Comparison Object

The comparator uses the bundled `data/examples/matlab_examples/BQ1989_Data.xlsx` dataset and estimates:
- **System Variables**: GDP growth and unemployment rate ($K=2$).
- **Lag Structure**: VAR(8) with an intercept.
- **Identification Scheme**: Blanchard & Quah (1989) long-run structural restrictions ($C(1)$ lower-triangular).

####Discrepancy Discrepancy
- **Tolerance Criterion**: Absolute and relative tolerance set to $1.0 \times 10^{-10}$.
- **Observed Discrepancy**: Maximum absolute difference between Python and MATLAB structural impact matrices: **$2.22 \times 10^{-16}$** (machine precision).

### 5.3 Prerequisites & Execution

Executing the MATLAB comparator requires:
1. Licensed local MATLAB installation.
2. `matlabengine` installed in the active Python virtual environment (`.venv`).
3. Local checkout of [VAR-Toolbox](https://github.com/ambropo/VAR-Toolbox).

Execute the comparison by pointing `VAR_TOOLBOX_DIR` to the local VAR-Toolbox path:

```bash
VAR_TOOLBOX_DIR=/path/to/VAR-Toolbox /opt/homebrew/bin/uv run python -m src.examples.academic.var.matlab_comparator
```

Alternatively, invoke `MATLABComparator` programmatically:

```python
from src.examples.academic.var.matlab_comparator import MATLABComparator

result = MATLABComparator("/path/to/VAR-Toolbox").run()
print("Max absolute difference:", result["max_abs_diff"])
```

---

## 6. Protocol for Reporting Validation Claims

When publishing or reporting empirical validation results, always document:
1. Exact dataset name, version, and date of download.
2. Feature transformations and sample adjustment rules.
3. Estimator specification (lags, deterministic terms, identification constraints).
4. Compared object (coefficients, standard errors, structural impact matrix, IRF paths).
5. Numerical tolerance and observed maximum absolute discrepancy.
