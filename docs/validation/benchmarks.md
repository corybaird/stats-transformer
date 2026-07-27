# Cross-Language Verification Benchmarks

This registry documents every numerical benchmark implemented in the `stats-transformer` test suite to ensure computational accuracy against established academic implementations.

These tests are located in `tests/verification/` and are automatically executed via `pytest`. They use the `matlabengine` Python library to bridge data between Python and local MATLAB installations.

| Model / Feature | Python Class | Academic Reference | Reference Script Path | Status |
|---|---|---|---|---|
| **Recursive SVAR (Cholesky)** | `SVARModel` | Kilian (2017) Ch 9 | `kilian_2017/Code_Kilian/9/figure9_1_chol` | ✅ Verified |
| **VECM (Johansen MLE)** | `VECMModel` | Kilian (2017) Ch 3 | `kilian_2017/Code_Kilian/3/mle_unknown_beta` | ✅ Verified |
| **Long-Run Restrictions (BQ)** | `BlanchardQuahModel` | VAR-Toolbox (BQ 1989) | `VAR/Replic/BQ1989/` | ✅ Verified |
| **Proxy SVAR (External IV)** | `ProxySVARModel` | VAR-Toolbox (GK 2015) | `VAR/Replic/GK2015/` | ✅ Verified |
| **Local Projections (LP-OLS)** | `LocalProjectionsModel` | VAR-Toolbox (JT 2025) | `VAR/Replic/JT2025/` | ✅ Verified |

## Running the Benchmarks
To run the benchmarks locally, ensure you have MATLAB installed and the `matlabengine` package available in your Python environment:

```bash
uv run pytest tests/verification/ -v
```

If `matlabengine` is missing, the PyTest suite will automatically `SKIP` these benchmarks to prevent CI/CD failures on agents lacking proprietary licenses.

## Directory Structure
- `tests/verification/kilian/`: Benchmarks comparing against Kilian & Lütkepohl (2017) MATLAB scripts.
- `tests/verification/var_toolbox/`: Benchmarks comparing against Cesa-Bianchi's VAR-Toolbox (4.0).
- `tests/verification/models/`: Model-centric test wrappers that import the author-specific assertions, allowing you to run `pytest tests/verification/models/test_svar.py` to trigger all SVAR-related benchmarks across multiple authors simultaneously.
