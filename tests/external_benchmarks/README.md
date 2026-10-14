# External reference outputs

Recorded outputs from MATLAB, R and Stata. None of the recorders import stats-transformer, so the files are independent of the code they check.

- `matlab/`: MATLAB R2025b with VAR-Toolbox 4.0, Kilian and Luetkepohl (2017) book code and the Econometrics Toolbox, 11 cases (see `matlab/README.md`)
- `r/`: R 4.6.1 in Docker image `st-ref-r:1` (`r/Dockerfile`, apt `r-cran-*` plus CRAN `vars`), 13 cases
- `stata/`: StataNow 19.5 MP on the host, 14 cases
- Rerun: `uv run python -m tests.external_benchmarks.<matlab.record_matlab_refs | r.record_r_refs | stata.record_stata_refs>` from the repo root; build the image first with `docker build -t st-ref-r:1 tests/external_benchmarks/r`
- Each JSON holds `engine`, `data` (path, sha256, row count), `settings`, `outputs`, `labels` (row and column names as the software reported them), `layout`, `driver` and its hash, timestamp and host
- Floats are written at 17 significant digits (R) or `%24.17g` (Stata), so they round-trip exactly
- Conventions differ between programs and are recorded in `layout`: constant first (R, MATLAB) or last (Stata), IV instrumented variable first in Stata, `N` versus `N-k` divisors, ML versus degrees-of-freedom covariance, variable-major versus lag-major VAR coefficient order
- Data are the repo files in `data/examples`; `bq1989_long` is read from the xlsx and written to a temporary CSV for R and Stata
- Cases with `input_from` read their input matrix from a MATLAB JSON (`inputs`), so all programs use byte-identical data
- Cross-program checks done when recording: VECM H1 and H1* agree across MATLAB, Stata and R; ARIMA agrees between Stata and R only (MATLAB uses a conditional likelihood); SVAR Canada agrees between Stata and R up to the documented T versus T-k scale
- `matlab/svar_keating` is MATLAB-only because Stata `svar` and R `vars::SVAR` did not converge on that model
- `matlab/bvar_kl_minnesota` uses the Litterman fixed-Sigma Minnesota prior, not the natural-conjugate prior in `BVARModel`
- `matlab/lp_jt2025` and `lpiv_jt2025` use long differences and lagged controls; `stata/lp_plain_jt2025` and `lpiv_plain_jt2025` run the regression the wrappers actually estimate
- A failing case writes `<case>.error.txt`
