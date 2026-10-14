# MATLAB reference outputs

Recorded from MATLAB R2025b (25.2.0.3150157 Update 4) on the host with the shared engine. No stats-transformer code is imported.

- Rerun: `/opt/homebrew/bin/uv run python -m tests.external_benchmarks.matlab.record_matlab_refs` from the repo root, with a MATLAB engine already shared (`matlab.engine.shareEngine`)
- One `<case>.json` per case, one driver per case in `drivers/st_ref_<case>.m`, case table in `cases.yaml`
- A failing case writes `<case>.error.txt` instead of JSON
- Toolbox: VAR-Toolbox 4.0 from `references/matlab_benchmarks/` (gitignored, not vendored). JSON stores SHA-256 of every toolbox or Kilian and Luetkepohl `.m` actually executed (Kilian and Luetkepohl keys are repo-relative) (found with the MATLAB profiler) plus the six files the task lists, so `VARmakelags.m` is hashed although these cases never call it
- Settings mirror `GO_BQ1989.m`, `GO_GK2015.m`, `GO_SW2001.m` (data load, lags, ident, horizons) with `inference = 0`; plotting, `cd`, `clear` and the bootstrap options are omitted
- Data are read from `data/examples/matlab_examples/*.xlsx`, byte-identical to the copies under `Replic/`; sentinel 123456789 mapped to NaN with `Num2NaN` as in GO_GK2015
- `inputs.ENDO` (and `inputs.IV`) are the exact matrices passed to `VARmodel`; the recorder asserts them against an independent pandas read of the sheet
- Array layouts are in each JSON `layout` block. Everything is column-major MATLAB order converted to nested lists, `IR[h][i][j]` is variable i, shock j; `VD[h][j][i]` is shock j, variable i
- `ols_longley`: `X'X` has condition number about 1e20, so `inv(X'X)` is built from the QR factor (`inv(R)*inv(R)'`). `beta` matches the NIST certified Longley values to the printed digits
- `gk2015_iv`: the toolbox zeroes columns 2 to n of `B` and slices 2 to n of `IR` and `VD`; only column 1 is identified. 126 of 396 instrument values are NaN and the first stage uses the common sample
- `engine.toolboxes` lists products whose files were executed, not licensed products
- Batch 2 (`generic_cases` in `cases.yaml`): `svar_keating` and `bvar_kl_minnesota` from the Kilian and Luetkepohl (2017) code copied to `tests/external_benchmarks/matlab/third_party/kilian_lutkepohl_2017/` (tracked; only the two folders these cases use), `vecm_macro` and `arima_realgdp` from the Econometrics Toolbox, `lp_jt2025` and `lpiv_jt2025` from `Replic/JT2025/GO_JT2025.m`
- `svar_keating` tightens the book's fsolve tolerance from 1e-4 to 1e-12 and adds a zero-restriction variant
- Deviation from the repo config rule: the case table is `cases.yaml` beside the script because the task restricts writes to this folder
