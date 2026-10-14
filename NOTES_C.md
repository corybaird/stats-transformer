# NOTES_C — VECM cointegration rank + SVEC optimizer fallback

Branch: `fix/vecm-rank-svec-optimizer` (based on `origin/main` 57394fa).

## Commits

1. `7aceb5c` FIX honour coint_rank in VECMModel and un-xfail rank-3 UK parity test
   - `src/.../reduced_form/vecm.py`: new `coint_rank=1` constructor arg, YAML `model.coint_rank` (same precedence as `k_ar_diff`), passed into `VECM(...)` in `build_model`. Default 1 keeps existing behaviour identical.
   - `tests/test_external_parity/test_wrapper_parity.py`: removed the strict xfail on `test_vecm_ghysels_ch7_uk_rank3_matches_r`; it now compares the full rank-3 `beta` (levels + `det_coef_coint` row, shape 5x3) and `alpha` (transposed, `alpha.T`) against R's recorded `uk_r3_beta` / `uk_r3_alpha` at `atol=1e-8`.
2. `9d4345c` FIX replace SVEC unconstrained Nelder-Mead fallback with multi-start SLSQP and restriction check
   - `src/.../structural/svec.py`: removed the unconstrained Nelder-Mead fallback. SLSQP is retried from deterministic starts (Cholesky factor of `sigma_u`, then scaled identity `diag(sqrt(diag(sigma_u)))`). If every attempt fails, or the fitted B violates the identifying restrictions beyond 1e-6, a `RuntimeError` is raised (no silent restriction-violating B). Violation is exposed as `SVEC.max_restriction_violation` (and `SVEC.restr_tol`).
3. `fb45199` DOCS update VECM rank and SVEC parity status in benchmarks and models docs
   - `docs/validation/benchmarks.md`: VECM verified row now mentions rank 3; removed the "VECMModel rank above 1" xfail row, the "R svec_canada not yet usable" row, the "SVEC Nelder-Mead fallback" bug row and the §2.2.4 SVEC future-cross-check row; added an SVEC verified row (Canada, `10^{-4}`).
   - `docs/library/models.md`: updated §2 summary rows and §5.2/§6.9 (rank fixed at 1 claim removed, `coint_rank` documented, SVEC verified discrepancy added).

## New tests

- `test_vecm_ghysels_ch7_uk_rank3_matches_r` (un-xfailed): statsmodels normalises the top `r` rows of `beta` to the identity, exactly like R's recorded `uk_r3_beta`. `alpha` compares as `alpha.T` (R records it r x K). Exact match at `atol=1e-8`; no cointegration-space/projection fallback needed because the normalisations coincide exactly.
- `test_svec_canada_matches_r` (new wrapper parity test, `tests/test_external_parity/test_wrapper_parity.py`):
  - Wrapper settings: `deterministic="coli"` (statsmodels case IV = urca `ca.jo(..., ecdet="trend")`: unrestricted constant + restricted trend), `k_ar_diff=2` (= R `K=3`), `coint_rank=1`, data `tests/integration/var/canada.csv` (left in place), variables ordered `prod, e, U, rw` as in R.
  - The recorded R `SR`/`LR` columns are only identified up to per-column scaling/sign: Ξ is fixed by the reduced form, and B → ΛB with any invertible diagonal Λ leaves Σ = BB′ and the zero restrictions unchanged. Both sides are therefore normalised column-wise by `sign(SR[j,j]) / ||SR[:,j]||` (the diagonal is nonzero in every column here), and then compared at `atol=1e-4` (observed max residual after normalisation: 1.1e-5).
  - Verified: `max_restriction_violation` ≈ 9.5e-13. No missing wrapper option was needed (`deterministic` passes through to statsmodels, which supports `coli`), so no xfail.
- `test_svec_raises_when_all_starts_fail` (`tests/test_models/test_svec.py`): monkeypatches `minimize` to always return `success=False`; asserts `RuntimeError` mentioning the restrictions.

## Design choices

- **Failure mode chosen: raise, not a `converged=False` flag.** The bug is that an unconstrained optimiser silently returned a restriction-violating B; a flag keeps the invalid B alive in the returned matrices and is easy to ignore downstream. Raising forces the caller to fix the setup, and the deterministic multi-start retry makes the failure path a genuine corner case only. (`max_restriction_violation` / `restr_tol` attributes remain for inspection.)
- **`coint_rank="auto"` not implemented.** The brief marks it optional; YAGNI — the rank-3 case is fully verified with the explicit arg. Add later if a pipeline needs Johansen trace-test rank selection (`statsmodels.tsa.vector_ar.vecm.select_coint_rank`), which slots into `__init__`/`build_model`.

## Checks

Baseline on `origin/main` (run locally):
- `440 passed, 7 skipped, 3 xfailed, 7 warnings in 11.57s`; `ruff`: `All checks passed!`; `mypy`: `Success: no issues found in 5 source files`.

Final on branch:
- `uv run pytest -q` → `444 passed, 7 skipped, 2 xfailed, 7 warnings in 10.27s` (3 new tests + the un-xfailed rank-3 test; xfail count 3 → 2 as intended)
- `uv run ruff check src tests` → `All checks passed!`
- `uv run mypy` → `Success: no issues found in 5 source files`

## Proposed CHANGELOG line

- `Fixed: VECMModel now honours the cointegration rank (`model.coint_rank`, default 1) instead of silently fixing it at 1, with rank-3 parity against R verified; SVEC estimation no longer falls back to an unconstrained optimiser that could return restriction-violating B (multi-start SLSQP with a restriction check exposed as `max_restriction_violation`).`

## Open questions

- None blocking. Noted: the R `svec_canada` reference was recorded without an RNG seed (`vars::SVEC` starts from `rnorm(l)`), but the likelihood for this just/over-identified setup converges to the same equivalence class regardless of start; after per-column normalisation both sides agree to ~1e-5.