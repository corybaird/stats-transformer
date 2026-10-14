# NOTES_C — VECM cointegration rank + SVEC optimizer fallback

Branch: `fix/vecm-rank-svec-optimizer` (based on `origin/main` 57394fa).

## Commits

1. `7aceb5c` FIX honour coint_rank in VECMModel and un-xfail rank-3 UK parity test
   - `src/.../reduced_form/vecm.py`: new `coint_rank=1` constructor arg, YAML `model.coint_rank` (same precedence as `k_ar_diff`), passed into `VECM(...)` in `build_model`. Default 1 keeps existing behaviour identical.
   - `tests/test_external_parity/test_wrapper_parity.py`: removed the strict xfail on `test_vecm_ghysels_ch7_uk_rank3_matches_r`; it compares the full rank-3 `beta` (levels + `det_coef_coint` row, shape 5x3) and `alpha` (as `alpha.T`) against R's recorded `uk_r3_beta` / `uk_r3_alpha` at `atol=1e-8` (exact match: statsmodels normalises the top r rows of beta to the identity, same as R).
2. `9d4345c` FIX replace SVEC unconstrained Nelder-Mead fallback with multi-start fit and restriction check
3. `fb45199` DOCS update VECM rank and SVEC parity status in benchmarks and models docs
4. `d2ec80e` FIX solve just-identified SVEC system directly, select best multi-start fit, raw parity vs R (round 1)

## Round 1 (flags C-1..C-4)

Root cause, found while fixing C-4: the concentrated SVEC likelihood depends on B only through `B B'`, whose unique minimum over positive-definite choices is `B B' = sigma_u` (equivalently `Sigma = A^{-1} B B' A^{-T}` with R's fixed `A = I`, see `vars::SVEC` source). For the Canada restriction set (`LR[1,2:4]=0`, `LR[2:4,4]=0`, `SR[4,2]=0`, r=1) the model is just-identified (10 free directions = K(K+1)/2), so the ML solution satisfies `B B' = sigma_u` **exactly** and the scale of every SR column is identified. The old SLSQP-with-equality-constraints solver sat at a feasible but suboptimal point (`f = -125.5`/`-133.9` vs R's `-135.898`) because of the rank-deficient constraint Jacobian (one LR restriction is linearly redundant). My round-0 parity test then erased the residual scale difference by unit-norm-rescaling columns — agy is right that it hid a real gap.

Fixes applied:

- **C-4 (estimator)**: `_fit_structural` now builds the restriction matrix E on `vec(B)` (unit rows for SR zeros; rows of Ξ at the affected column block for LR zeros), takes the free directions `Sb = null_space(E)`, and:
  1. If just-identified (`Sb.shape[1] == K(K+1)/2`): solves the system {restrictions, `B B' = sigma_u`} directly with `scipy.optimize.root` (`method="lm"`, determined overdetermined by design) from deterministic starts (Cholesky factor, scaled identity) plus seeded random starts. Any root solution has `B B' = sigma_u`, i.e. the global ML optimum.
  2. Otherwise (over-identified, or the solve failed): maximises the concentrated likelihood along the free directions (BFGS) from projected deterministic starts and 2 seeded random starts.
  Every start is evaluated; among feasible candidates (restrictions within tol, finite objective) the best objective wins (ties within 1e-9 keep the first, deterministic, start) — covers C-2.
- **C-4 (parity test)**: now compares RAW `SR` and `LR` with only column-sign alignment (flip a column whose diagonal sign differs); no rescaling. Tolerance: `atol=1e-6`; observed maxima on this data are 4.0e-10 (SR) and 1.7e-9 (LR), so ~3 orders of magnitude of margin, guarding BLAS/platform variation.
- **C-4 (objective)**: reached `f = -135.8976838414245` vs R's `-135.8976838414244` (diff 1.1e-13); `max |BB'_py − BB'_R|` = 4.7e-11; R's own `BB' − sigma_u` = 4.7e-11, ours = 1.1e-16. Parity test passes raw, so no xfail was needed.
- **C-1**: violations are computed on the raw optimizer output (`opt_res.x` / `sol.x`) and the SR entries are only clamped to their target zeros *after* validation passes.
- **C-2**: covered by the best-feasible selection above.
- **C-3**: second forced-failure case added: `root` (and BFGS) claim `success=True` but return a B violating `SR[0,1]=0`; asserts `RuntimeError` and non-zero `max_restriction_violation` (attached to the exception as `err.max_restriction_violation`, set on `self` before raising). The first case (all optimizer runs fail) also patches `root` now.

## New/updated tests

- `test_vecm_ghysels_ch7_uk_rank3_matches_r` (un-xfailed): rank-3 `beta`/`alpha` vs R at `atol=1e-8`.
- `test_svec_canada_matches_r`: wrapper parity vs R `svec_canada` (raw matrices, signs aligned, `atol=1e-6`, violation check).
- `test_svec_raises_when_all_starts_fail`, `test_svec_raises_on_restriction_violation` (`tests/test_models/test_svec.py`).

## Design choices

- **Failure mode: raise, not a `converged=False` flag.** The bug is silently returning a restriction-violating B; raising forces the caller to fix the setup. `SVEC.max_restriction_violation` (and `restr_tol=1e-6`) remain inspectable attributes, and on failure the violation is attached to the exception.
- **`coint_rank="auto"` not implemented** — optional in the brief (YAGNI); add later via `statsmodels.tsa.vector_ar.vecm.select_coint_rank` if a pipeline needs trace-test rank selection.

## Checks

Baseline on `origin/main` (run locally):
- `440 passed, 7 skipped, 3 xfailed, 7 warnings in 11.57s`; ruff `All checks passed!`; mypy `Success: no issues found in 5 source files`.

Final on branch (after round 1):
- `uv run pytest -q` → `445 passed, 7 skipped, 2 xfailed, 7 warnings in 9.48s`
- `uv run ruff check src tests` → `All checks passed!`
- `uv run mypy` → `Success: no issues found in 5 source files`

## Proposed CHANGELOG line

- `Fixed: VECMModel now honours the cointegration rank (`model.coint_rank`, default 1) instead of silently fixing it at 1, with rank-3 parity against R verified; SVEC estimation no longer falls back to an unconstrained optimiser — a just-identified restriction set is solved directly from the ML system (B B' = Sigma_u), over-identified ones maximise the concentrated likelihood from several deterministic starts, the identifying restrictions are validated to 1e-6 (violation exposed as `max_restriction_violation`), and the fit raises instead of returning a restriction-violating B.`

## Round 2 (flags C2-1, C2-2)

- **C2-1**: agy is right — my round-1 rewrite built the LR rows of the restriction matrix as `row[j*K:(j+1)*K] = Xi[i, :]` (row j of B in row-major vec), which encodes `B Xi^T`, not `(Xi B)[i, j]` (column j of B dotted with row i of Xi, i.e. indices `j, K+j, ...`). The just-identified path was unaffected only because `direct_residuals` computes `Xi @ B` directly and the free-dimension count coincided. Fixed: the row construction moved to a module-level helper `_restriction_matrix` using `row[j::K] = Xi[i, :]`, and the violation logic to `_restriction_violation`. Two unit tests added: (a) `E @ vec(B)` reproduces the restricted quantities on a random `(Xi, SR, LR, B)` (fails with the old indexing), (b) `restriction_violation(Sb @ g) < 1e-12` for random `g` drawn from the free directions.
- **C2-2**: the over-identified fallback now accepts a fit when the objective is finite and did not get worse relative to its start, instead of requiring BFGS's `success` flag (finite-difference BFGS at tight tolerance stops with a precision-loss flag while still at a good point); `gtol` loosened to 1e-8. A test fits the Canada data with one extra SR zero (`SR[0,0]=0`, 9 free directions < K(K+1)/2): it fits, restrictions hold (violation 2.5e-16), and the concentrated objective (-38.19) is no better (lower) than the just-identified optimum (-135.90) minus 1e-9, plus the same model without the extra zero still fits.

## Open questions

- None blocking.
- Note for the record: `vars::SVEC`'s default start is unseeded `rnorm(l)`, so R's recorded `svec_canada` point is reproducible here because the ML solution of this just-identified system is unique (up to column signs) — verified by the raw-value parity test.