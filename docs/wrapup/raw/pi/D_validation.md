# Task D: validation triage summary (pi)

## Coverage
- 46 entities (31 registry models incl all public ones + 15 non-registered public classes): 73 evidence rows
- CSV: docs/wrapup/raw/pi/D_validation.csv

## Counts by status (rows)
| Status | Rows | Note |
| --- | --- | --- |
| verified | 10 | 2 "verified" are same-lib trivialities (ols/robust_ols/logit vs statsmodels = identical code path); 1 real external: volatility_svar vs R svars::id.cv (wrapper-tested; lambda diag only, rtol=1e-2); 3 internal-reference (bq trend recomputation, RestrictedVAR ma_rep, VARForecaster ma_rep, RollingOriginEvaluator hand-calc, GMM vs linearmodels IVGMM)
| partial | 19 | script or recorded reference exists but no assert (vecm json), external-vs-raw-lib not wrapper (all R parity except volatility), or printed-only (mroz/grunfeld), or DGP-recovery (DiD, DFM, gmm, bvar limit)
| claimed-only | 27 | doc/table claims (Stata/MATLAB parity, R svars/plm/etc, machine-precision numbers) with no recorded artifact or runnable evidence
| none | 17 | no external validation at all: arima, panel_iv, kmeans, svec, svar (masked-model), sign_restrictions, cvm_svar, non_gaussian_svar, tvecm, stvar, local_projections, bsvar_sign_zero, pandemic_bvar, GIRFEngine, SVARBootstrap, SVEC wrapper, probit (docs parity claims only)

## Models with NO external validation, ordered by risk (x = complexity x user-facing)

| Model | Why risky | Complexity | Exposure |
| --- | --- | --- | --- |
| tvar | degenerate fallback bug + params clobber (A-report P0); only shape tests + claimed-only Stata parity | high | docs "Stata threshold parity Verified" |
| tvecm | ECT off-by-one (A-report P0); claimed tsDyn TVECM | high | registry model |
| stvar | no external, untested numerics | high | registry |
| sign_restrictions | exact-zero restriction dead (accepts ~0 draws) | high | registry + paper feature |
| cvm_svar / non_gaussian_svar / independence_svar | data-driven identification, no external check; svar.R has id.dc commented out | medium-high | registry |
| bsvar_sign_zero | Bayesian sampler, 1 test (draws>=1) | high | registry |
| local_projections | no numeric test anywhere in repo; no lag augmentation vs documented spec | medium | paper-facing (LP-IV used in paper narrative) |
| pandemic_bvar | no external; weighted-NIW posterior | high | registry |
| svec | MLE fitting, only shape asserts | high | kpsw example runs |
| svar | A/B-model parity claim false (integration covers volatility only) | medium | registry |
| arima | no parity at all | low | registry |

## Gitignored / missing evidence locations (sizes)

| Path | Size | In git | Role |
| --- | --- | --- | --- |
| references/matlab_benchmarks (VAR-Toolbox + Kilian-Lutkepohl code) | 191M (kilian_2017 183M) | IGNORED | only source for the claimed MATLAB parity runs (BQ1989 replication in Replic/) |
| data/raw | 915M | IGNORED | fred_cache etc for featurize examples |
| data/final | 100M | IGNORED | pipeline outputs |
| data/temp | 1.4M | IGNORED | scratch outputs |
| data/examples/academic/ghysels_2018 | absent locally | IGNORED per data.md (not in checkout) | textbook dataset |
| reports/overleaf (paper + tables + figures) | - | IGNORED (whole dir) | paper itself is untracked |
| docs/wrapup (audit outputs incl this file) | - | tracked? per repo rules write-only; wrapup NOT ignored | audits |

## Key structural weaknesses
1. External R-parity integrations test the UNDERLYING libs (statsmodels/linearmodels) not the wrappers, except VolatilitySVARModel — the "Verified" doc rows generalize raw-lib parity to wrapper behavior
2. Stata/MATLAB engine adapters ship and run but nothing recorded is persisted; all machine-precision numbers live only in docs
3. vecm reference values recorded (var_results.json) but no assertion consumes them
4. benchmark suite in this container: every engine unavailable → all Stata/MATLAB benchmarks unreproducible here (rpy2 also absent despite pyproject benchmarks extra)

## Not checked
- MATLAB run inside references/matlab_benchmarks (191M vendor tree; engines unavailable, task rules forbid MATLAB)
- Stata .dmo/.do artifacts search depth inside that tree
- docs/validation/testing_suite.md claim that matlab comparator requires local VAR-Toolbox checkout (path exists, untracked)
- CI workflows (.github) not inspected
