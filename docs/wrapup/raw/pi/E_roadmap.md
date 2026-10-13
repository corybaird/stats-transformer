# Task E: remaining roadmap to v1 + arXiv (pi)

- No implementation here. Priorities from A–D findings (docs/wrapup/raw/pi/{A_code,B_docs,C_arxiv,D_validation}.{md,csv}) + docs/roadmap.md + docs/archive/refactoring_progress.md + CHANGELOG.md
- Effort: S = half day, M = 1–3 days, L = week+

## Release blockers (v1 must fix)

| Item | Why | Effort | Depends on |
| --- | --- | --- | --- |
| Fix `_upsample_entity` groupby key column (B P0) | README quickstart + full-pipeline CLI crash on pandas 3.x; highest-visibility breakage | S | none |
| Fix TVECM ECT off-by-one (`tvecm.py:35`) (A P0) | wrong numbers in shipped model silently; blocks any "Verified"-grade claim for tvecm | M (needs new parity test) | tvecm tsDyn parity test |
| TVAR/TVECM degenerate-threshold fallback (A P0) | silently fits both regimes on full sample when user gamma out of trim range; shares sum to 2 | S | none |
| `FeatureEngineer.save_features` writes nothing (A P0) | documented persistence contract broken (`run()` claims to save) | S | none |
| Pipeline `create_visualizations` DataFrame truthiness crash (A P0) | stage="visualization" without model summary crashes | S | none |
| Fix/kill zero-restriction draw loop in sign_zero.py (A P1) | exact-zero restriction with random rotations = ~0 accepts, silent failure mode in registry model | M | restricted-subspace sampling (Arias et al.) OR emit warning + document |
| YAML-driven multivariate models lose `target_variables` (6 classes, A P1) | direct YAML instantiation crashes for svar/bq/proxy/sign/volatility/independence | S | none |
| Track `reports/overleaf` in git or move behind explicit artifact store (D summary) | paper itself untracked; arXiv claims not reproducible from repo | S | none |
| Persist Stata/MATLAB benchmark outputs as tracked JSON/log for: BQ (2.2e-16), VAR (5.77e-15), VECM (2e-3), ols/logit/pca Stata 0.00, TVAR threshold | every "machine precision" claim currently claimed-only (D csv); abstractions without artifacts | M | access to MATLAB/Stata engines outside container |

## Before arXiv submission

| Item | Why | Effort | Depends on |
| --- | --- | --- | --- |
| Fix Nakamura–Steinsson claim: table says 1e-8 match, script reports max absolute error 0.319 + WARNING (C P0) | one contradicted number undermines the whole validation table | M (diagnose sample-gap/`changeraw` periods or restate) | none |
| Rework validation_examples.tex rows to match actual evidence tiers (D csv statuses) | "parities" that are raw-lib-vs-R or self-referential pandas-vs-FeatureEngineer reads as stronger than they are | M | D_validation mapping |
| Rewrite models.md inventory table counts + §module paths (32 vs 34 vs 33; pca.py/kmeans.py, A/B P1) | reviewers will count; doc-consistency test doesn't catch counts or file paths | S | none |
| Add tests asserting WRAPPER (not raw statsmodels/linearmodels) in R parity: regression, panel, var, logit (D rows partial→verified) | external validation currently never touches wrapper code except volatility_svar | M | none (R container exists for parity workflow) |
| Assert the recorded-but-unused vecm_coef in var_results.json against VECMModel | recorded reference with no consumer; free win (D csv) | S | none |
| Add Limitations + Availability + tested-versions sections to main.tex (C P1) | arXiv software-paper standard; no version/DOI/dependency pin statement | S | none |
| Decide LP story: implement lag augmentation or fix models.md §5.7 spec text (A P1) | estimator as shipped contradicts paper-cited Jordà spec | M | none |
| DiD either implement CS-style influence-function SEs or downgrade doc/table status to "demo" (A/Audit-A P1) | naive 2-sample SEs vs claimed did::att_gt/csdid parity | L (M for doc downgrade) | — |
| Regenerate + commit paper figures from `stats_transformer_paper_figures.py`; call `_make_panel_coefficients` or drop stale panel figure (C P1) | 3-of-4 generator; stale committed panel_coefficients.pdf | S | none |
| Purge stale pytest cache + `-rs` dev-machine paths from baseline artifacts (A P2) | audit artifacts reference `../Users/cory/...` | S | none |
| Fix ruff F821 ×2 (Chap2.py missing var; notebook renamed class) (A) | CI ruff is blocking gate per CHANGELOG — these are the only 2 errors | S | none |
| Update CHANGELOG 1.6.0 stale lines: tests count (274→410), "no model implements predict" (2 do), remove `docs/extensions/models.md` ref (B P2) | changelog is submission-evidence surface | S | none |

## Next minor (post-v1)

| Item | Why | Effort | Depends on |
| --- | --- | --- | --- |
| Per-horizon HAC (Newey–West) SEs + lag augmentation in LocalProjections/LP-IV | matches lpirfs; removes HC3-under-serial-correlation understatement | M | TVECM fix pattern |
| R parity for: robust_ols (sandwich), iv (AER), panel_iv (fixest), pca (prcomp), gmm (gmm::), svec (vars::SVEC), independence/cvm/non_gaussian (svars id.*), tvecm (tsDyn), stvar (sstvars), LP (lpirfs) | closes the 27 claimed-only rows (D csv) | L (S each) | parity.yml workflow extension |
| Volatility SVAR: compare B matrix + rtol 1e-4 (not sorted lambda at 1e-2); DGP numeric assert (known B, λ=4/0.25) | current strongest external check has loosest tolerance and uses neither B nor DGP answer | S | none |
| Numeric DGP-recovery tests: TVAR (fixture has known coefficients), sign/BSVAR, LP, kmeans seed policy | test suite strong on shapes, weak on numbers (A P1) | M | none |
| Wire KMeans n_clusters + seeds from YAML; plumb seed for LP-IV/others missing it (B/A P1) | "Configured" claim currently false for kmeans | S | none |
| `zscore` leakage option: train-period or rolling standardization (A P1) | look-ahead in any forecasting use of featurized panels | M | none |
| Per-column transformation failure should raise (or collect-and-raise) instead of log-and-continue (A P1) | silent missing features propagate to model stage | S | none |
| Roadmap §6.2 cleanup: expose RestrictedVAR via `restricted_var` key; relocate GIRFEngine/SpecificationRunner per roadmap plan | roadmap already proposes it; cheap consistency with models/ = estimators-only | M | none |
| Add upsample regression test (annual→monthly fixture) | P0 had zero coverage; keep it covered | S | quickstart fix |
| Mark stale roadmap items in docs/archive/refactoring_progress.md: Granger "to implement" → done (1.2.0); BaseIRFEstimator listed as implemented but removed (1.6.0); Sentiment/CBI pipelines unaffected | archived doc contradicts changelog | S | none |

## Later

| Item | Why | Effort | Depends on |
| --- | --- | --- | --- |
| Roadmap Tier 1–2 items (VARX, connectedness, Stambaugh, max-share SVAR) | roadmap's own triage; highest user demand per docs | L | v1 stability |
| Roadmap Tier 3–4 (GARCH suite, Bai-Perron, FAVAR, GaR) | breadth, not credibility | L | — |
| Stargazer-style multi-model comparison table (refactoring doc "Dynamic Table Export") | real gap in TableGenerator | M | none |
| Sentiment/CBI pipelines (refactoring doc "Further Steps") | out-of-repo research projects; archive doc only | L | none |
| Bootstrap + influence-function inference suite (SVARBootstrap has zero tests; DiD CS bootstrap) | completes inference story | L | — |
| CI: doc-vs-code test for model counts, module paths, and validation-status claims (extend test_doc_code_consistency) | current test checks importability only; misses counts/paths/statuses | M | none |

## Roadmap items already done / stale (mark, don't do)

| Item | Where | Status |
| --- | --- | --- |
| Fry-Pagan median-target selection | roadmap.md §2.6 marked [Completed] | done ✓ (`get_representative_draw` in SignZero + BayesianSignZero) |
| "Granger Causality Tester: implement" | refactoring_progress.md | done since 1.2.0 (granger.py + tests) — stale |
| "BaseIRFEstimator" as shipped component | refactoring_progress.md | removed 1.6.0 (`models/timeseries/base_irf.py` deleted) — stale |
| `restricted_var` pipeline key | roadmap §6.2 "before release" framing | still absent — move to Next minor (done above) |
| "MATLAB comparator at `src/examples/software_benchmarks/matlab_comparator.py`" (benchmarks.md §3) | module does not exist — stale doc ref (B P0) |
| `tests/verification/` documented in testing_suite.md + architecture.md | removed in 1.6.0 — stale |
| Tier 2 "Proxy SVAR + Sign" | partially exists (proxy_svar.py + sign loop separate); roadmap implies joint method — keep planned but rephrase |
| Roadmap §5.1–5.6 Tier 4 | untouched, correctly planned | keep |

## Not planned (deliberate scope)
- Bayesian MCMC samplers (roadmap §6.1 explicitly deferred) — keep out of v1
- MATLAB/R engine adapters for CI in this container — parity runs stay in dedicated workflow (`parity.yml`) with recorded outputs committed as evidence