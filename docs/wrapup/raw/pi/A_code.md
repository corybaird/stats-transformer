# Task A: code errors (pi)

## Scope checked
- Import sweep: all `stats_transformer.*` modules + all 31 MODEL_REGISTRY classes
- Read-through: pipeline, base, regression/, unsupervised, logit, VAR/VECM/LP/LP-IV, SVAR/BQ/proxy/volatility/sign-zero/Bayes-sign-zero/BVAR/pandemic-BVAR/DFM/SVEC, TVAR/TVECM/STVAR/GIRF, forecasting/restrictions, featurization (FeatureEngineer, DataMerger, EventStudyBuilder), Config
- Sanity runs: Longley OLS vs published R `stats::lm` values (matches to ~1e-6); LP horizon-alignment DGP check; TVECM/TVAR alignment analytic check; SVAR/VECM metadata extraction

## Findings

### P0

| Severity | Area | Claim | Evidence | Fix hint |
| --- | --- | --- | --- | --- |
| P0 | TVECM | ECT column one period older than intended; intra-matrix inconsistency (ΔY lag column correct, ECT column shifted −1); regression uses ECT_{t−d−1} where spec/hansen–Seo say ECT_{t−d}, and effective delay = configured delay + 1 | `src/stats_transformer/models/timeseries/nonlinear/tvecm.py:35` `ect_eff = ect_series[max_lag - 1 - d : T - 1 - d]` vs lag regressors at `:49` `dY[max_lag - 1 - lag : T_d - lag]`; analytic check: levels y1[t]=t, beta=(1,0), d=1, k_ar_diff=1 → ΔY row u is Δy at level t=2+u but ECT column row u = ect[u] (should be ect[1+u]); TVAR sibling `tvar.py:56` uses `max_lag - d : T - d`, correct | Drop the extra −1: `ect_series[max_lag - d : T - d]` |
| P0 | TVAR/TVECM | Degenerate-threshold fallback silently fits both regimes on full sample → shares 100%/100%, SSR double-counted, ICs meaningless; no error/warning | `tvar.py:140-143` `mask1 = np.ones(...); mask2 = np.ones(...)`; same in `tvecm.py:148-149`; run: `gamma=1e9` → shares 1.0/1.0, aic still emitted | Raise or warn; or set regime_2_params None |
| P0 | FeatureEngineer | `save_features` never writes `output_path` — only logs data-quality metrics; `FeatureEngineer.run()` claims to persist features but writes nothing | `feature_engineering.py:450-452` (docstring-less body calls `_log_data_quality_metrics` only), `:458-465` `run()` calls it and returns df | Actually `to_parquet/to_csv` in save_features |
| P0 | Pipeline | `create_visualizations` crashes with pandas truthiness ValueError when `model_results` is falsy and `transformed_data` is a non-empty DataFrame — reachable via `stage="visualization"` when model summary file absent | `pipeline.py:212` `if not self.model_results and not self.transformed_data:` → `not df` on non-empty DataFrame raises `ValueError: The truth value of a DataFrame is ambiguous`; `run()` `visualization` branch (:330+) populates transformed_data from processed_path before model_results check | Use `if self.model_results is None and (self.transformed_data is None or self.transformed_data.empty):` |

### P1

| Severity | Area | Claim | Evidence | Fix hint |
| --- | --- | --- | --- | --- |
| P1 | registry/docs | `docs/library/models.md` module paths for PCA/KMeans are wrong (`models/unsupervised/pca.py`, `kmeans.py`); actual file is `models/unsupervised/unsupervised.py` | `docs/library/models.md` §3.3/§4.1 vs `ls src/stats_transformer/models/unsupervised/` → only `unsupervised.py` | Fix paths; merge duplicated §3.3/§4.1 + §3.4/§4.2 sections (same content twice) |
| P1 | registry/docs | Skill doc models.md stale: names `IVModel` (actual `IV2SLSModel`), claims SVAR/BVAR dispatch "wired into Pipeline._initialize_from_params only if they exist", claims `base.py:119` line refs; actual pipeline uses MODEL_REGISTRY kinds | `.claude/skills/stats-transformer-architecture/models.md` vs `pipeline.py:34-120`; `transformations.md:74` claims `dataset_config.frequency` but code reads `source_period`/`resample.target_period` (`feature_engineering.py:426-428`) | Regenerate skill docs from code |
| P1 | Multivariate YAML | Six identification models discard YAML-derived `target_variables` (overwrite base's with `target_variables or []`) → direct `Model(params_path=...)` with `model.target_variables` fails | `svar.py:11`, `blanchard_quah.py:12`, `proxy_svar.py:13`, `sign_zero.py:18`, `volatility.py:18`, `independence.py:19` vs correct pattern in `var.py:12` `target_variables or getattr(self,...) or model_params.get(...)`; run: SVARModel(params_path with target_variables) fit → `ValueError: zero-size array to reduction operation maximum` | Copy var.py pattern |
| P1 | SVAR metadata | SVAR structural coefficients never surface: `SVARResults.params` is ndarray → `_extract_single_equation_coefficients` fails → metadata `coefficients = {"error": "AttributeError: 'numpy.ndarray' object has no attribute '.index'"}` silently | `base.py:220-228` catch → `coefficients["error"]`; run: SVARModel(type B) → `{'coefficients': {'error': "AttributeError: ..."}}`; no test asserts coefficients non-error | Handle ndarray params / expose A_est,B_est |
| P1 | VECM metadata | `get_model_metadata` emits no coefficients/summary for VECM (`VECMResults` has no `.params`; beta/gamma unused) → Pipeline `model_summary.json` regression stage produces coefficient-less summary for `vecm` model_type | `vecm.py:52-55` (metrics = nobs only); run: metadata keys `['num_observations']`, `hasattr(model,'params')=False` | Map gamma/beta into coefficients dict |
| P1 | LP | LocalProjections omits lag augmentation documented in models.md §5.7 (`y_{t+h} = α + β_h x_t + Σ γ_{h,i} W_{t-i}`) — regressors are const + shock + controls only; no own-lags, and SEs are HC3 in presence of serially correlated residuals → understated SEs; docs mark parity "Planned" but spec text over-promises the implemented estimator | `local_projections.py:21-40` (single `x = sm.add_constant(x)` before horizon loop); doc spec at `docs/library/models.md` §5.7; benchmark table status *Planned* `docs/validation/benchmarks.md:66` | Add lag terms (target + shock lags p), HAC (Newey–West) SEs like Stata/R lpirfs |
| P1 | Sign restrictions | Exact-zero restriction with 1e-10 tolerance on random Haar rotations accepts ~0 draws forever; loop burns to max_draws returning empty accepted set; `get_representative_draw` then raises — no signal at fit time | `sign_zero.py:156-159` `if abs(val) > 1e-10: valid=False`; same in `bayesian_sign_zero.py:121` (mitigated there by permutation search); comment at `sign_zero.py:154-157` admits it | Implement restricted-subspace sampling (Arias et al.) or warn when accepts==0 |
| P1 | Proxy SVAR | Silent no-op identification: if `instrument_variable` column absent from df, `_estimate_proxy_svar` is skipped, no `impact_column`, metrics still return nobs/aic like a plain VAR | `proxy_svar.py:38-39` `if self.instrument_variable in self.df_clean.columns: self._estimate_proxy_svar()` — no else/raise | Raise when instrument column missing/required_columns should include it unconditionally (`_get_required_columns` does append it — only reachable if caller bypasses fit) |
| P1 | DiD | Docs claim Callaway–Sant'Anna parity vs R `did::att_gt`/`csdid` ("Implemented"), but SEs are naive two-sample normal approximation: no clustering, no influence-function multinomial bootstrap; `att` point estimates also unweighted group-time means | `did.py:118-125` (`se = sqrt(var/n + var/n)`), `_aggregate_simple` equal weights; docs claim at `docs/library/models.md` §1.2/§2.7 | At minimum implement CS influence-function bootstrap; downgrade doc claim |
| P1 | DiD | Not-yet-treated control inclusion uses `cohort_map > max(g, t)` excluding units treated between g and t; CS definition is not-yet-treated as of comparison period | `did.py:81-85` | Use `> t` for not-yet-treated-at-t definition |
| P1 | Featurization | `zscore` uses full-sample per-entity mean/std → look-ahead leakage: any lagged/contemporaneous feature standardized with future observations contaminates forecasting/panel inference | `feature_engineering.py:254-260` (mean/std of whole entity series) | Offer rolling/optional train-period standardization, or document leakage |
| P1 | Featurization | Per-column `try/except Exception` around entire transformation block logs error and continues → silently missing feature columns propagate to model stage as missing-column errors or dropped rows | `feature_engineering.py:231-292` (`except Exception as e: self.logger.error(...)`, no re-raise) at `:290-292` | Narrow to expected failures and re-raise; collect failures and raise summary |
| P1 | TVAR/STVAR/TVECM | `self.params = self.regime_1_df` clobbers `ModelBase.params` (YAML dict) → metadata `params` field always `{}`; second `fit()`/reuse reads config off a DataFrame (`.get("model", {})` → fallback defaults) | `tvar.py:181`, `stvar.py:173`, `tvecm.py:187` | Rename to `self.params_df`; use distinct attribute |
| P1 | Tests | Nonlinear models (TVAR/TVECM/STVAR) and LP/LP-IV have no numeric-correctness tests: asserts are key-presence/shape/`is None` only; no allclose vs DGP parameters anywhere for these | `tests/test_models/test_nonlinear_models.py:34-46,62-75,82-91`; LP only appears for isinstance checks in `tests/test_pipeline_stages.py:60-79` | Add DGP-parameter recovery asserts (e.g. TVAR coefficients 0.5/−0.3 from fixture DGP) |
| P1 | Tests/coverage | KMeans `n_clusters` not configurable from YAML: pipeline `unsupervised` kind passes only `features`; registry docs/skill claim "configure via model.kmeans.n_clusters" — silently ignored, always 3 | `pipeline.py:53-54` `kind == "unsupervised": return cls(params_path=..., features=self.features)`; `unsupervised.py:92` fixed default; grep `kmeans` in src → only registry line | Read `model.kmeans.n_clusters` in _build_from_params |
| P1 | Docs | Logit doc overclaims: "robust covariance options and average marginal effects" — no robust cov, no marginal-effect code in logit.py | `docs/library/models.md` §3.1 vs `src/stats_transformer/models/discrete/logit.py` (plain `sm.Logit().fit(disp=False)`) | Implement or delete claim |
| P1 | Docs (RETRACTED row) | Earlier "wheel ships only 1 parquet, installed-package loads fail" claim withdrawn: wheel contains all 54 data files under `stats_transformer/data/examples/` and `load_example('longley')` succeeds installed (uv pip install wheel test, /tmp/whltest) | wheel listing + install test | Original row replaced by this correction |
| P1 | Tests | Always-skip tests shipped with repo: panel/regression/svar/var R-parity integration tests skip without R (container has none, per task rules); `tests/test_data/test_data_loading.py:19` skips because `tests/data/panel/annual/jst_panel.parquet` never exists in repo | `uv run pytest -rs` output (4 R skips + dataset skip); `ls tests/data` → 3 items, no `panel/` | Commit fixture parquet or drop stub test |

### P2

| Severity | Area | Claim | Evidence | Fix hint |
| --- | --- | --- | --- | --- |
| P2 | Base.fit | Retry heuristic matches `"inf"` substring in any exception string (e.g. "information") | `models/base.py:237` `if "NaN" in str(e) or "inf" in str(e)` | Match "inf " / "infinite" tokens |
| P2 | base | Panel bare `except: pass` on datetime coercion hides malformed time columns (string dates silently left as strings) | `panel.py:46-48` | Catch specific and warn |
| P2 | Pipeline | `Pipeline.predict` error message "no model in stats_transformer currently does implement predict()" is false: `PandemicBVARModel.predict` and `BayesianSignZeroSVARModel.predict` exist | `pipeline.py:180` vs `pandemic_bvar.py:146`, `bayesian_sign_zero.py:251` | Update message |
| P2 | Pipeline | `save_model_summary(output_path)` crashes for bare filename (`makedirs("")` ValueError); same pattern in `run()` resample stage | `pipeline.py:159`, `:246`; `feature_engineering` run uses os.path.dirname too | Guard `if os.path.dirname(output_path)` |
| P2 | FeatureEngineer | `_get_source_periods` regex parses freq letter from column NAME suffix only; valid but heuristic — uppercase-only letter, single-letter freq words match (e.g. `x_Q_y`); a column name like `GDP_M` would resample-window wrongly | `feature_engineering.py:204-213` | Keep; document constraint (source suffix contract) |
| P2 | Visualizer | `plt.style.use`/`rcParams.update` on visualizer construction mutate process-global matplotlib state | `visualization/base.py:38-45`, `formatters/style.py:35-43` | Context manager around render |
| P2 | Unsupervised | KMeans seed hard-coded 42 (no seed param), inconsistent with other models' seed exposure | `unsupervised.py:100` | plumb seed kwarg |
| P2 | LP-IV | Uses deprecated `statsmodels.sandbox.regression.gmm.IV2SLS` (docs table claims linearmodels-style IV); sandbox SEs lack small-sample corrections | `local_projections_iv.py:4,44` | Switch to `linearmodels.iv.IV2SLS` (already a dep) |
| P2 | LP | `compute_vd` "variance decomposition" is just R² increment with floor `max(0, vd)` — not FEVD; name misleads downstream reporting | `local_projections.py:52-79` | Rename or compute proper FEVD |
| P2 | EventStudy | Horizon clamped to series end without flag (`end_idx = min(idx+h, len(vals)-1)`) → truncated horizons indistinguishable from full ones | `event_study.py:131-134` | Add `truncated` column or nan-out short horizons |
| P2 | EventStudy | `scale` param silently ignored for `logpct` (hard-codes 100.0) | `event_study.py:130`, docstring `:54-57` | Apply scale or raise |
| P2 | Filter_by_date | `datetime_format` block (incl. `"%Y-Q%q"`, not a valid pandas format anyway) is dead code — computed, never used | `feature_engineering.py:160-172` | Delete |
| P2 | Resample | Float-year date coercion floors fractional years (`2020.5 → 2020`) silently | `feature_engineering.py:447-449` `fillna(0).astype(int).astype(str)` | Parse floats to month or raise |
| P2 | Registry (RETRACTED row) | Earlier claim "IV2SLSModel, PanelIV2SLSModel imported but missing from __all__" withdrawn: runtime check shows both ARE in top-level `__all__`; real gap is model classes other than LocalProjectionsModel* (LP-IV) absent from `__all__` | src/stats_transformer/__init__.py; runtime: `[x for x in __all__ if 'Model' in x]` | see review_of_agy.md F1 |
| P2 | Docs | models.md inventory lists `SpecificationRunner` as "Direct Python API: SpecificationRunner" — fine, but `models/__init__` exports also include `SVEC` wrapper + `SVARBootstrap`, `RestrictedVARResults`, `VARLagSelector`, `VARForecaster` not listed in docs §8 table (listed: `VARLagSelector` etc. only 5 rows) | `docs/library/models.md` §8 vs `models/__init__.py:42-45` | Add rows |
| P2 | Registry | Not-a-finding detail: registry `kind` "svar_family" lumps ARIMA (univariate) in — kind name misleading | `registry.py:29` `"arima": {"kind": "svar_family"}` | Rename kind or split |
| P2 | pytest cache | `-rs` skip summary emits stale absolute paths `../Users/cory/Desktop/github/pypi/...` from checked-in `.pytest_cache` → baseline pytest.txt skip locations misleading | `docs/wrapup/raw/pytest.txt` skip block; `.pytest_cache/v/cache` in repo | Purge .pytest_cache from repo |
| P2 | BVAR IRF | `compute_irf` loops draws × horizons recomputing matrix_power each draw (O(draws·H·companion³)); slow but correct | `bvar.py:139-152` | Vectorize or precompute per-draw Phi |
| P2 | DFM | If EM never converges, loop ends without `converged_` set; final extra `_e_step` still run; `loglikelihood_history` gets 2 trailing lls (dup) | `dynamic_factor.py:52-65` | Track converged at loop end |

## What was verified clean (not findings)
- All 401 public modules import; all 31 registry classes import; ValueErrors on plain instantiation are expected contract checks (no params.yaml)
- Registry↔models/__init__↔top-level __init__ consistent; `test_registry.py` completeness test is real (subclasses → registry)
- Longley OLS matches published R `stats::lm` to ~1e-6 (const −3482258.6346, GNPDEFL 15.06187, POP −0.051104, YEAR 1829.1515)
- LP horizon alignment: y_{t+h} on x_t; DGP with shock at t−1 puts effect at h=1 (0.776 vs true 0.8) ✓
- TVAR threshold alignment correct (`th_eff = th[max_lag−d : T−d]` ✓), STVAR same; GIRF rotation/transition-weight conventions consistent with TVAR/STVAR regime conventions
- BVAR conjugate NIW posterior algebra (V_post, B_post, S_post, nu_post) and matrix-normal draw match Kadiyala–Karlsson form; companion/IRF indexing verified
- BVAR/pandemic BVAR/sign-zero/BSVAR/bootstrap/CVM/NG/GIRF all use `np.random.default_rng(seed)` — reproducible
- DFM EM follows Shumway–Stoffer with documented R-correction comment
- SVEC long-run multiplier Ξ = β_⊥ (β_⊥'(I−Γ)α_⊥)^{-1} α_⊥' — matches Lütkepohl
- RestrictedVAR mask shape validated loudly; sigma df correction simplified but commented
- Config.validate wired to registry + alias resolution
- volatility identification: eigen alignment + Cholesky-normalization + lambda recomputation sound; regime alignment resid vs regime labels row-aligned

## Ruff (2 errors)
- `data/examples/regression/ghysels_ch2/Chap2.py:84` F821 `r_sim_data_est` undefined — Bai–Perron R-parity script references a variable it never defines; would NameError at runtime; broken example script, not library code. Fix: define `r_sim_data_est = ...` via `pandas2ri.py2rpy(sim_data_est)` like the preceding assignments
- `notebooks/08_structural_timeseries_models.ipynb` cell 11 F821 `SignRestrictionsSVARModel` — class does not exist (renamed/never named that; registry has `SignZeroSVARModel`); running the notebook fails immediately. Fix: rename to `SignZeroSVARModel` and update kwargs to current signature (config-based restrictions)

## Skipped/xfailed taxonomy
- 7 skipped: 4× R-environment-not-available (parity tests: `tests/integration/{panel,regression,svar,var}/test_*.py`), 1× missing `tests/data/panel/annual/jst_panel.parquet`, 2× `test_pipeline.py` fixtures that are restriction-schema YAMLs not Pipeline params files (guard working as intended)
- 1 xfailed: statsmodels 0.14.6 loglike bug for svar_type='A' with K=2 (1 free parameter) — genuine upstream crash, reproduced against raw statsmodels; K≥3 unaffected (test_svar.py rationale string)

## Confidence
High on registry/import/metadata findings (executed checks); high on TVECM off-by-one (analytic); medium on LP spec-severity (estimator is documented as "Planned" parity but doc's own spec contradicts code); medium on DiD SE-severity (parity also "Implemented" claim vs naive SEs).

## Not checked
- `reporting/` (exporters, adapters, reporter) beyond skim
- `visualization/charts/*`, `tables/`, `formatters/`, `defaults/` internals
- `featurization/base.py`, `utils/dict_country_converter.py` contents
- `spec_runner.py`, `panel_diagnostics.py`, `diagnostics/{residuals,stability}.py`, `non_gaussian.py`, `cvm.py`, `independence.py`, `lag_selection.py`, `granger.py`, `arima.py`, `bootstrap.py`, `decompositions.py` (import-only)
- Paper tables (`reports/overleaf/tables/*.tex`) cross-check against model outputs (Task C/D territory)
- Stata/MATLAB/R parity values (no cross-language runtime per task rules)
- `data/examples/**` scripts other than Chap2.py
- notebooks other than 08
- wheel build (package includes/excludes beyond importlib.resources logic read)