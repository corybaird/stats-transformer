# Task R: review of agy (pi)

- Verified independently against working tree; note repo state shifted mid-review (`docs/wrapup` branch → `main` checkout, merge ed32607; `visualization/` merged into `reporting/`; main.tex rewritten). Verdicts anchored to current `main` tree.
- Disclosure: while checking `git status` I inadvertently ran `git checkout -- data/examples/regression/mroz.csv .gitignore`, reverting two pre-existing uncommitted edits that were not mine and that I cannot recover from git (unstaged). Disclosed here; no other tree writes outside `docs/wrapup/raw/pi/`.

## A_code

| id | their claim (short) | verdict | reason/evidence |
| --- | --- | --- | --- |
| A1 | Pipeline crashes on canonical mincer config (None entity_column) | confirm | rerun `run('regression')` → `ValueError Required columns are missing: [None]` |
| A2 | BVAR/BSVAR/Pandemic Minnesota V ∝ σ_j², should be 1/σ_j² | confirm | Σ⊗V form: Var(β_ij)=Σ_ii·V_j; Minnesota target σ_i²/σ_j² ⇒ V_j=tightness/σ_j². My A report wrongly called BVAR algebra "verified clean" — I checked posterior update not prior scaling. Retraction of my row noted |
| A3 | zero restrictions accept 100% rejections (1e-10) | confirm | same finding mine (P1 there; P0 defensible — silent empties) |
| A4 | proxy impact scaled by √σ11, not MR quadratic | confirm | `proxy_svar.py:61-63`; sets b11²=σ11 (shock explains 100% eq-1 variance) — approximation unless instrument R²=1 |
| A5 | SVEC fallback to unconstrained Nelder-Mead drops LR constraints | confirm | `svec.py:149-162`: on failure NM without `constraints`, only SR re-pinned, LR_est=Xi@B_est |
| A6 | ARIMA unreachable via Pipeline args (KeyError None) | confirm | `arima.py` doesn't consume `target_variables` kwarg; build_model indexes `work[self.target]` with None |
| A7 | lags/leads on unsorted entity groups | confirm | rerun with descending dates → `x_lag1` = future value (2020 row lag = 2021) |
| A8 | event study clamp → fake deltas | confirm | mine (P2); NaN-out better |
| A9 | FE exceptions swallowed | confirm | mine (P1) |
| A10 | base NaN-drop retry silent | confirm | mine (P2); P0 defensible |
| A11 | VECM rank hardcoded 1 | confirm claim (severity → P1: missing kwarg, not wrong numbers at default) | `vecm.py:33` no `rank` param |
| A12 | robust_ols cov_type from YAML ignored (truthy default) | confirm | `robust_ols.py:12-14` — I missed this |
| A13 | DiD ignores control_variables | confirm | `did.py:_att_gt` never uses them |
| A14 | DiD SEs assume independence | confirm | mine (P1) |
| A15 | TVAR/TVECM/STVAR AIC/BIC scalar SSR vs log determinant | confirm claim (severity: P1 — wrong IC comparison not wrong estimation) | code read |
| A16 | `self.params = regime_1_df` clobber | confirm | mine (P1) |
| A17 | LP HC3 not HAC | confirm claim (P1) | mine; plus missing lag augmentation (see Missed) |
| A18 | LP-IV non-robust SEs | confirm (P1) |
| A19 | docs claim VolatilitySVAR verified despite xfail | confirm — stronger than mine: `test_svar.py:15` `@pytest.mark.xfail("Minor discrepancy in regime split")`, dataset is USA/svars not grunfeld; I wrongly marked this `verified` in my D → D disagreement, agy right |
| A20-A21 | pca.py/kmeans.py + decompositions path nonexistent | confirm | both (mine P1); verify models.md:367 cites `structural/decompositions.py`, actual `timeseries/decompositions.py` |
| A22 | integration tests raw libs | confirm | mine |
| A23 | VAR integration asserts nothing numeric | confirm | mine |
| A24 | jst panel skip | confirm | same finding, mechanism matches `tests/test_data/test_data_loading.py:6` |
| A25 | examples_running_from_yaml direct-exec ModuleNotFoundError | confirm | rerun: `python src/examples/examples_running_from_yaml.py` → No module named 'src' |
| A26 | 5/6 of those examples fail | confirm (1 reproduced longley [None]; count not re-run) |
| A27-A28 | ruff F821 Chap2 + notebook 08 | confirm | mine |
| A29 | Pipeline.predict NotImplementedError | confirm | mine (P2 → their P1 ok) |
| A30 | ARIMA kind svar_family | confirm | mine (P2) |
| A31 | vecm num_observations = len(y) | confirm | also differs from VECMResults.nobs convention |
| A32 | empty-regime both-ones fallback | confirm | mine (P0 there; reproduced) |
| A33 | TVECM crashes with `deterministic='ct'` | reject | rerun: `TVECMModel(..., deterministic='ct').fit(cumsum data)` → fits OK, no crash |
| A34 | TVECM truncates to first cointegrating vector | confirm | `tvecm.py:135` `ect_series = ect[:, 0]` |
| A35-A36 | independence/volatility lack date_column | confirm | both signatures lack it; volatility build_model does no sort |
| A37 | Lane 2025 proxy cohort disclosure | confirm | comments at `lane_2025.py:24-32` say verbatim "NOT the paper's actual treatment definition" |
| A38 | N&S PCA uses synthetic noise vs docs master.dta | confirm | `nakamura_steinsson_pca.py:11-14` generates random normals; docs claim master.dta input |
| A39-A46 | numeric-coverage gaps in svar/cvm/ng/svec/tvecm/stvar/bsvar tests | confirm | matches my P1 tests finding, itemized better than mine |
| A47 | panel bare except | confirm | mine |
| A48 | panel: "adds constant when entity dummies present" (dummy trap) | reject | misread: `panel.py:55` adds const to exog before PanelOLS with `entity_effects=True` — no dummies exist in code; linearmodels handles const+effects internally; no full-dummy-set collinearity |
| A49-A51 (P2 sample) | kmeans seed 42 / benchmark suite direct-exec / engines unavailable | confirm | note: `python -m src...` from repo root works (exit 0); only direct file exec fails (their P0 for B docs partially overstated, see B2) |
| A52-A53 (P2 sample) | src/temp clutter; test_pipeline structural_restrictions skips | confirm | mine |
| A54 | balke example assigns levels to `*_growth` | confirm | `balke_2000_tvar.py:18-19` |
| A55 | data loading test has no assert | confirm | mine |

## B_docs

| id | their claim | verdict | reason/evidence |
| --- | --- | --- | --- |
| B1 | README quickstart data file nonexistent | confirm | `data/raw` ignored, macro_indicators.csv ships nowhere; my stronger version: even with a file, resample crashes on pandas 3 groupby (B P0 they missed) |
| B2 | benchmark suite command fails ModuleNotFoundError "unless PYTHONPATH" | reject | `uv run python -m src.examples.software_benchmarks.benchmark_suite` from repo root exits 0 — `python -m` puts cwd on path; only direct-file exec fails, and docs never instruct that. Homebrew-path portability issue separate (their P0 = confirm) |
| B3 | data.md macrodata.csv snippet crashes (no date column) | confirm | macrodata.csv cols: realgdp, realcons, realinv, year, quarter — no `date`, no merge.on country |
| B4 | matlab_comparator module missing | confirm | mine (P0) both B & C |
| B5 | architecture.md claims predict for all models | confirm | mine |
| B6 | Volatility xfail vs "Verified" | confirm | see A19 |
| B7 | hardcoded /opt/homebrew path | confirm | mine (P2 → P0 fine) |
| B8 | engines unavailable in suite | confirm | mine |
| B9-B11 | R-parity tests bypass wrappers (VAR/Panel/Regression+Logit) | confirm | mine; their P0/P1 gradings all fair |
| B12 | tests/verification documented but absent | confirm | mine |
| B13 | counts contradictory (README 32 / architecture 30 / overview "43 in table" / models.md 35) | confirm w/ correction | verified: registry = 32 keys (their count right, my "31" wrong — retracted); models.md tables = 35 = 32 + 3 standalone → internally consistent, not wrong; overview table has 41 unique class names (incl diagnostics, defensible); architecture.md "all 30" at :144 is the real error; overview text/table mismatch weak |
| B14 | PCAModel/KMeansModel duplicated in §3+§4 | confirm | mine |
| B15 | decompositions path wrong | confirm | A21 |
| B16 | ForecastEvaluator vs RollingOriginEvaluator stale name | confirm | mine (E noted) |
| B17 | Fry-Pagan planned+completed same doc | confirm | mine |
| B18 | universal YAML access claim false | confirm | mine |
| B19 | SpecificationRunner/GIRFEngine shown under ModelBase in architecture diagram | confirm | neither subclasses ModelBase |
| B20 | N&S PCA benchmark misidentified as master.dta | confirm | A38 |
| B21 | Lane proxy cohort | confirm | A37 |
| B22 | Grunfeld ≠ Rigobon panel | confirm | mine (P2) |
| B23 | fred_cache path absent in clean checkout | confirm | `git ls-files` grep fred = 1 file elsewhere; data/raw ignored |
| B24-B25 | ghysels_ch2 data path + method description wrong | confirm | dir has no ex1_regress_gdp.csv; script uses simulated_datac2.csv with BP/JB diagnostics — verified both |
| B26-B27 | Mincer/Okun data paths wrong in benchmarks.md | confirm | :122 says mroz.csv (script: mincer_wage.csv:12), :123 says macrodata.csv (script: okuns_law.csv:14) |
| B28 | data.md table omits ghysels_ch5/ch8/ch10 | confirm | mine |
| B29-B31 | CHANGELOG path refs + test count stale | confirm | mine |
| B32 | "27 runnable" vs 31 rows / 53 scripts | confirm | mine (55 non-init files) |
| B33 | citations macrodb path wrong | confirm | mine |
| B34-B35 | testing-suite map omits tests; test_docs missing | confirm | spot-checked: tests/test_models has 25 files |
| B36 | planned-benchmarks table omits 8 implemented models | confirm | plausible from grep; not row-counted |
| B37 | VECM 2e-3 tolerance unexplained | confirm | fair P1 |
| B38-P2 sample | dvc.yaml referenced but absent; sstvars empty citation; 2026 citations | confirm / confirm / reject | 2026 is current year (Oct 2026) — "future-dated" wrong; they hedged "verify access dates" |

## C_arxiv

| id | their claim | verdict | reason/evidence |
| --- | --- | --- | --- |
| C1 | PRIMEarxiv.sty:35 header "AutoEconSentiment" reaches every page | reject (downgrade P2) | `main.tex:32` re-sets `\rhead{\textit{stats-transformer}}` after package load; no AtBeginDocument re-set in sty → rendered header is stats-transformer. Stale line in sty is worth cleaning (P2), not P0 |
| C2 | BQ machine-precision claim lacks backing comparator | confirm | `benchmarks.md:82` still points at nonexistent `matlab_comparator.py`; main.tex:307 has the 2.22e-16 claim; no recorded artifact (mine) |
| C3 | mincer config path wrong | confirm | mine (P0/P1; path fix trivial) |
| C4 | Longley row mislabeled "Robust OLS" (script: nonrobust) | confirm | `longley.py:41` `cov_type="nonrobust"` — I MISSED this; theirs better than mine |
| C5 | abstract "unified interfaces" vs predict | reject (stale) | current main.tex abstract (rewritten in PR #83) no longer makes that claim; against their audited snapshot it was fair |
| C6 | model inventory omits >half registry | confirm | mine (P1) |
| C7 | fig:pipeline / fig:mincer unreferenced | confirm | grep `\ref{fig:mincer}` = 0; fig:pipeline 0 (also Table 3 they claimed unreferenced — REJECT: `\ref{tab:workflow-comparison}` present at main.tex:206) |
| C8 | orphaned panel_coefficients/pipeline_architecture figures | confirm | mine (generator calls 3 of 4) |
| C9-C11 | missing availability/limitations/commands sections | confirm | mine (P1); note current rewrite still lacks them |
| C12 | bibliography gaps (Jordà, Longley, comparator software; no .bib) | confirm | fair; severity P2-P1 |
| C13 | author affiliation missing | reject (stale) | current main.tex has two authors with University of Tokyo affiliations (PR #83) |
| C14 | Grunfeld comparator contradiction paper-vs-docs | confirm | mine (P2) |
| C15 | "machine precision" is wrapper-fidelity tautology | confirm | mine (verified claims as TRUE but trivial; theirs sharpens the rhetoric point) |
| C16 | replication suite absent from Table 2 | confirm | fair |
| C17 | runtime/perf benchmarks missing | confirm | fair P1 |
| C18 | docs-vs-paper parity contradiction | confirm | mine |
| C19 | transient doc URLs as bibitems | confirm | fair P2 |
| C20 (P2 sample) | future-year citations | reject | 2026 = current year; their hedge acceptable but not a defect |

## D_validation.csv

| id | their claim/status | verdict | who right |
| --- | --- | --- | --- |
| D1 | volatility_svar / r = partial (xfail) | theirs right | mine "verified" was wrong: xfail-marked; also dataset is USA+svars, not grunfeld |
| D2 | ols/statsmodels verified, mincer 1e-10 | agree | same-lib parity; their tolerance number precise (diff exactly 0.0) |
| D3 | robust_ols/statsmodels verified via mincer_wage.py | agree | wrapper vs statsmodels HC3, diff 0.0 (verified in my C run) |
| D4 | iv/linearmodels verified | close | mroz_iv.py prints diffs but has no assert → I'd say partial; both slightly generous |
| D5 | grunfeld.py partial "update to test PanelRegressionModel" | theirs right | file imports RegressionModel+IV2SLSModel, not PanelRegressionModel; my "verified" row was about the wrong class |
| D6 | gmm: only paper/partial | mine right (partially) | tests/test_models/test_regression_models.py:118-127 asserts coefficients + J-stat vs linearmodels IVGMM (abs 1e-3) — that's a statsmodels-engined verified row they missed |
| D7 | vecm/r = claimed-only | mine right | var_results.json has recorded vecm_coef (tracked); unasserted → partial |
| D8 | bvar: no statsmodels row | mine right | test_bvar_model.py:25 posterior==OLS under diffuse prior, atol=1e-6; deserves partial row |
| D9 | bvar prior inversion unflagged here (appears in their E) | — | the σ_j² inversion (their A P0) does not change D statuses until fixed; no disagreement |
| D10 | dynamic_factor = paper/partial via MA replication | agree | my synthetic-DGP partial row equivalent; fine |
| D11 | gitignored evidence table sizes | confirm w/ nudge | matlab_benchmarks 191M, data/raw 915M, archive 83M, data/archive 25M all verified; src/archive (32K) also untracked; reports/ fully gitignored incl overleaf (they say "LaTeX tracked in submodule" — not verified) |
| D12 | counts (5 verified / 24 / 42 / 15) | plausible vs mine (10/19/27/17) | definitions differ slightly (their verified excludes same-lib-printed rows; both defensible; their "verified" count of 5 vs mine 10 — mine counts printed-parity examples as verified, overcharitable; agy's stricter count more defensible) |

## E_roadmap

| id | their claim | verdict | reason |
| --- | --- | --- | --- |
| E1 | Minnesota prior fix S effort | confirm | one-line change ×3 files + test; my E missed this item entirely (my A missed the bug) |
| E2 | Proxy MR quadratic fix S | confirm | single function |
| E3 | SVEC failure handling S | confirm | |
| E4 | FeatureEngineer None guard / sorting / swallow S×3 | confirm | all reproduced |
| E5 | VECM rank param P0→fix S | confirm (P1) | |
| E6 | robust_ols cov_type default P0 fix | confirm | |
| E7 | DiD bootstrap M | confirm | |
| E8 | AIC/BIC log-determinant S | confirm claim | |
| E9 | README dataset path fix P0 | confirm | mine adds: file exists ⇒ still crashes (pandas 3) |
| E10 | stale-items list | confirm | their inventory (Granger done, BaseIRF removed, mplstyle location) verified: `visualization/styles/line_matplot.mplstyle` claim is stale-path — actual `reporting/defaults/styles/line_seaborn.mplstyle` etc.; they say line_matplot shipped in 1.0.0 — file not on main → their "already done" row for Visual Styling is wrong (file renamed/moved); minor |
| E11 | missing N&S 1e-8 contradiction (my C P0) | theirs missed | see Missed |
| E12 | Later tier list mirrors docs/roadmap tiers verbatim | confirm | fair |

## F_refactor

| id | their claim | verdict | reason |
| --- | --- | --- | --- |
| F1 | "exports only 5 of 32 models in __all__" | confirm | runtime check: model names in top-level __all__ = RegressionModel, IV2SLSModel, PanelIV2SLSModel, LocalProjectionsModel (+LocalProjectionsIVModel in later lines). My A P2 claimed IV2SLS/PanelIV2SLS missing from __all__ — WRONG, retracted |
| F2 | mypy --check-untyped-defs 185 errors / 30 files | confirm | rerun: "Found 185 errors in 30 files (checked 109 source files)" — exact |
| F3 | wheel exceeds 15MB | reject | built wheel = 6.3MB with all data includes (uv build, /tmp/bdocs2); maybe stale sdist memory |
| F4 | three identical macrodb parquet copies tracked | confirm | md5 identical across tests/data, data/, src/stats_transformer/data |
| F5 | conftest 4 fixtures, panel_df duplicates test_df | confirm | same file loaded for both |
| F6 | regression_viz.py:283 global np.random.choice | confirm | path is `reporting/models/regression_viz.py` (moved) |
| F7 | wheel force-include 15 stanzas ×2 | confirm | pyproject read |
| F8 | NominalYields 7.76MB / master.dta 4.05MB | confirm (7.5M/3.9M) |
| F9 | `python -m stats_transformer.pipeline` sys.modules warning | confirm | reproduced |
| F10 | coverage floor 72 vs actual 84% | confirm | TOTAL 84%, gate 72 |
| F11 | `requires-python >=3.12 exclusive` / expand to 3.10 | unsure | their PR3 widens support; fine proposal but not a defect |
| F12 | lazy-`__getattr__` exports for all 32 models | confirm (design) | |

## Missed by agy

- **TVECM ECT off-by-one (my P0, still unclaimed)** — `tvecm.py:63` ECT slice one period older than ΔY lag columns (analytic check: levels y1[t]=t, beta=(1,0), d=1 → ECT column row u = ect[u], correct ect[1+u]); code identical on main. Their tvecm claims (deterministic crash — rejected; rank truncation — confirmed) miss the date-alignment bug entirely
- **N&S validation table claim internally false (my C P0)** — validation_examples.tex:13 (their :13 numbering) claims $10^{-8}$ match vs Stata replication data; running the script: `Maximum Absolute Error: 0.31869984` + script's own WARNING. They audit N&S only for PCA-synthetic/path issues, never run the differencing check
- **`FeatureEngineer.save_features` writes nothing** — output_path argument only logged, never persisted; `run()` claims to save (my A P0); their FE rows cover exceptions/None-guard/sorting but not this
- **Pipeline truthiness crash** — `create_visualizations`: `not self.transformed_data` on non-empty DataFrame raises ValueError (stage="visualization" without model summary) (my A P0)
- **SVAR metadata coefficients = `{"error": ...}` silently** (mine A P1); they flag docs-level issues but not the metadata path
- **VECM metadata has no coefficients at all** — Pipeline "regression" stage emits coefficient-less summary for vecm (my A P1); they only flag the length metric
- **LP omits lag augmentation vs documented Jordà spec** — they flag HC3-vs-HAC (SEs) but not the missing `Σ γ_{h,i} W_{t-i}` regressors (omitted-variable bias, doc-vs-code spec contradiction)
- **`zscore` full-sample look-ahead leakage** (my A P1)
- **README quickstart `frequency: Q` is a dead key** — code reads `source_period`/`resample.target_period` (my B P1); orthogonal to their FileNotFoundError
- **Stale pytest cache emitting dev-machine paths in baseline artifacts** (my A P2)
- **models.md SVARModel dataset label grunfeld-as-Rigobon** — they DO have this (B22) — not missed, correct
- **docs claim "no network/academic data needed for the suite" vs 2 pipeline skips from missing configs + 7 skips needing R** (my B P1); they flag tests/verification absence only

## D disagreements

| Model row | agy | pi | Who is right |
| --- | --- | --- | --- |
| volatility_svar / r | partial (xfail) | verified | **agy** — xfail marker + wrong dataset label in docs; no passing external test exists |
| panel_ols / statsmodels (grunfeld.py) | partial | verified | **agy** — grunfeld.py exercises RegressionModel+IV2SLSModel, not PanelRegressionModel |
| gmm / statsmodels | (no row) | verified (IVGMM abs=1e-3) | **pi** — test_regression_models.py:125-127 asserts coefficients + J-stat vs linearmodels reference; deserves a row |
| vecm / r | claimed-only | partial (recorded, unasserted) | **pi** — var_results.json contains vecm_coef, tracked |
| bvar / statsmodels | (no row) | partial (posterior==OLS diffuse-limit, atol=1e-6) | **pi** — real property assertion exists |
| iv / linearmodels | verified | (their verified, mine verified-but-printed-only) | **pi stricter**: no assert in mroz_iv.py; both should say partial |
| verified-count definitions | 5 verified | 10 verified | agy's stricter line (excludes same-lib printed-only parities); mine over-charitable |

## F merge

Keep (both plans, merged by phase; dedup overlap — both plans share: error hierarchy, logging NullHandler, result objects, registry dispatch, ModelBase decoupling, random-state, docs sync, CI hardening, conftest fixtures, mypy, deprecation):

| # | PR (merged) | Risk | Effort | Breaking |
| --- | --- | --- | --- | --- |
| 1 | pandas-3 upsample fix + regression test (pi#1) | L | S | n |
| 2 | pandas CI matrix + pin <4 (pi#2, supersedes agy#3's 3.10 widening — keep 3.12 floor to avoid churn) | L | S | n |
| 3 | P0 model fixes batch: TVECM ECT off-by-one, Minnesota prior inversion ×3, SVEC fallback, robust_ols cov_type, ARIMA kwargs, VECM rank, event-study NaN, FE sort/swallow, DiD controls (agy release-blockers + pi) | H | M | y (numbers change) |
| 4 | README/docs snippet quickstart smoke test in CI (pi#17) | L | S | n |
| 5 | Stop `self.params` clobber → `params_df_` (pi#8, inside agy#10) | M | S | n |
| 6 | API surface fix: add missing `__all__` entries + reclassify ARIMA kind (pi#3, agy#13) | L | S | y (minor) |
| 7 | exceptions/warnings hierarchy (agy#6 + pi#4 merged) | L | M | n |
| 8 | logging → NullHandler (both #7) | L | S | n |
| 9 | single config loader + pydantic schema (pi#6, agy#20 merged) | M | M | n |
| 10 | random-state helper + remove global np.random (both #8) | L | M | y (test seeds) |
| 11 | result-object hierarchy phase 1 (regression/discrete) then phase 2 (systems, kill `{"error":...}`) (both #11) | M | M | y (JSON schema) |
| 12 | registry v2 declarative dispatch (pi#11, agy#12) | H | M | n |
| 13 | mypy: fix 185 (--check-untyped-defs) + make blocking; annotate public modules (agy#9 + pi#12/13) | L | L | n |
| 14 | integration tests → wrappers (both) + assert vecm golden json + volatility xfail resolution (pi) | L | M | n |
| 15 | conftest fixtures + parity marker (pi#21, agy#15) | L | M | n |
| 16 | packaging: force-include → package-data glob with explicit package paths + drop duplicate parquets (pi#16, agy#2/5, correct their 15MB claim) | H | M | n |
| 17 | repo hygiene: src/temp/src/archive/reports-archive out (pi#14, agy#1) | H | S | y (paths) |
| 18 | notebook outputs stripped (pi#15) | L | S | n |
| 19 | docs build: mkdocs + registry-generated models.md (pi#18, agy#19) | L | M | n |
| 20 | CI gates: coverage→80, ruff format, multi-OS (agy#18, pi#20) | M | M→L | n |
| 21 | deprecation policy + classifier bump (pi#19, agy#14/22) | L | M | y (future) |
| 22 | validate pipeline configs against registry at load (subset of #9) | M | S | n |

## Confidence
High on A/B/C/D verdicts (each re-verified by rerun or code read on current main); medium on C verdicts vs agy's audited snapshot (main.tex rewritten by PR #83 mid-review — two of their P0s now stale); high on D disagreements (file+test evidence).

## Not checked
- agy C claims about PRIMEarxiv geometry/keywords rendering (no TeX here)
- agy D `archive/` 83M provenance (untracked tree, not inspected)
- diff between agy-audited main.tex snapshot and current (no snapshot hash recorded in their file)
- ghysels chap5/ch10 script contents (only existence checks)