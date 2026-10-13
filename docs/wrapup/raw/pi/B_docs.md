# Task B: docs accuracy (pi)

## Scope checked
- README.md, CHANGELOG.md, docs/overview.md, docs/roadmap.md, docs/library/{models,architecture,data,citations}.md, docs/validation/{benchmarks,testing_suite}.md
- Executed: README quickstart (all 4 stages + CLI full run), sample example modules, benchmark suite, wheel build + installed-package `load_example` test
- Cross-checked counts vs MODEL_REGISTRY (31 keys / 33 ModelBase subclasses) and data registry (34 `load_example` keys)

## P0

| Severity | Area | Claim | Evidence | Fix hint |
| --- | --- | --- | --- | --- |
| P0 | README quickstart | Documented quickstart crashes: `Pipeline.run(stage="resample")` and `uv run python -m stats_transformer.pipeline --config params.yaml` (full run) both fail `KeyError: 'country'` in `_upsample_entity` — modern pandas groupby excludes the key column, `_upsample_entity` assumes it present | `feature_engineering.py:388` `group[entity_col].iloc[0]`; fixture run /tmp/bdocs with README §3.2 params.yaml → `KeyError: 'country'` at feature_engineering.py:388; exit≠0. Downsampling path (monthly→quarterly) works; every annual→finer config crashes | Guard: re-attach entity col from `group.name` (as `_apply_entity_transformations` already does) |
| P0 | benchmarks.md §3 | Documented MATLAB comparator command dead: module `src.examples.software_benchmarks.matlab_comparator` does not exist anywhere in repo → `ModuleNotFoundError: No module named 'src.examples.software_benchmarks.matlab_comparator'` | run 2026-10-09; `find src -name matlab_comparator*` empty; §3 command at bottom of benchmarks.md §3 | Point at `software_benchmarks.timeseries.blanchard_quah_benchmark` or create module |

## P1

| Severity | Area | Claim | Evidence | Fix hint |
| --- | --- | --- | --- | --- |
| P1 | Model counts | Three different counts for same thing, all wrong: README `:33` + overview `:27/:40` say "32 model classes"; models.md tables list 34 rows; registry has 31 keys + 3 direct-only classes (SpecificationRunner, RestrictedVAR, GIRFEngine); actual ModelBase subclasses = 33 (incl. abstract UnsupervisedModel) | `grep "32"` README.md:33; `uv run` subclass census | State one count with definition; wire it to the doc-consistency test |
| P1 | overview §3 | "Every model is accessible via YAML pipeline configuration (`model.model_type`) or direct Python" — false: SpecificationRunner, RestrictedVAR, GIRFEngine, GrangerCausalityTester, 4 diagnostics classes are not registry-reachable; models.md itself marks them `None` | `docs/overview.md:40` vs registry.py (no `SpecificationRunner` entry) vs models.md §1 "YAML Pipeline Key: None" | Reword to "all 31 registry models are pipeline-dispatchable" |
| P1 | README §3.2 YAML | Quickstart config uses `frequency: Q` — not a key the code reads; `resample_dataset` reads `source_period`/`resample.target_period`; dataset silently unresampled (or crashes via P0) | README.md:57-58 vs `feature_engineering.py:426-428` (source_period, target_period); data.md §3.2 repeats the dead key (`frequency: Q`) | Rename to `source_period: Q` + add `resample:` block; test quickstart YAML in CI |
| P1 | architecture §2/§4.2 | File structure ships `tests/verification/` ("Opt-in external verification (MATLAB)") — directory does not exist (removed in CHANGELOG 1.6.0) | `ls tests/verification` → exit 2; CHANGELOG 1.6.0 "Removed ... dead `tests/verification/` stubs"; same row in testing_suite.md:43 | Delete from both docs |
| P1 | architecture §4.2 | "`ModelBase` interface: fit / **predict** / get_model_metadata" — ModelBase has no `predict` abstract; two concrete models define their own; Pipeline.predict raises NotImplementedError for the rest | `models/base.py` (abstracts: build_model/get_summary/get_model_metrics only); `pipeline.py:180` | Drop predict from the contract, or make docs match reality |
| P1 | benchmarks §2.1 "Verified" | Machine-precision claims vs Stata/MATLAB have no recorded artifact in repo; benchmark suite run in container yields "No active engine" for all 7 benchmarks (only rpy2-backed R parity exists); engine modules expect local StataNow/macOS paths | run: `uv run python -m src.examples.software_benchmarks.benchmark_suite` → all `No active engine`; `modules/stata_engine.py:7` `/Applications/StataNow/utilities`; grep for recorded discrepancies (2.22e-16, 5.77e-15 …) → only in docs, no JSON/log | Record engine outputs (env, date, max-diff) in repo, or scope claims to R-parity integration tests |
| P1 | benchmarks §2.2 table | "2.2 Planned External Software Benchmarks" contains `RobustOLSModel`… with status *Planned* — but R HC1 robust-SE parity is already in `tests/integration/regression/R` (HC1 assert) | benchmarks.md §2.2 row 1 vs `tests/integration/regression/regression.R` + `test_regression.py:36-41` (HC1 allclose) | Promote to verified for R |
| P1 | benchmarks §4 | "all 27 runnable example modules" — table has 31 rows, `src/examples` has 55 non-init .py files; 11 catalogued "runnable" replications have no `__main__` → `python -m` exits 0 with zero output | count vs docs; `grep -L __main__` → src/examples/academic/{coibion_gorodnichenko_2012,jarocinski_karadi_2020,lane_2025,...}.py (11 files); run `python -m src.examples.academic.coibion_gorodnichenko_2012` → silent exit 0 | Add `if __name__ == "__main__"` runners; recount |
| P1 | testing_suite | "The test suite does not require MATLAB, network access, or external academic replication data" vs same doc §Optional local checks + 2 test_pipeline skips caused by missing `references/configs/timeseries/identification/structural_restrictions.yaml` model fields | testing_suite.md:23 & 24 pytest skips (structural_restrictions.yaml) | Verify skip-free default suite, or rephrase |
| P1 | testing_suite/integration | "multi-language integration tests (R, Stata)" — Dockerfile installs R only; Stata integration does not exist | `tests/integration/Dockerfile:4-13` (r-base only); no Stata artifacts in tests/integration/ | Drop Stata mention |
| P1 | CHANGELOG 1.6.0 | Coverage claim "64% → 80.35% (78 tests to 274)" — suite now has 410 tests; stale snapshot presented as current | CHANGELOG.md:47 vs `uv run pytest --collect-only -q` → 410 | Reword as point-in-time or update |
| P1 | CHANGELOG 1.6.0 | Changelog references `docs/extensions/models.md` ("corrects five class names") — file absent from repo | `ls docs/extensions` → missing; CHANGELOG.md:75 | Remove ref or restore file |
| P1 | data.md §2 | Table missing 3 registered dataset keys: `ghysels_ch5`, `ghysels_ch8`, `ghysels_ch10` (registered, files shipped, 34 keys vs 31 rows) | `list_examples()` output (34) vs data.md table row count (31) | Add rows |
| P1 | architecture §2 | Tree includes root-level `models/` dir ("serialized model artifacts") — not in repo | `ls models` → absent | Drop from tree |
| P1 | models.md §3.4/§4.2 | "K-Means ... Configure the number of clusters via `model.kmeans.n_clusters`" (skill models.md) and KMeans "Stata cluster kmeans" parity "Implemented" — `n_clusters` not YAML-readable (always 3, random_state=42 hard-coded) | skill `.claude/.../models.md` KMeans row; pipeline `unsupervised` branch passes features only; `unsupervised.py:100` | Wire config; adjust doc status |

## P2

| Severity | Area | Claim | Evidence | Fix hint |
| --- | --- | --- | --- | --- |
| P2 | benchmarks commands | All suite commands hard-code `/opt/homebrew/bin/uv` (dev Mac) — breaks plain `uv run` convention used elsewhere in docs | benchmarks.md §2/§3, testing_suite.md §Run | Use `uv run` |
| P2 | citations §2.1 | Wrong dataset location: "Location: `data/examples/macrodb_gdp_inflation.parquet`" — actual shipped path `src/stats_transformer/data/macrodb_gdp_inflation.parquet` (data.md §1 says so) | citations.md:172 vs data.md:6 | Fix path |
| P2 | citations §2.2 | Stock-Watson "Location" points at example script, not the dataset (`data/examples/matlab_examples/SW2001_Data.xlsx`); description "from 1960 to 2000" unverified | citations.md:180-182 vs data.md table | Point at data file |
| P2 | citations §3 | "MATLAB VAR-Toolbox: Cesa-Bianchi, Ambrogio. (2026)" — year is current (today 2026-10), not future; earlier "future-dated" note withdrawn; citation omits co-author Pierferdinando Ferroni (VAR-Toolbox is Cesa-Bianchi & Ferroni) | citations.md §3 | Add co-author; same missing co-author in main.tex bibitem vartoolbox |
| P2 | citations §3 | Empty citation stub: "R `sstvars` Package:" with no content | citations.md §3 | Fill or drop |
| P2 | overview §3 | "Discrete Choice ... and marginal effects" — no marginal-effects code anywhere (logit/probit return prsquared/llr only) | overview.md:44; grep marginal in src → none | Drop or implement |
| P2 | overview §3 | Claims "32 ... classes across five functional domains" — table itself lists 7 domains | overview.md:40 vs rows below | Fix "five" |
| P2 | CHANGELOG 1.6.0 | "Pipeline.predict() now raises NotImplementedError instead of AttributeError; no model in the library implements predict()" — false: PandemicBVARModel.predict, BayesianSignZeroSVARModel.predict exist | CHANGELOG.md Changed vs grep "def predict" (2 hits) | Update claim |
| P2 | CHANGELOG 1.4/1.3.1 | 1.3.1 added `TimeSeriesDecompositions` / "tests/verification/matlab_comparator.py"; 1.6.0 removed that directory — history fine, but README §4 and architecture.md still describe install script and skill for tools the script targets (`claude|codex|kilo` only, no generic "all tools" copy to `.claude/`) | scripts/install-agent-skill.sh:24 targets .kilo; SKILL source `.agents/skills/...` while working copy also at `.claude/skills/...` (manually maintained, not by script) | Make script author `.claude/` too or document which is source of truth |
| P2 | architecture §4.3 | Claims metadata contains "covariance matrices" — metadata dict has coefficients/std_err only, no covariance | architecture.md:107 vs `base.py get_model_metadata` | Reword |
| P2 | architecture §3 | Tree omits `visualization/formatters/` (exists) and `models/registry.py` labeled "YAML dispatcher" (it's only a lookup table; dispatch lives in pipeline.py) | `ls visualization` vs architecture.md tree; pipeline.py dispatch | Add formatters; reword |
| P2 | data.md §3.2 | YAML sample lacks `raw_data_file`/`output_path` needed by pipeline stages (`features` raises "No data path found" without merge output having run) | data.md:113-126 vs pipeline.run effective_data_path logic | Make sample complete or note prerequisites |
| P2 | models.md §1 inventory | `SVARModel` row says benchmark "Blanchard & Quah data" while §6.1 too; `VolatilitySVARModel` benchmark "Rigobon Break Panel (grunfeld.csv)" — grunfeld is not a volatility-break panel | models.md §1.1 rows | Sanity-check dataset associations |
| P2 | models.md §5.3 | `RestrictedVAR` "Direct Python API: `RestrictedVAR`" but not exported from `stats_transformer.models` `__all__`? It IS (`__all__` includes) — doc fine; however docs don't mention `RestrictedVARResults` companion — nit | `models/__init__.py` | Skip or add |

## Snippet/execution ledger (run 2026-10-09)

| Snippet | Result |
| --- | --- |
| README `pip install stats-transformer` | N/A (offline) |
| README `load_example("macrodb_gdp_inflation")` / `list_examples()` | OK (34 keys) |
| README §3.3 stage run "resample" | FAIL KeyError 'country' (P0) |
| README §3.3 stage run "features" | OK from existing merged parquet |
| README §3.3 stage run "regression" (panel_ols) | OK, metrics returned |
| README §3.3 stage run "visualization" | OK (empty results, no model summary available) — works only because stage="visualization" tolerated missing summary; see A-report P0 truthiness crash when transformed_data set without summary |
| README §3.3 CLI `python -m stats_transformer.pipeline --config params.yaml` | FAIL (KeyError 'country) |
| benchmarks §2 `benchmark_suite` | Runs; all engines "unavailable" in container → no parity numbers |
| benchmarks §3 `matlab_comparator` module | ModuleNotFoundError (P0) |
| examples regression: longley, mroz_iv, grunfeld, macro_var, spector_logit | All run, self-checks pass, print conclusions |
| examples academic: coibion_gorodnichenko_2012, jarocinski_karadi_2020, ghysels_chap6, var/gertler_karadi_2015 | exit 0 but ZERO output (no `__main__`) — "runnable" claim misleading (P1) |
| wheel build + install test | 54/54 data files packaged; installed `load_example('longley')` + macrodb OK |
| data.md §3.1 `load_example("bauer_swanson_2023", member="cpi")` | OK |

## Validation-claim backing (benchmarks.md / models.md "Verified" rows)

| Claimed row | Backing script | Recorded result artifact | Reproducible here? |
| --- | --- | --- | --- |
| RegressionModel vs R/Stata/MATLAB | `tests/integration/regression/{regression.R,test_regression.py}` + `src/examples/regression/longley.py` | regression_results.json ✓ (R only) | R-only; Stata/MATLAB numbers unbacked |
| PanelRegressionModel vs R `plm` | `tests/integration/panel/` | panel_results.json ✓ | Skip (no R in container) |
| LogitModel vs R/Stata | `tests/integration/regression/` | regression_results.json ✓ | R-only |
| PCAModel vs R/Stata | none found | none | No artifact |
| VARModel vs Stata/R | `tests/integration/var/` | var_results.json ✓ | R-only |
| VECMModel vs Stata/MATLAB | `tests/integration/var/` (vecm_coef in json) | ✓ partial | R-derived |
| BlanchardQuahModel vs MATLAB | benchmark exists (`timeseries/blanchard_quah_benchmark.py`) | none (engine output never persisted) | No |
| SVARModel vs R `svars` | `tests/integration/svar/` | svars_results.json ✓ | Skip |
| VolatilitySVARModel vs R `id.cv` | `tests/integration/svar/` | svars_results.json ✓ | Skip |
| TVARModel vs Stata threshold | `src/examples/…/tvar_benchmark.py` | none | No artifact |

→ 6/10 "Verified" rows have recorded JSON backing (all R-derived); 4 rows (PCA, BQ, TVAR, and all Stata/MATLAB columns) have no persisted artifact in repo.

## Contradictions matrix

| Where | Says | Where | Says |
| --- | --- | --- | --- |
| overview.md:40 | "all 32 classes … accessible via YAML model.model_type" | models.md §1 | several rows `YAML Pipeline Key: None`; registry has 31 keys |
| testing_suite.md:24,43 | `tests/verification/` exists, "not collected by default" | CHANGELOG 1.6.0 | removed `tests/verification/` stubs; dir absent |
| CHANGELOG 1.6.0 | "no model in the library implements predict()" | code | 2 models implement predict |
| roadmap §6.1 | "General Bayesian MCMC Sampling … explicitly deferred" | code | BayesianSignZeroSVARModel samples posterior draws per rotation (MCMC-free but Monte Carlo) | 
| roadmap §6.2 | RestrictedVAR not exposed via params.yaml | models.md §5.3 | consistent (None) — OK |
| README:33 | "32 implemented model classes and utilities" | models.md tables | 34 rows |
| benchmarks §4 intro | "27 runnable example modules" | table | 31 rows |
| data.md §1 | wheel includes data/examples | ✓ verified TRUE | — |

## Confidence
High on counts, dead commands, no-`__main__` catalog (all executed); medium on which "Verified" rows warrant retraction (engine runs may exist on dev machine, just not in repo).

## Not checked
- notebooks cell-by-cell execution (08 already known broken per ruff F821 in A-report; would need kernels + heavy reruns)
- docs/archive/library/* (explicitly archived) + reports/overleaf tables (A/C/D territory)
- references/configs/*.yaml full sweep (only the ones implicated by skips/tests)
- citations.md §4.1 literature accuracy (bibliographic vetting out of scope)
- pyproject test-path config claims beyond py312 target (ruff/mypy "CI gates" not inspectable from repo state)