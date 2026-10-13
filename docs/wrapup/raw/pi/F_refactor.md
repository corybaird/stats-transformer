# Task F: refactor & long-horizon best practices (pi)

## Current state

- **Packaging**: hatchling; version 1.6.0 with `Development Status :: 3 - Alpha` classifier while CHANGELOG/roadmap target "v1" (`pyproject.toml:13`); `requires-python >=3.12`; deps are lower-bound-only, no upper pins; `pandas>=2.2.3` but environment resolves 3.0.2 and `_upsample_entity` breaks on pandas 3 groupby semantics (B P0) — no pandas 3 CI matrix
- **Data packaging**: 15 `force-include` stanzas ×2 targets (sdist + wheel) copying `data/examples/**` into two different layouts; fragile — registry fallback path `examples/{path}` vs wheel layout `stats_transformer/data/examples/` only meets accidentally (`src/stats_transformer/data/__init__.py:57-79`)
- **Extras**: `benchmarks` (rpy2), `notebooks` (jupyter), `all` — clean; `mypy`/`ruff` only in `[dependency-groups].dev` (not installable via extra) (`pyproject.toml:88-92`)
- **Public API**: top-level `__init__` imports `IV2SLSModel`, `PanelIV2SLSModel` but omits both from `__all__` (A P2); `SVEC` class + `SVECModel` both exported (one is internals); `RestrictedVAR`, `SVARBootstrap`, `GIRFEngine`, diagnostics exported from `models` — API surface undifferentiated vs utilities; registry `NOT_PIPELINE_EXPOSED` only covers subclasses
- **Registry design**: plain dict + `kind` string dispatch through 9 `if` branches in `Pipeline._build_from_params`/`_build_from_args` (`pipeline.py:63-121`); kind names leak ("svar_family" includes ARIMA); models have heterogeneous constructor signatures forcing per-kind branches (LP needs `target,shock_var` positional, DiD takes column names, VAR takes `target_variables`) — the "unified contract" claimed in docs is real only for regression family
- **Result objects**: no result classes — `fit` returns loose dicts; `self.model` holds raw statsmodels/sklearn results, plain dicts (`{"beta":...}`), or even `self` (TVAR sets `self.model = self` AND `self.params = self.regime_1_df` clobbering the YAML dict, A P1); metadata schema inconsistent per model (some no `coefficients`, SVAR emits `{"error": ...}`)
- **Typing**: `py.typed` present but honest only for 5 mypy-enabled files (7 of ~109 files have any `->` annotation, ~6%); mypy `continue-on-error: true` in CI (`ci.yml:31`) so gate is decorative; pyproject comment references missing `docs/extensions/roadmap.md`
- **Errors/warnings**: zero custom exceptions/warnings (grep = 0); string-matching error retry in `ModelBase.fit` (`"NaN" in str(e)`); per-column exceptions logged-and-dropped in FeatureEngineer; bare `except: pass` in panel.py
- **Logging**: three bespoke `_setup_logging` implementations (ModelBase, FeatureEngineer, DataMerger) each adding stdout handlers and a Jupyter suppression hack (`setLevel(CRITICAL+1)`) — library attaching handlers at import/instantiation is the classic mistake
- **Randomness**: `np.random.seed(42)` global calls alongside `default_rng(seed)` instances; `KMeans(random_state=42)` hard-coded; no global seeding policy; MC bands in paper figures seeded implicitly
- **Config**: three YAML loaders (Config, ModelBase._load_params, FeatureEngineer._load_params) with different key paths (`data.featurization` vs `featurization` fallback); `Config.validate` exists but models re-read raw dicts again downstream; dot-notation `Config.get` only used by Pipeline
- **Tests**: 410 tests, 84% coverage, gate 72; layout mirrors repo loosely; fixtures ad hoc (`create_synthetic_data` inline); 4 R-parity tests skip silently (env-gated); numeric coverage concentrated in same-lib parity
- **Clutter**: `src/temp/` (504K conversion scripts, `__pycache__` tracked-able), `src/archive/`, `data/archive/`, `reports/archive/`, `data/temp/`, `data/final/` (100M, gitignored), `docs/wrapup/`; `reports/overleaf` (paper) fully gitignored; notebooks 1.8M with big outputs committed (936K single file); `references/matlab_benchmarks` 191M gitignored silently
- **CI**: ci.yml (ruff only F821/F822/E9, pytest+cov 72 floor, mypy continue-on-error), parity.yml (nightly R docker), publish.yml; no docs build (no mkdocs/sphinx), no wheels matrix, no pandas-version matrix
- **Deprecation policy**: none — breaking changes silently landed across 1.2→1.6; no `DeprecationWarning` anywhere; semver claimed (1.x with breaking changes inside x)

## Target state

- API surface like sklearn/statsmodels: thin top-level namespace, per-area subpackages with explicit `__all__`; utilities under `_internal` or doc-marked non-stable; single `__version__` source
- Result objects per estimator family returning dataclasses (sklearn `results_classes_`-style): `OLSResults` with `.params/.bse/.pvalues/.conf_int/.summary/.to_dict` — one schema, JSON round-trip, no `{"error": ...}` escape hatches
- Registry as declarative table mapping model_type → (cls, spec) with the spec describing constructor wiring — kill per-kind branches in Pipeline
- Single `errors.py` + `warnings.py` module: `DataValidationError`, `SpecificationError`, `IdentificationError`; warnings via `warnings.warn(category=...)` not logger.error
- Logging via `logging.getLogger(__name__)` only, library never adds handlers; expose `logging.getLogger("stats_transformer")` null-ish default
- Config: one loader, one validation pass, one schema (pydantic or typed dataclass), dot-keys dead, `params` dict passed to models read-only
- Seeding policy: single `random_state` kwarg everywhere (int or Generator), `check_random_state`-style helper like sklearn
- Typing: py.typed made honest — full annotations for public modules, mypy strict on `models/base.py`, `pipeline.py`, `registry.py`; drop `continue-on-error`
- CI: matrix (pandas 2.3 / pandas 3.0), mypy blocking, docs build (mkdocs via mkdocstrings), coverage floor 80
- Repo hygiene: `src/temp`, `src/archive`, duplicated archives out of git; CI runs quickstart YAML from README (guards the class of P0 found in B)

## Phased PR plan

| # | PR title | Scope | Risk | Effort | Breaking |
| --- | --- | --- | --- | --- | --- |
| 1 | Fix pandas-3 upsample crash + add annual→monthly regression test | featurization + tests | L (touches quickstart path) | S | n |
| 2 | Add pandas 2.3 + 3.0 CI matrix; pin `pandas>=2.2.3,<4` | pyproject, ci.yml | L | S | n |
| 3 | Export fix: add IV2SLSModel/PanelIV2SLSModel to `__all__`; drop `SVEC` from public `__all__` (keep import path) | `__init__.py` | L (SVEC public import) | S | y (minor) |
| 4 | `errors.py`/`warnings.py` + replace string-matching retry & log-and-drop with library exceptions | models/base, feature_engineering | M (error-message contracts) | M | n (new warnings logged) |
| 5 | Consolidate `_setup_logging` into one `utils/logging.py`; remove handler attachment | base, feature_engineering, data_merger | L (visible log behavior change) | S | n |
| 6 | Single config loader: `Config` becomes canonical; delete `_load_params` duplicates; one schema-shape | utils/config, base, feature_engineering, data_merger | M | M | n |
| 7 | `random.py` helper (check_random_state); plumb seed through KMeans/LP-IV/diagnostics; remove `np.random.seed` globals | models + featurization | L (test seeds) | M | y (n_clusters semantics unchanged, defaults kept) |
| 8 | Stop `self.params` clobber in TVAR/STVAR/TVECM → `params_df_` attribute; restore YAML dict on refit | 3 nonlinear models + tests | M | S | n |
| 9 | Result objects phase 1: `fit()` keeps signature but adds `results_` dataclass with fixed schema for regression + logit/probit/unsupervised | regression, discrete, unsupervised | M | M | n |
| 10 | Result objects phase 2: system models (VAR/VECM/SVAR family) uniform `coefficients/equations/irf` schema; kill `{"error":...}` extraction fallback (raise + warn instead) | timeseries models, visualization adapters | M | L | y (metadata JSON schema) |
| 11 | Registry v2: declarative spec `field(constructor_kwargs=...)` replacing `kind` branches; drop "svar_family" mislabel (ARIMA → "univariate") | registry.py, pipeline.py | H (all model_type paths) | M | n (config keys stable) |
| 12 | mypy: annotate + strict-check registry.py; enable `--strict` on base/pipeline; make mypy blocking in CI; delete stale `docs/extensions/roadmap.md` ref | pyproject, 3 modules | L (annotation churn nearby) | M | n |
| 13 | typing sweep: annotate public modules (regression, discrete, unsupervised, timeseries base files); add `_internal` marker on utils | src/stats_transformer/** | M | L | n |
| 14 | Move `src/temp` + `src/archive` + `reports/archive` + `data/archive` out of git (into research repo or delete); add `.gitignore` patterns; move paper-figure inputs under `reports/overleaf` CI-tracked? | repo hygiene | H (paths referenced by docs) | S | y (paths) |
| 15 | Stop committing notebook outputs (strip widgets/figures, jupyter nbstripout hook); shrink 936K file | notebooks, .gitignore, docs | L (links in README) | S | n |
| 16 | Unify data-path resolution: registry entries carry explicit `package_path` matching wheel layout; delete the 30 `force-include` stanzas for a single package-data glob | pyproject, data/registry.py, data/__init__.py | H (wheel layout) | M | n (paths inside package unchanged) |
| 17 | README quickstart CI: run the documented YAML+stages in ci.yml as smoke test (would have caught B P0) | ci.yml, tests | L | S | n |
| 18 | Docs build: mkdocs + mkdocstrings from docstrings; models.md generated from registry (kills the 32/34 count drift) | docs, pyproject, ci.yml | L | M | n |
| 19 | Deprecation policy: `DeprecationWarning` helper + docs note ("deprecated in 1.x, removed 2.0"); bump classifier 3-Alpha → 4-Beta at v1 tag | pyproject, docs, CHANGELOG | L | S | n |
| 20 | Coverage floor 72→80; enable ruff default lint gradually (per-file pragmas, roadmap the ~250 violations) | pyproject, ci.yml | M | L | n |
| 21 | Tests: shared `conftest.py` fixtures for synthetic panels/TS; port `create_synthetic_data` duplicates; mark R-parity with dedicated marker (`-m parity`) instead of env-var skip | tests/** | L | M | n |
| 22 | `xarray` dependency audit: if only used in one helper, demote to optional extra or drop | pyproject, featurization | L | S | n |

- Order: 1-9 are v1-blocker-adjacent quick wins (each ≤M); 10-13 form the "statsmodels-grade API" arc; 14-19 hygiene/docs before tagging v1; 20-22 continuous.

## Not covered here (cross-refs)
- Numerical bugs & validation triage → `docs/wrapup/raw/pi/A_code.md`, `D_validation.csv`; doc fixes → `B_docs.md`; release sequencing → `E_roadmap.md`