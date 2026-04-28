# stats-transformer Publish, Refactor, & Robust Testing Plan

This plan bridges the goals of refactoring the repository architecture, adding robust example datasets, and running a safe, PyPI-ready publishing pipeline. All work strictly adheres to the `github-agent` branch/commit hygiene workflow and the `pypi-publisher` release methodologies.

## User Review Required

> [!WARNING]  
> We need to decide what specific example datasets to add for testing before executing this plan.
>
> **Questions:**
> 1. For the `Refactoring` phase, do you have preferred subfolder names in mind for `models/` and `visualization/` (e.g., grouping models by `regression`, `panel`, `unsupervised`), or should I propose the detailed structure?
> 2. What missing datasets should we use as examples for robust testing? Can we use local mock data, or do we need to pull specific datasets?
> 3. What should the target version number be for the PyPI release (e.g., `v0.1.0`)?

## Proposed Changes

We will execute this work across distinct execution phases, enforcing the rule that all changes happen on scoped feature/release branches and land via PR.

---

### Phase 1: Refactor Models & Visualization Packages

**Goal:** Clean up the internal structure to be robust and professional prior to publishing, specifically targeting the cluttered `models` and `visualization` directories.

1. **Branching:** Create `refactor/reorganize-modules` off `main`.
2. **Refactor `src/stats_transformer/models`:**
   - Reorganize the 6+ files (`regression.py`, `panel.py`, `robust_ols.py`, `unsupervised.py`, etc.) into logical subdirectories (e.g., `regression/`, `panel/`, `unsupervised/`).
   - Clean up the `__init__.py` to ensure top-level imports remain unaffected (or intentional breaking changes are documented).
3. **Refactor `src/stats_transformer/visualization`:**
   - Create subfolders for the 7+ files (e.g., group `model_viz.py`/`regression_viz.py`, `data_viz.py`/`eda.py`, and `viz_utils.py`).
   - Fix internal imports.
4. **Committing & PR:**
   - Logically group the commits (e.g., `REFACTOR organize models`, `REFACTOR organize visualization`, `FIX internal cross-imports`).
   - Push and open a PR. **The user manually merges.**

---

### Phase 2: Robust Testing and Dataset Integration

**Goal:** Provide more example datasets to make sure `stats-transformer` is tested robustly.

1. **Branching:** Create `feat/add-example-datasets` off `main`.
2. **Dataset Additions:**
   - Add new datasets into `tests/data/` or `data/raw/` (depending on the repository's convention for example data).
3. **Test Refactor:**
   - Update `tests/` to consume the new datasets, validating the core transformer pipeline handles them edge-case free across all refactored models and plots.
4. **Committing & PR:**
   - Split commits logically (e.g., one commit for raw dataset additions, one for test updates).
   - Push and open a PR. **The user manually merges.**

---

### Phase 3: PyPI Readiness Audit and Release Prep

**Goal:** Ensure the project meets modern PyPI publishing standards and cut the release.

1. **Branching:** Create `release/v<TARGET_VERSION>` off `main`.
2. **Audit & Fixes (via pypi-publisher rules):**
   - `pyproject.toml`: Audit metadata (authors, description, urls, dependencies). Assure `[build-system]` is configured.
   - `README.md`: Ensure a quick-start example uses the newly added datasets.
   - `CHANGELOG.md`: Generate notes for this release (including the refactor).
   - `src/`: verify `__version__` and `__all__` imports are clean. Marker files like `py.typed`.
3. **Publish CI/CD Setup:**
   - Add/Verify `.github/workflows/publish.yml` for PyPI Trusted Publishing via OIDC on tag events.
4. **Committing & PR:**
   - Logical commits (e.g., `UPDATE pyproject.toml bounds`, `ADD publish.yml workflow`, `DOCS update changelog bounds`).
   - Push and open a PR. **The user manually merges.**

---

### Phase 4: The PyPI Release (Manual trigger)

1. Once the `release/v<TARGET_VERSION>` PR is merged into `main`, the user locally applies the Git tag:
   ```bash
   git tag v<TARGET_VERSION>
   git push origin v<TARGET_VERSION>
   ```
2. The GitHub Action OIDC workflow will handle the build and secure publish to PyPI.

## Verification Plan

### Automated Tests
- We will run `uv run pytest` locally after the refactor and testing phases to ensure the test suite is green.
- The CI workflow must run `pytest` cleanly before the PyPI build process begins.

### Manual Verification
- After successful OIDC publish, we will smoke-test the package in a clean temporary directory:
  ```bash
  mkdir /tmp/stats-transformer-smoketest && cd /tmp/stats-transformer-smoketest
  uv init && uv add stats-transformer
  uv run python -c "import stats_transformer; print(stats_transformer.__version__)"
  ```
