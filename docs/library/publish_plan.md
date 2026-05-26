# stats-transformer Publishing Plan

This plan tracks the work needed to move `stats-transformer` from a functional research library to a publishable package. The repository is close: the package builds, the local test suite passes, and a PyPI Trusted Publishing workflow exists. The remaining work is mostly release hardening, metadata polish, and public-facing documentation.

## Current Readiness

Status: **almost ready for a first public package release, not yet ready for a broad announcement.**

Verified locally:

- `uv run pytest`: 34 passed, 1 skipped.
- `uv build`: wheel and source distribution build successfully.
- `.github/workflows/publish.yml`: publishes on `v*.*.*` tags using PyPI Trusted Publishing.

Release blockers to resolve before tagging:

- **Python version mismatch:** `pyproject.toml` requires Python `>=3.12`, but CI tests Python 3.10 and 3.11. Either lower `requires-python` after compatibility verification or change CI to 3.12/3.13 only.
- **Installed usage smoke test:** confirm `uv add stats-transformer` in a clean environment, then import `stats_transformer`, run a minimal pipeline, and import chart/table helpers.
- **Public API audit:** decide which classes are stable enough for `stats_transformer.__all__`, and keep experimental models accessible through submodules if their APIs may change.
- **Runtime dependency audit:** move notebook-only and development-only packages out of runtime dependencies where possible.
- **README examples:** keep examples focused on installed-package imports (`stats_transformer`, not `src.stats_transformer`) and use a small bundled example path that works after installation.
- **Package data audit:** confirm bundled files needed at runtime, such as `py.typed`, visualization styles, and package example data, are included in the wheel.
- **Project metadata polish:** add keywords, project URLs for documentation/issues, and confirm the license classifier matches the license file.

## Release Questions

Before cutting the release:

1. Is the next public tag `v1.0.0`, matching `pyproject.toml`, or should the project start with a lower public version such as `v0.1.0` while APIs settle?
2. Should example datasets be shipped inside the package, kept only in the repository, or split into a separate examples artifact later?
3. Which planned extensions are promised in documentation, and which should remain explicitly experimental?

## Execution Plan

Use scoped branches and PR review for release-facing changes.

---

### Phase 1: Release Readiness Audit

**Goal:** remove blockers that can break installation, CI, or first-user experience.

1. Align `requires-python`, CI matrix, and lockfile metadata.
2. Audit package contents with `uv build` and inspect the wheel.
3. Smoke-test a clean install in a temporary directory.
4. Check that README code paths and imports work outside the repository.
5. Review runtime dependencies and move optional/tutorial dependencies behind extras if appropriate.

### Phase 2: Robust Testing and Dataset Integration

**Goal:** make examples and tests representative enough for public users.

1. Keep tiny deterministic fixtures in `tests/data/`.
2. Keep larger pedagogical datasets under `data/examples/` or package data only when they are needed by installed examples.
3. Add pipeline tests that cover:
   - single-country and multi-country panels;
   - missing dates and uneven frequencies;
   - categorical fixed effects;
   - model output serialization;
   - visualization generation in a headless environment.
4. Add one installed-package smoke test script that users can copy from the README.

### Phase 3: PyPI Release Prep

**Goal:** make the tag publish cleanly and make the project page credible.

1. Update `CHANGELOG.md` with final release notes.
2. Confirm `pyproject.toml` metadata, classifiers, project URLs, and dependency bounds.
3. Confirm GitHub repository settings include a PyPI Trusted Publisher environment named `pypi`.
4. Run:
   ```bash
   uv run pytest
   uv build
   ```
5. Open a release PR and merge after CI is green.

### Phase 4: The PyPI Release

1. Tag the release:
   ```bash
   git tag v<TARGET_VERSION>
   git push origin v<TARGET_VERSION>
   ```
2. Confirm the GitHub Action builds and publishes to PyPI.
3. Smoke-test the published package:
   ```bash
   tmpdir=$(mktemp -d)
   cd "$tmpdir"
   uv init
   uv add stats-transformer
   uv run python -c "import stats_transformer; print(stats_transformer.__version__)"
   ```

## Verification Plan

- Automated: `uv run pytest`, `uv build`, CI matrix, publish workflow on tag.
- Manual: clean install, README quickstart, one notebook smoke run, PyPI project page review.
- Documentation: README, changelog, architecture docs, and planned extensions should describe current capabilities separately from future work.
