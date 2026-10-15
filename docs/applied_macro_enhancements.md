# Applied Macro Features Guide

User-facing overview of the applied-macro enhancements in `stats-transformer` and the bundled academic replication benchmarks. All features are available in the released package; the replication data ships with the package and loads via `load_example()`.

---

## 1. Macro-grade `LocalProjectionsModel`

Jordà (2005) local projections for macro applications: cumulative impulse responses, lag augmentation, and Newey-West standard errors.

- `cumulative` controls the dependent variable of each horizon regression:
  - `False` (default): levels, $y_{t+h}$.
  - `True`, `"cumulative"`, or `"baseline"`: $y_{t+h} - y_{t-1}$ (cumulative response relative to the pre-shock baseline).
  - `"diff"` or `"change"`: $y_{t+h} - y_t$ (long difference).
  - Unknown string values raise `ValueError`.
- `lags=P` injects lags $t-1, \dots, t-P$ of the target, the shock, and every control into each horizon regression.
- `cov_type="HAC"` fits Newey-West standard errors with bandwidth $h+1$ per horizon; a user-supplied `maxlags` in `cov_kwds` applies to every horizon. `"standard"` selects classical OLS standard errors.

```python
from stats_transformer.models.timeseries.reduced_form.local_projections import LocalProjectionsModel

lp = LocalProjectionsModel(
    target="y", shock_var="shock", controls=["x1", "x2"],
    horizon=36, cumulative=True, lags=4, cov_type="HAC",
)
lp.fit(df)
irf = lp.compute_irf()  # columns: horizon, effect, stderr, lower_ci, upper_ci, pvalue
```

## 2. `PCAModel` scaling toggles

Yield-curve and factor decompositions (Diebold & Li, Nelson-Siegel, Stock-Watson) typically demean without dividing by standard deviation, so that variance scale carries economic meaning.

- `standardize=False`: demean only, skip unit-variance normalization.
- `with_mean=False`: skip demeaning as well; `PCAModel` then performs a non-centered SVD (sklearn's `PCA` always centers) instead.

```python
from stats_transformer.models.unsupervised.unsupervised import PCAModel

pca = PCAModel(features=columns, n_components=3, standardize=False)
pca.fit(df)
```

## 3. Direct multi-step forecasts (`VARForecaster.forecast_direct`)

Iterated VAR forecasts recurse one step at a time; direct forecasts estimate a separate $y_{t+h}$ regression per horizon to protect against lag-specification error at long horizons.

```python
from stats_transformer.models.timeseries.reduced_form.forecasting import VARForecaster

fc = VARForecaster.forecast_direct(df, target_variables=["y1", "y2"], lags=2, steps=4)
```

## 4. Out-of-sample prediction (`RegressionModel.predict`)

Applies fitted coefficients to new data, automatically appending a constant when the fitted model includes an intercept. Accepts a `DataFrame`, a `Series` (single row), a NumPy array, or a list of rows. For models fit with entity fixed effects, the entity column is required in `new_data` and entities unseen in training raise `ValueError` (no silent zero-effect predictions).

```python
model = RegressionModel(target="y", independent_variables=["x1", "x2"])
model.fit(df)
model.predict(pd.DataFrame({"x1": [1.0], "x2": [2.0]}))
model.predict(pd.Series({"x1": 1.0, "x2": 2.0}))
```

---

## Bundled replication benchmarks

Both replications live in `src/examples/academic/` and are covered by `tests/test_academic_replications.py` at the tolerances stated below.

### Hoogteijling, Martens & van der Wel (2026)

Replicates the Table 2 row "Diebold and Li" (out-of-sample $R^2$ across 6 maturities, $n = 2, 3, 4, 5, 7, 10$ years) using `PCAModel(standardize=False)` and `RegressionModel`. Asserts each replicated $R^2$ against the published values within `0.0015` (the replication's `TOLERANCE`).

```python
from examples.academic.vanderwel_2026 import Vanderwel2026Replication
result = Vanderwel2026Replication().run()
```

### Nakamura & Steinsson (2018b), Appendix local projections

Replicates the Jordà (2005) monetary-policy multiplier at horizon 36 and the IRF paths for IP, CPI, and the real policy rate. Each horizon regression is estimated with `RegressionModel` on a hand-built design (lag blocks of the macro series plus shock levels and lags), following the paper's specification. Assertions, matching the replication's tolerances:

- multiplier at $h=36$ within `0.01` of the published `1.95` (replicated: `1.9471`),
- IRF paths within a maximum absolute difference of `0.01` of the authors' published IRF paths (`RomerJordaIRF.dta`) across the first 49 horizons.

```python
from examples.academic.nakamura_steinsson_2018b_jorda import NakamuraSteinsson2018bJordaReplication
result = NakamuraSteinsson2018bJordaReplication().run()
```