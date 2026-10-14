# Applied Macro Enhancements & Academic Replications Guide
*Specification and implementation roadmap for branch `feat/applied-macro-enhancements` in `stats-transformer`*

---

## 1. Overview & Objectives

During empirical replications against published macroeconomic literature (*Hoogteijling, Martens & van der Wel, 2026* and *Nakamura & Steinsson, 2018b*), `stats-transformer` proved its underlying numerical accuracy (reproducing paper outputs bit-identically to 6 decimal places). However, several key macro-econometric patterns required manual workarounds outside the library.

This feature branch delivers:
1. **Four core applied macro enhancements** to `LocalProjectionsModel`, `PCAModel`, `VARForecaster`, and `RegressionModel`.
2. **Two new verified academic replication benchmarks** integrated into `src/examples/academic/`.
3. **Bundled replication datasets** registered into `src/stats_transformer/data/registry.py`.
4. **Top-level model exports** in `src/stats_transformer/__init__.py`.

---

## 2. Data Migration: Files, Sources, and Target Folders

To enable self-contained automated tests and reproducible academic examples, copy the following datasets into the `stats-transformer` repository structure:

### Summary Table of Data to Copy

| Paper Reference | Source File(s) in Workspace | Destination in `stats-transformer` | File Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Vanderwel et al. (2026)** | `/workspace/replications/vanderwel-2026/replication/python/data/LW_monthly.csv` | `data/examples/academic/vanderwel_2026/LW_monthly.csv` | ~2.2 MB | Monthly smoothed GSW Treasury yields (maturities 1–120m, 1971:08–2018:12). |
| **Nakamura & Steinsson (2018b)** | `/workspace/other_replications/macroeconomics_replication/data/original_file_structure/nakamura_steinsson_2018_b/Replication/Appendix/Program/temp/MonetaryData.txt` | `data/examples/academic/nakamura_steinsson_2018_b/MonetaryData.txt` | ~19 KB | Monthly macro series (`logIP`, `UE`, `logCPI`, `logCMPI`, `FFR`, 1969m1–1996m12). |
| **Nakamura & Steinsson (2018b)** | `/workspace/other_replications/macroeconomics_replication/data/original_file_structure/nakamura_steinsson_2018_b/Replication/Appendix/Program/temp/RomerShock.txt` | `data/examples/academic/nakamura_steinsson_2018_b/RomerShock.txt` | ~3.2 KB | Romer & Romer (2004) monetary policy shock series. |
| **Nakamura & Steinsson (2018b)** *(Validation Benchmark)* | `/workspace/other_replications/macroeconomics_replication/data/original_file_structure/nakamura_steinsson_2018_b/Replication/Appendix/Program/temp/RomerJordaIRF.dta` | `data/examples/academic/nakamura_steinsson_2018_b/RomerJordaIRF.dta` | ~4.8 KB | Authors' published benchmark IRF paths across 50 horizons for regression assertions. |

---

### Step-by-Step Shell Commands to Copy the Data

Run these commands from the root of `stats-transformer`:

```bash
# 1. Create target directories
mkdir -p data/examples/academic/vanderwel_2026
mkdir -p data/examples/academic/nakamura_steinsson_2018_b

# 2. Copy Vanderwel et al. (2026) data
cp /workspace/replications/vanderwel-2026/replication/python/data/LW_monthly.csv \
   data/examples/academic/vanderwel_2026/LW_monthly.csv

# 3. Copy Nakamura & Steinsson (2018b) data
NS_SRC=/workspace/other_replications/macroeconomics_replication/data/original_file_structure/nakamura_steinsson_2018_b/Replication/Appendix/Program/temp
cp "$NS_SRC/MonetaryData.txt" data/examples/academic/nakamura_steinsson_2018_b/
cp "$NS_SRC/RomerShock.txt"   data/examples/academic/nakamura_steinsson_2018_b/
cp "$NS_SRC/RomerJordaIRF.dta" data/examples/academic/nakamura_steinsson_2018_b/
```

*(Optional Parquet Optimization: For package distribution size efficiency, `LW_monthly.csv` can also be stored as `LW_monthly.parquet.gzip` (~600 KB) matching other datasets in `data/examples/academic/`)*.

---

### Data Registration in `src/stats_transformer/data/registry.py`

Add entries to `EXAMPLE_DATASETS`:

```python
"vanderwel_2026": {
    "description": "Monthly smoothed Gürkaynak-Sack-Wright Treasury yield curve maturities (1 to 120 months)",
    "time_period": "Monthly (1971-08 to 2018-12)",
    "source": "Hoogteijling, Martens & van der Wel (2026), IJF",
    "files": {
        "yields": _file("academic/vanderwel_2026/LW_monthly.csv"),
    },
},
"nakamura_steinsson_2018_b": {
    "description": "Monthly macro series and Romer-Romer monetary policy shocks for local projections",
    "time_period": "Monthly (1969-01 to 1996-12)",
    "source": "Nakamura & Steinsson (2018b), JME Appendix",
    "files": {
        "macro_data": _file("academic/nakamura_steinsson_2018_b/MonetaryData.txt"),
        "shocks": _file("academic/nakamura_steinsson_2018_b/RomerShock.txt"),
        "published_irfs": _file("academic/nakamura_steinsson_2018_b/RomerJordaIRF.dta"),
    },
},
```

---

## 3. Core Econometric Enhancements Specification

### Enhancement 1: Macro-grade `LocalProjectionsModel`
**File:** `src/stats_transformer/models/timeseries/reduced_form/local_projections.py`

#### Motivation
Canonical macroeconomic local projections (Jordà 2005; Ramey 2016; Nakamura & Steinsson 2018) estimate cumulative impulse responses using multi-lag control blocks and Newey-West standard errors:
$$Y_{t+h} - Y_t = \alpha^{(h)} + \beta^{(h)} \text{Shock}_{t+1} + \sum_{p=1}^P \gamma_p^{(h)} X_{t-p} + \varepsilon_{t+h}$$

#### Specification Changes
Update `__init__` and `build_model`:
```python
class LocalProjectionsModel(ModelBase):
    def __init__(
        self,
        target: str,
        shock_var: str,
        horizon: int = 8,
        controls: Optional[List[str]] = None,
        date_column: Optional[str] = None,
        cumulative: bool = False,       # NEW: Y(t+h) - Y(t) instead of Y(t+h)
        lags: int = 0,                  # NEW: inject p lags of target, shock, & controls
        cov_type: str = "HAC",          # NEW: 'HAC' (Newey-West) default for macro LPs
        cov_kwds: Optional[dict] = None,# NEW: e.g. {'maxlags': h + 1}
        **kwargs: Any
    ):
```
- When `cumulative=True`, the dependent variable is $Y_{t+h} - Y_t$ (or $Y_{t+h+1} - Y_t$).
- When `cov_type='HAC'`, fit with `cov_type='HAC', cov_kwds={'maxlags': h + 1}` to handle serial correlation introduced by overlapping forecast errors.

---

### Enhancement 2: Demean-Only Toggle in `PCAModel`
**File:** `src/stats_transformer/models/unsupervised/unsupervised.py`

#### Motivation
In yield-curve and macroeconomic factor analysis (e.g., Diebold-Li, Nelson-Siegel, Stock-Watson), yields are demeaned but **not** divided by their standard deviations. Dividing by standard deviation distorts the economic meaning of the Level, Slope, and Curvature components.

#### Specification Changes
```python
class UnsupervisedModel(ModelBase):
    def __init__(
        self,
        ...,
        standardize: bool = True,  # NEW: allows disabling variance-scaling
        with_mean: bool = True,    # NEW: demeaning toggle
        **kwargs: Any
    ):
        ...
        self.standardize = standardize
        self.with_mean = with_mean
        self.scaler = StandardScaler(with_mean=self.with_mean, with_std=self.standardize)
```
- Setting `standardize=False` performs pure demeaning without dividing by variance.

---

### Enhancement 3: Direct Forecasting in `VARForecaster`
**File:** `src/stats_transformer/models/timeseries/reduced_form/forecasting.py`

#### Motivation
While iterated VAR forecasts recurse one step at a time ($y_t \to y_{t+1} \to \dots \to y_{t+h}$), empirical papers often use *direct multi-step forecasting* ($y_{t+h} = \alpha + \beta y_t$) to protect against lag specification error at long horizons.

#### Specification Changes
Add a direct forecast method:
```python
def forecast_direct(self, df: pd.DataFrame, target: str, features: List[str], horizons: List[int]) -> pd.DataFrame:
    """Estimates direct h-step predictive regressions for each horizon in horizons."""
```

---

### Enhancement 4: Prediction Accessor & Fast-Path in `RegressionModel`
**File:** `src/stats_transformer/models/regression/regression.py`

#### Motivation
Currently, after calling `model.fit(df)`, there is no `predict(new_df)` method. Users must extract `model.get_results().params`, manually align columns, prepend constants, and calculate matrix products. In rolling backtests, `fit()` also incurs overhead by copying dataframes and computing AIC/BIC on every iteration.

#### Specification Changes
```python
class RegressionModel(ModelBase):
    def predict(self, new_data: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
        """Applies fitted model parameters to new_data, handling constant addition automatically."""
        if self.model is None:
            raise ValueError("Model must be fit before calling predict()")
        # Extract params, format design matrix, return dot product
        ...

    def fit(self, df: pd.DataFrame, drop_na: bool = True, compute_metrics: bool = True) -> Dict[str, Any]:
        """compute_metrics=False skips AIC/BIC/condition number calculations for fast loops."""
```

---

## 4. New Academic Examples & Regression Tests

Add the verified replication scripts into `src/examples/academic/`:

1. **`src/examples/academic/vanderwel_2026.py`**:
   - Replicates Table 2 row "Diebold and Li" out-of-sample $R^2$ across 6 maturities ($n=2, 3, 4, 5, 7, 10$ years):
     `[-0.236, -0.192, -0.165, -0.147, -0.141, -0.059]`.
   - Uses `PCAModel(standardize=False)` and `RegressionModel`.

2. **`src/examples/academic/nakamura_steinsson_2018b_jorda.py`**:
   - Replicates Jordà Local Projections multiplier at $h=36$ ($1.9471 \approx 1.95$).
   - Validates IRF paths across 50 horizons against `RomerJordaIRF.dta`.
   - Uses `LocalProjectionsModel(cumulative=True, cov_type='HAC')`.

3. **`tests/test_academic_replications.py`**:
   - Unit tests running both replications and asserting numerical tolerance ($< 0.0015$ and $< 10^{-4}$).

---

## 5. Verification Checklist

- [ ] Data copied to `data/examples/academic/vanderwel_2026/` and `data/examples/academic/nakamura_steinsson_2018_b/`.
- [ ] Datasets registered in `src/stats_transformer/data/registry.py` and documented in `docs/library/data.md`.
- [ ] `LocalProjectionsModel` supports `cumulative=True`, `lags=P`, and `cov_type='HAC'`.
- [ ] `PCAModel` supports `standardize=False`.
- [ ] `RegressionModel` has working `.predict(new_df)`.
- [ ] `src/stats_transformer/__init__.py` exports all primary model classes at top-level.
- [ ] Unit tests pass: `uv run pytest tests/test_academic_replications.py`.
- [ ] Linting clean: `uv run ruff check .`.
