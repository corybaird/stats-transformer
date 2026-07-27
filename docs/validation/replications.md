# Academic Replications & Verification Registry

This document serves as the master registry for all models in `stats-transformer`, tracking their numerical replication status, exact code locations, input datasets, and mathematical tolerance limits against established software ecosystems (MATLAB, R, Python reference libraries, and Stata).

---

## 1. What "Verified" Means

In `stats-transformer`, a status of **✅ Verified** indicates that an automated PyTest suite or script executes both the Python model and the reference implementation (or frozen benchmark output) on identical input data and asserts **numerical equivalence within strict tolerance limits** (`rtol=1e-4` to `1e-10`). 

A status of **🚧 Demo Built** means an executable script exists replicating the paper's figures/data pipeline, but automated cross-language numerical assertion tests are pending.

---

## 2. Master Model Replication & Location Matrix

| Model Class | Python Test Location | Reference Script / Package | Input Dataset | Target Object Verified | Numerical Tolerance | Status |
|---|---|---|---|---|---|---|
| **`VARModel`** | `tests/replication/r_packages/var/var.py` | R `vars::VAR` (`var.R`) | `canada.csv` | Coef matrix $\hat{A}$ & SEs | `rtol=1e-6` | ✅ Verified |
| **`VECMModel`** | `tests/replication/kilian/vecm.py` | MATLAB `mle_unknown_beta.m` | `realgnp.txt` | Residual cov $\hat{\Sigma}_u$ | `rtol=1e-4` | ✅ Verified |
| **`LocalProjectionsModel`** | `tests/replication/var_toolbox/local_projections.py` | MATLAB `GO_JT2025.m` | `JT2025_Data.xlsx` | Impulse response vector $IR(h)$ | `rtol=1e-4` | ✅ Verified |
| **`LocalProjectionsIVModel`** | `tests/timeseries_extensions.py` | Stata `ivreg2` / MATLAB | `JT2025_Data.xlsx` | Direct horizon-by-horizon LP-IV | Shape test | 🚧 Demo Built |
| **`RestrictedVAR`** | `tests/models/var_specification.py` | MATLAB (Kilian Ch 3) | Synthetic | Zero-constrained $\hat{A}$ | Parity test | 🚧 Demo Built |
| **`SVARModel`** | `tests/replication/kilian/svar.py` | MATLAB `figure9_1.m` | `data.m` | Cholesky IRFs $IRF(h)$ | `rtol=1e-5` | ✅ Verified |
| **`BlanchardQuahModel`** | `tests/replication/var_toolbox/bq.py` | MATLAB VAR-Toolbox | `BQ1989_Data.xlsx` | Long-run impact matrix $B_0$ | `rtol=1e-10` | ✅ Verified |
| **`ProxySVARModel`** | `tests/replication/var_toolbox/proxy_svar.py` | MATLAB `GO_GK2015.m` | `GK2015_Data.xlsx` | Normalized impact column $b_1$ | `rtol=1e-4` | ✅ Verified |
| **`VolatilitySVARModel`** | `tests/replication/r_packages/svar/svar.py` | R `svars::id.cv` (`svar.R`) | `data_usa.csv` | Volatility eigenvalues $\Lambda$ | `rtol=1e-2` | ✅ Verified |
| **`SignZeroSVARModel`** | `tests/models/structural_restrictions.py` | MATLAB (Kilian Ch 12 / Uhlig 2005) | Synthetic | Sign-satisfying rotation $Q$ | Draw test | 🚧 Demo Built |
| **`IndependenceSVARModel`** | `tests/models/data_driven_svar.py` | R `svars::id.ngv` | Synthetic | Non-Gaussian impact matrix | Shape test | 🚧 Demo Built |
| **`SVEC`** | `tests/models/svec.py` | R `vars::SVEC` | Synthetic | Cointegrated structural $B_0$ | Rank test | 🚧 Demo Built |
| **`RobustOLSModel`** | `tests/replication/r_packages/regression/regression.py` | R `sandwich::vcovHC` | `ex2_regress_gdp_us.csv` | HC0, HC1, HC2, HC3 SEs | `rtol=1e-6` | ✅ Verified |
| **`PanelRegressionModel`** | `tests/replication/r_packages/panel/panel.py` | R `plm` (`panel.R`) | `grunfeld.csv` | FE & RE coefs and SEs | `rtol=1e-6` | ✅ Verified |
| **`IV2SLSModel`** | `tests/replication/r_packages/regression/regression.py` | R `AER::ivreg` (`regression.R`) | `ex2_regress_gdp_us.csv` | 2SLS Coefs & SEs | `rtol=1e-6` | ✅ Verified |
| **`LogitModel`** | `tests/replication/r_packages/regression/regression.py` | R `stats::glm` (`regression.R`) | `ex2_regress_gdp_us.csv` | Logit MLE Coefs | `rtol=1e-5` | ✅ Verified |
| **`PCAModel`** | `tests/models/unsupervised.py` | `scikit-learn` | Synthetic | Explained Variance Ratio | `rtol=1e-8` | ✅ Verified |
| **`KMeansModel`** | `tests/models/unsupervised.py` | `scikit-learn` | Synthetic | Cluster Centers & Inertia | `rtol=1e-8` | ✅ Verified |

---

## 3. Running Automated Replications

To run all automated replication checks locally:

```bash
uv run pytest tests/replication/ -v
```

> [!NOTE]
> External engine dependencies (e.g. `matlabengine` or `Rscript`) are dynamically checked. If an engine is missing, PyTest safely skips those specific benchmarks to ensure CI/CD compatibility.

---

## 4. Executable Academic Paper Demonstrations

The repository also includes standalone executable empirical demonstrations in `src/examples/academic/` replicating published macro papers:

| Paper | Script Location | Dataset Location | Target Object Verified | Status |
|---|---|---|---|---|
| **Nakamura & Steinsson (2018)** | `src/examples/academic/nakamura_steinsson.py` | `data/examples/academic/nakamura_steinsson_2018/` | Stata `master.dta` yield diffs | ✅ Verified (Stata Parity) |
| **Bauer & Swanson (2023)** | `src/examples/academic/bauer_swanson.py` | `data/examples/academic/bauer_swanson_2023/` | MATLAB log-diff & lag outputs | ✅ Verified (MATLAB Parity) |
| **Bauer, Bernanke & Milstein (2023)** | `src/examples/academic/bauer_bernanke_milstein.py` | `data/examples/academic/bbm_2023/` | Daily monetary surprise transformations | ✅ Verified (Python Parity) |
| **Stock & Watson (2001)** | `src/examples/academic/var/stock_watson_2001.py` | Bundled `statsmodels` Macrodata | Macro VAR(4) Table 1 & Fig 1 | ✅ Executable Demo |
| **Blanchard & Quah (1989)** | `src/examples/academic/var/blanchard_quah_1989.py` | `data/examples/matlab_examples/BQ1989_Data.xlsx` | Supply/demand long-run $B_0$ | ✅ Executable Demo |
| **Gertler & Karadi (2015)** | `src/examples/academic/var/gertler_karadi_2015.py` | `references/matlab_benchmarks/Replic/GK2015/` | FF4 futures Proxy SVAR $b_1$ | ✅ Executable Demo |
| **Jordà & Taylor (2025)** | `src/examples/academic/var/jorda_taylor_2025.py` | `references/matlab_benchmarks/Replic/JT2025/` | Local Projections Fig 5a & 6a | ✅ Executable Demo |

Run any academic demonstration directly via module execution:

```bash
/opt/homebrew/bin/uv run python -m src.examples.academic.nakamura_steinsson
/opt/homebrew/bin/uv run python -m src.examples.academic.var.gertler_karadi_2015
```
