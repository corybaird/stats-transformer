# Planned & Implemented Models Catalog

This document provides a detailed, triaged catalog of all models, structural identification paradigms, diagnostic suites, and counterfactual tools in `stats-transformer`.

---

## 1. Overview & Triage Matrix

The model extensions are organized into four implementation tiers based on algorithmic complexity, optimization requirements, and dependency depth.

```
  [ Tier 1: Reduced-Form Baseline ] ──► Linear VAR, VECM, Lag Selection, Diagnostics
                 │
                 ▼
  [ Tier 2: Structural Identification ] ──► Exact/Over-ID, Sign/Zero, Narrative, SVEC
                 │
                 ▼
  [ Tier 3: Data-Driven SVAR ] ──────► Volatility Breaks, Distance Covariance, Non-Gaussian ML
                 │
                 ▼
  [ Tier 4: Non-Linear Dynamics ] ───► TVAR, TVECM, STVAR, GIRF Simulations
```

| Triage Tier | Complexity | Primary Benchmarks | Target Models & Tools |
| --- | --- | --- | --- |
| **Tier 1: Linear Reduced-Form** | Low / Baseline | R `vars`, `statsmodels` | OLS VAR, VECM, Restricted VAR, Lag Selection (`VARselect`), Residual Diagnostics, Companion Stability |
| **Tier 2: Structural Identification** | Medium | R `vars`, VAR-Toolbox | Short/Long-Run SVAR, Blanchard-Quah, SVEC, Sign & Zero Restrictions, Narrative Restrictions, Bootstrap Bounds |
| **Tier 3: Data-Driven SVAR** | Medium / High | R `svars` | Changes in Volatility (Rigobon), Distance Covariance (ICA), Cramér-von Mises (CVM), Non-Gaussian ML, Permutation/Sign Alignment |
| **Tier 4: Non-Linear & Regime-Switching** | High | R `tsDyn`, R `sstvars` | Threshold VAR (TVAR), Threshold VECM (TVECM), Smooth Transition VAR (STVAR), Generalized IRF (GIRF) |

---

## 2. Tier 1: Linear Frequentist Reduced-Form Models

These models represent the foundational estimation engine. All subsequent structural and non-linear identification methods build upon reduced-form residuals and covariance matrices.

### 2.1 Reduced-Form OLS VAR (`VARModel`)

- **Description**: Vector Autoregressive model estimating a system of $K$ endogeneous variables on $p$ lags.
- **Specification**: $Y_t = A_1 Y_{t-1} + \dots + A_p Y_{t-p} + C D_t + u_t$, where $u_t \sim \mathcal{N}(0, \Sigma_u)$.
- **Deterministic Terms**: Supported options: `const`, `trend`, `both`, `none`.
- **Estimation Method**: Equation-by-equation OLS or Seemingly Unrelated Regression (SUR).
- **Module Location**: `src/stats_transformer/models/timeseries/reduced_form/var.py`
- **Benchmark Target**: R `vars::VAR` / Python `statsmodels.tsa.vector_ar.var_model.VAR`.

### 2.2 Cointegrated Vector Error Correction Model (`VECMModel`)

- **Description**: Multivariate model capturing long-run equilibrium relationships (cointegration) alongside short-run dynamics.
- **Specification**: $\Delta Y_t = \Pi Y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta Y_{t-i} + C D_t + u_t$, where $\Pi = \alpha \beta'$.
- **Cointegration Testing**: Johansen Trace and Maximum Eigenvalue rank tests.
- **Module Location**: `src/stats_transformer/models/timeseries/reduced_form/vecm.py`
- **Benchmark Target**: R `vars::ca.jo` and `vars::vec2var`.

### 2.3 Restricted VAR (`RestrictedVARModel`)

- **Description**: Reduced-form VAR enforced with zero coefficient restrictions at the equation level.
- **Specification**: Mask matrix $M \in \{0, 1\}^{K \times (K \cdot p + m)}$ zeroing specific lag or deterministic coefficients.
- **Module Location**: `src/stats_transformer/models/timeseries/reduced_form/restrictions.py`
- **Benchmark Target**: R `vars::restrict` / Kilian & Lütkepohl (2017) Ch. 3.

### 2.4 Lag Selection Engine (`LagSelection`)

- **Description**: Automatic lag length selection across a user-defined range $[1, p_{\max}]$.
- **Information Criteria**: Akaike ($AIC$), Hannan-Quinn ($HQ$), Schwarz/Bayesian ($SC/BIC$), and Final Prediction Error ($FPE$).
- **Module Location**: `src/stats_transformer/models/timeseries/reduced_form/lag_selection.py`
- **Benchmark Target**: R `vars::VARselect`.

### 2.5 Diagnostic Suites

- **Residual Autocorrelation**: Portmanteau and adjusted Portmanteau $Q$-tests.
- **Multivariate Normality**: Jarque-Bera, skewness, and kurtosis tests.
- **ARCH Heteroskedasticity**: Multivariate ARCH-LM test.
- **Stationarity & Roots**: Modulus of eigenvalues of the companion matrix $A_c$.
- **Module Location**: `src/stats_transformer/models/timeseries/diagnostics/`

---

## 3. Tier 2: Structural Identification & SVEC

Structural VAR models isolate structural shocks $\epsilon_t$ from reduced-form errors $u_t$ using linear, inequality, or narrative constraints ($u_t = B \epsilon_t$ or $A u_t = B \epsilon_t$).

### 3.1 Short-Run & Long-Run SVAR (`SVARModel`, `BlanchardQuahModel`)

- **Short-Run Restrictions**: Cholesky triangular decomposition or explicit structural constraints on matrix $A$ and $B$ such that $A \Sigma_u A' = B B'$.
- **Long-Run Restrictions**: Restrictions placed on the long-run cumulative impact matrix $C(1) = A(1)^{-1} B$ (Blanchard & Quah 1989).
- **Module Location**: `src/stats_transformer/models/timeseries/identification/svar.py`, `blanchard_quah.py`
- **Benchmark Target**: R `vars::SVAR` / Blanchard & Quah (1989).

### 3.2 Structural VECM (`SVECModel`)

- **Description**: Combining cointegration rank restrictions $\beta$ with structural short-run and long-run restrictions.
- **Identification**: Separates permanent shocks (matching cointegration rank $r$) from transitory shocks.
- **Module Location**: `src/stats_transformer/models/timeseries/structural/svec.py`
- **Benchmark Target**: R `vars::SVEC` / King, Plosser, Stock, and Watson (1991).

### 3.3 Sign & Zero Restrictions (`SignZeroSVARModel`)

- **Description**: Non-Bayesian set identification imposing directional inequality signs ($+ / -$) and exact zeros ($0$) on impulse response paths.
- **Algorithm**: QR decomposition of random orthonormal matrices $Q$ (Rubio-Ramírez, Waggoner, & Zha 2010).
- **Inference**: Draw acceptance loop generating structural shock candidates; median-target selection and quantile bounds.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/sign_zero.py`
- **Benchmark Target**: VAR-Toolbox `Uhlig2005` / Rubio-Ramírez et al. (2010).

### 3.4 Narrative Restrictions (`NarrativeSVARModel`)

- **Description**: Constraining the sign or magnitude of structural shocks and historical contributions during specific historical dates (e.g. 1973 Oil Crisis, 2008 Financial Crisis).
- **Module Location**: Integrated into `sign_zero.py` engine.
- **Benchmark Target**: Antolín-Díaz & Rubio-Ramírez (2018).

### 3.5 Structural Bootstrap Engines (`BootstrapInference`)

- **Resampling Methods**: Residual bootstrap, Wild bootstrap (Rademacher / Normal weights), and Moving-Block bootstrap for time-series dependence.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/bootstrap.py`

---

## 4. Tier 3: Data-Driven SVAR Identification

Data-driven structural identification methods leverage statistical properties of reduced-form errors (heteroskedasticity, independence, non-Gaussianity) to identify structural impact matrices without requiring explicit theoretical restrictions.

### 4.1 Changes in Volatility (`HeteroskedasticSVARModel`)

- **Description**: Identification via discrete volatility regime shifts across known break dates.
- **Specification**: Reduced-form error covariance matrix shifts across regimes: $\Sigma_u^{(1)} = B B'$ and $\Sigma_u^{(2)} = B \Lambda B'$, where $\Lambda$ is diagonal.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/volatility.py`
- **Benchmark Target**: Rigobon (2003) / R `svars::id.cv`.

### 4.2 Distance Covariance & ICA (`IndependenceSVARModel`)

- **Description**: Identification by maximizing structural shock independence via Distance Covariance minimization or FastICA.
- **Objective**: Minimize pairwise distance covariance between structural shock components: $\sum_{i < j} \mathcal{V}^2(\epsilon_i, \epsilon_j)$.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/independence.py`
- **Benchmark Target**: Matteson & Tsay (2017) / R `svars::id.dc`.

### 4.3 Cramér-von Mises Distance (`CVMSVARModel`)

- **Description**: Identification by testing mutual independence using the Cramér-von Mises distance metric.
- **Optimization**: Fast $O(T \log T)$ approximation (Huo & Székely 2016) for sample sizes where $O(T^2)$ calculation is computationally intensive.
- **Benchmark Target**: R `svars::id.cvm`.

### 4.4 Non-Gaussian Maximum Likelihood (`NonGaussianSVARModel`)

- **Description**: Identification assuming structural shocks follow non-Gaussian distributions (e.g. Student-t, Mixture of Normals).
- **Estimation**: Direct Maximum Likelihood estimation over the unconstrained mixing matrix $B$.
- **Benchmark Target**: Lanne, Meitz, & Saikkonen (2010) / R `svars::id.ng`.

### 4.5 Permutation & Sign Alignment (`AlignmentEngine`)

- **Description**: Post-processing algorithm resolving column ordering and sign ambiguities across bootstrap iterations.
- **Algorithm**: Hungarian algorithm matching structural shock columns to a reference impact matrix.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/alignment.py`

---

## 5. Tier 4: Non-Linear & Regime-Switching Models

Nonlinear multivariate models relax the assumption of linear, state-invariant dynamics to capture asymmetric responses, financial stress regimes, or business cycle threshold shifts.

### 5.1 Threshold VAR (`TVARModel`)

- **Description**: Two-regime Threshold VAR model where system dynamics switch based on an observed threshold variable $y_{t-d}$ relative to threshold value $\gamma$.
- **Specification**: 
  $$Y_t = (A^{(1)}_0 + \sum_{i=1}^p A^{(1)}_i Y_{t-i}) I(y_{t-d} \le \gamma) + (A^{(2)}_0 + \sum_{i=1}^p A^{(2)}_i Y_{t-i}) I(y_{t-d} > \gamma) + u_t$$
- **Estimation**: Grid search over threshold candidate values $\gamma$ and delay parameters $d$.
- **Module Location**: `src/stats_transformer/models/timeseries/nonlinear/tvar.py`
- **Benchmark Target**: R `tsDyn::TVAR`.

### 5.2 Threshold VECM (`TVECMModel`)

- **Description**: Combining threshold regime-switching behavior with long-run cointegrated Vector Error Correction dynamics.
- **Specification**: Threshold search applied to error correction term $z_{t-1} = \beta' Y_{t-1}$.
- **Module Location**: `src/stats_transformer/models/timeseries/nonlinear/tvecm.py`
- **Benchmark Target**: R `tsDyn::TVECM`.

### 5.3 Smooth Transition VAR (`STVARModel`)

- **Description**: Continuous regime-switching model with smooth transition weights defined by a logistic or exponential function $G(y_{t-d}; \gamma, c)$.
- **Transition Function**: $G(y_{t-d}; \gamma, c) = (1 + \exp(-\gamma (y_{t-d} - c)))^{-1}$.
- **Module Location**: `src/stats_transformer/models/timeseries/nonlinear/stvar.py`
- **Benchmark Target**: R `sstvars::STVAR`.

### 5.4 Generalized Impulse Response Functions (`GIRFEngine`)

- **Description**: History-dependent, state-conditional impulse responses for non-linear models where responses depend on shock sign, magnitude, and initial state.
- **Simulation**: Monte Carlo history-based simulation comparing shocked trajectories against baseline trajectories (Koop, Pesaran, & Potter 1996).
- **Module Location**: `src/stats_transformer/models/timeseries/nonlinear/girf.py`
- **Benchmark Target**: R `sstvars::GIRF`.

---

## 6. Counterfactuals & Diagnostic Suite

### 6.1 Historical Decompositions (`HistoricalDecomposition`)

- **Description**: Decomposing historical series trajectories into cumulative structural shock contributions: $Y_t = \sum_{j=0}^{t-1} \Theta_j \epsilon_{t-j} + \text{Initial Conditions}$.
- **Output**: Long DataFrame suitable for stacked bar visualization.

### 6.2 Counterfactual Conditional Forecasting

- **Description**: Simulating system trajectories under hypothetical shock scenarios (e.g. setting specific structural shocks to zero or enforcing target path constraints).

---

## 7. Deferred Scope & Ecosystem Monitoring

### 7.1 Explicitly Deferred Scope

- **Bayesian VAR & SVAR**: Bayesian estimation (BVAR, Gibbs sampling, MCMC, priors) is explicitly deferred to preserve a lightweight, frequentist codebase.
- **High-Dimensional Regularization**: Penalized VARs (LASSO, Ridge) are deferred to a future high-dimensional proposal.

### 7.2 Technical Backlog & Implementation Scope

- **Phase 3 Data-Driven SVAR Backlog**:
  - Cramér-von Mises (CVM) identification (`CVMSVARModel`).
  - Non-Gaussian Maximum Likelihood estimation.
  - Fast $O(T \log T)$ distance covariance approximation (Huo & Székely 2016).
- **Phase 4 Non-Linear Models Backlog**:
  - Vectorized candidate grid search for TVAR threshold variable selection.
  - Direct Johansen cointegration rank integration for TVECM estimators.
  - Multi-threaded parallel Monte Carlo simulation engine for GIRF computations.

### 7.3 Ecosystem Monitoring (`tsecon`)

- Features present in external packages (such as `tsecon` Rust/PyO3 implementation) like FAVAR, Diebold-Yilmaz connectedness measures, MIDAS, and panel time series (MG/CCE-MG) are tracked for potential future consideration but are not active parity targets.

