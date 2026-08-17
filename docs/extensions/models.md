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

| Triage Tier | Complexity | Primary Benchmarks | Target Models & Tools | Status |
| --- | --- | --- | --- | --- |
| **Tier 1: Linear Reduced-Form** | Low / Baseline | R `vars`, `statsmodels` | OLS VAR, VECM, Restricted VAR, Lag Selection (`VARselect`), Residual Diagnostics, Companion Stability | **Implemented** |
| **Tier 2: Structural Identification** | Medium | R `vars`, VAR-Toolbox | Short/Long-Run SVAR, Blanchard-Quah, SVEC, Sign & Zero Restrictions, Narrative Restrictions, Bootstrap Bounds | **Implemented** |
| **Tier 3: Data-Driven SVAR** | Medium / High | R `svars` | Changes in Volatility (Rigobon), Distance Covariance (ICA), Cramér-von Mises (CVM), Non-Gaussian ML, Permutation/Sign Alignment | **Implemented**; permutation matching *Planned* |
| **Tier 4: Non-Linear & Regime-Switching** | High | R `tsDyn`, R `sstvars` | Threshold VAR (TVAR), Threshold VECM (TVECM), Smooth Transition VAR (STVAR), Generalized IRF (GIRF) | **Implemented** |

Per-model status is given in each section heading below. **Implemented** means the class exists and is importable; *Planned* means it is described here but not yet in the codebase.

---

## 2. Tier 1: Linear Frequentist Reduced-Form Models

These models represent the foundational estimation engine. All subsequent structural and non-linear identification methods build upon reduced-form residuals and covariance matrices.

### 2.1 Reduced-Form OLS VAR (`VARModel`) — **Implemented**

- **Description**: Vector Autoregressive model estimating a system of $K$ endogeneous variables on $p$ lags.
- **Specification**: $Y_t = A_1 Y_{t-1} + \dots + A_p Y_{t-p} + C D_t + u_t$, where $u_t \sim \mathcal{N}(0, \Sigma_u)$.
- **Deterministic Terms**: Supported options: `const`, `trend`, `both`, `none`.
- **Estimation Method**: Equation-by-equation OLS or Seemingly Unrelated Regression (SUR).
- **Module Location**: `src/stats_transformer/models/timeseries/reduced_form/var.py`
- **Benchmark Target**: R `vars::VAR` / Python `statsmodels.tsa.vector_ar.var_model.VAR`.

### 2.2 Cointegrated Vector Error Correction Model (`VECMModel`) — **Implemented**

- **Description**: Multivariate model capturing long-run equilibrium relationships (cointegration) alongside short-run dynamics.
- **Specification**: $\Delta Y_t = \Pi Y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta Y_{t-i} + C D_t + u_t$, where $\Pi = \alpha \beta'$.
- **Cointegration Testing**: Johansen Trace and Maximum Eigenvalue rank tests.
- **Module Location**: `src/stats_transformer/models/timeseries/reduced_form/vecm.py`
- **Benchmark Target**: R `vars::ca.jo` and `vars::vec2var`.

### 2.3 Restricted VAR (`RestrictedVAR`) — **Implemented**

- **Description**: Reduced-form VAR enforced with zero coefficient restrictions at the equation level.
- **Specification**: Mask matrix $M \in \{0, 1\}^{K \times (K \cdot p + m)}$ zeroing specific lag or deterministic coefficients.
- **Module Location**: `src/stats_transformer/models/timeseries/reduced_form/restrictions.py`
- **Benchmark Target**: R `vars::restrict` / Kilian & Lütkepohl (2017) Ch. 3.

### 2.4 Lag Selection Engine (`VARLagSelector`) — **Implemented**

- **Description**: Automatic lag length selection across a user-defined range $[1, p_{\max}]$.
- **Information Criteria**: Akaike ($AIC$), Hannan-Quinn ($HQ$), Schwarz/Bayesian ($SC/BIC$), and Final Prediction Error ($FPE$).
- **Module Location**: `src/stats_transformer/models/timeseries/reduced_form/lag_selection.py`
- **Benchmark Target**: R `vars::VARselect`.

### 2.5 Diagnostic Suites — **Implemented**

- **Residual Autocorrelation**: Portmanteau and adjusted Portmanteau $Q$-tests.
- **Multivariate Normality**: Jarque-Bera, skewness, and kurtosis tests.
- **ARCH Heteroskedasticity**: Multivariate ARCH-LM test.
- **Stationarity & Roots**: Modulus of eigenvalues of the companion matrix $A_c$.
- **Module Location**: `src/stats_transformer/models/timeseries/diagnostics/`

### 2.6 Conjugate Bayesian VAR (`BVARModel`) — **Implemented**

- **Description**: VAR under a natural-conjugate Normal-Inverse-Wishart prior with Minnesota prior moments, estimated analytically rather than by MCMC.
- **Specification**: $B | \Sigma \sim \mathcal{MN}(\bar B, \Sigma \otimes \bar V)$, $\Sigma \sim \mathcal{IW}(\bar S, \bar\nu)$, with closed-form posterior moments $(\bar B, \bar V, \bar S, \bar\nu)$ updated from the Minnesota prior and the data.
- **Minnesota Prior**: `lambda1` (overall tightness), `lambda2` (cross-variable tightness), `lambda3` (lag decay), `lambda4` (constant term looseness); own-lag-1 prior mean is a random walk.
- **Inference**: Posterior draws of $(\Sigma, B)$ for IRF credible bands; no Gibbs sampler required since the joint posterior is available in closed form.
- **Module Location**: `src/stats_transformer/models/timeseries/reduced_form/bvar.py`
- **Benchmark Target**: R `BVAR::bvar` / Kadiyala & Karlsson (1997) natural-conjugate prior.

---

## 3. Tier 2: Structural Identification & SVEC

Structural VAR models isolate structural shocks $\epsilon_t$ from reduced-form errors $u_t$ using linear, inequality, or narrative constraints ($u_t = B \epsilon_t$ or $A u_t = B \epsilon_t$).

### 3.1 Short-Run & Long-Run SVAR (`SVARModel`, `BlanchardQuahModel`) — **Implemented**

- **Short-Run Restrictions**: Cholesky triangular decomposition or explicit structural constraints on matrix $A$ and $B$ such that $A \Sigma_u A' = B B'$.
- **Long-Run Restrictions**: Restrictions placed on the long-run cumulative impact matrix $C(1) = A(1)^{-1} B$ (Blanchard & Quah 1989).
- **Module Location**: `src/stats_transformer/models/timeseries/identification/svar.py`, `blanchard_quah.py`
- **Benchmark Target**: R `vars::SVAR` / Blanchard & Quah (1989).

### 3.2 Structural VECM (`SVEC`) — **Implemented**

- **Description**: Combining cointegration rank restrictions $\beta$ with structural short-run and long-run restrictions.
- **Identification**: Separates permanent shocks (matching cointegration rank $r$) from transitory shocks.
- **Module Location**: `src/stats_transformer/models/timeseries/structural/svec.py`
- **Benchmark Target**: R `vars::SVEC` / King, Plosser, Stock, and Watson (1991).

### 3.3 Sign & Zero Restrictions (`SignZeroSVARModel`) — **Implemented**

- **Description**: Non-Bayesian set identification imposing directional inequality signs ($+ / -$) and exact zeros ($0$) on impulse response paths.
- **Algorithm**: QR decomposition of random orthonormal matrices $Q$ (Rubio-Ramírez, Waggoner, & Zha 2010).
- **Inference**: Draw acceptance loop generating structural shock candidates; median-target selection and quantile bounds.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/sign_zero.py`
- **Benchmark Target**: VAR-Toolbox `Uhlig2005` / Rubio-Ramírez et al. (2010).

### 3.4 Narrative Restrictions (part of `SignZeroSVARModel`) — **Implemented**

- **Description**: Constraining the sign or magnitude of structural shocks and historical contributions during specific historical dates (e.g. 1973 Oil Crisis, 2008 Financial Crisis).
- **API**: no separate class. Pass `narrative_restrictions` in the `SignZeroSVARModel` restriction config; they are evaluated by `_check_narrative` inside the draw-acceptance loop.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/sign_zero.py`
- **Benchmark Target**: Antolín-Díaz & Rubio-Ramírez (2018).

### 3.5 Structural Bootstrap Engines (`SVARBootstrap`) — **Implemented**

- **Resampling Methods**: Residual bootstrap, Wild bootstrap (Rademacher / Normal weights), and Moving-Block bootstrap for time-series dependence.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/bootstrap.py`

---

## 4. Tier 3: Data-Driven SVAR Identification

Data-driven structural identification methods leverage statistical properties of reduced-form errors (heteroskedasticity, independence, non-Gaussianity) to identify structural impact matrices without requiring explicit theoretical restrictions.

### 4.1 Changes in Volatility (`VolatilitySVARModel`) — **Implemented**

- **Description**: Identification via discrete volatility regime shifts across known break dates.
- **Specification**: Reduced-form error covariance matrix shifts across regimes: $\Sigma_u^{(1)} = B B'$ and $\Sigma_u^{(2)} = B \Lambda B'$, where $\Lambda$ is diagonal.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/volatility.py`
- **Benchmark Target**: Rigobon (2003) / R `svars::id.cv`.

### 4.2 Distance Covariance & ICA (`IndependenceSVARModel`) — **Implemented**

- **Description**: Identification by maximizing structural shock independence via Distance Covariance minimization or FastICA.
- **Objective**: Minimize pairwise distance covariance between structural shock components: $\sum_{i < j} \mathcal{V}^2(\epsilon_i, \epsilon_j)$.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/independence.py`
- **Benchmark Target**: Matteson & Tsay (2017) / R `svars::id.dc`.

### 4.3 Cramér-von Mises Distance (`CVMSVARModel`) — **Implemented**

- **Description**: Identification by testing mutual independence using the Cramér-von Mises distance metric.
- **Optimization**: Multi-start optimization over orthogonal rotation angles.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/cvm.py`
- **Benchmark Target**: R `svars::id.cvm`.

### 4.4 Non-Gaussian Maximum Likelihood (`NonGaussianSVARModel`) — **Implemented**

- **Description**: Identification assuming structural shocks follow non-Gaussian distributions (Student-t).
- **Estimation**: Direct Maximum Likelihood estimation over unconstrained mixing matrices and degrees of freedom parameters.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/non_gaussian.py`
- **Benchmark Target**: Lanne, Meitz, & Saikkonen (2010) / R `svars::id.ng`.

### 4.5 Permutation & Sign Alignment (module functions) — **Implemented** (partial)

- **Description**: Post-processing resolving column ordering and sign ambiguities across bootstrap iterations.
- **API**: module-level functions, not a class: `align_signs`, `align_permutation_to_target`, `align_to_cholesky`.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/alignment.py`
- **Status**: sign and reference-matrix alignment are implemented. Hungarian-algorithm permutation matching (`scipy.optimize.linear_sum_assignment`) is *Planned*; there is no `AlignmentEngine` class.

---

## 5. Tier 4: Non-Linear & Regime-Switching Models

Nonlinear multivariate models relax the assumption of linear, state-invariant dynamics to capture asymmetric responses, financial stress regimes, or business cycle threshold shifts.

### 5.1 Threshold VAR (`TVARModel`) — **Implemented**

- **Description**: Two-regime Threshold VAR model where system dynamics switch based on an observed threshold variable $y_{t-d}$ relative to threshold value $\gamma$.
- **Specification**: 
  $$Y_t = (A^{(1)}_0 + \sum_{i=1}^p A^{(1)}_i Y_{t-i}) I(y_{t-d} \le \gamma) + (A^{(2)}_0 + \sum_{i=1}^p A^{(2)}_i Y_{t-i}) I(y_{t-d} > \gamma) + u_t$$
- **Estimation**: Grid search over threshold candidate values $\gamma$ and delay parameters $d$.
- **Module Location**: `src/stats_transformer/models/timeseries/nonlinear/tvar.py`
- **Benchmark Target**: R `tsDyn::TVAR`.

### 5.2 Threshold VECM (`TVECMModel`) — **Implemented**

- **Description**: Combining threshold regime-switching behavior with long-run cointegrated Vector Error Correction dynamics.
- **Specification**: Threshold search applied to error correction term $z_{t-1} = \beta' Y_{t-1}$.
- **Module Location**: `src/stats_transformer/models/timeseries/nonlinear/tvecm.py`
- **Benchmark Target**: R `tsDyn::TVECM`.

### 5.3 Smooth Transition VAR (`STVARModel`) — **Implemented**

- **Description**: Continuous regime-switching model with smooth transition weights defined by a logistic or exponential function $G(y_{t-d}; \gamma, c)$.
- **Transition Function**: $G(y_{t-d}; \gamma, c) = (1 + \exp(-\gamma (y_{t-d} - c)))^{-1}$.
- **Module Location**: `src/stats_transformer/models/timeseries/nonlinear/stvar.py`
- **Benchmark Target**: R `sstvars::STVAR`.

### 5.4 Generalized Impulse Response Functions (`GIRFEngine`) — **Implemented**

- **Description**: History-dependent, state-conditional impulse responses for non-linear models where responses depend on shock sign, magnitude, and initial state.
- **Simulation**: Monte Carlo history-based simulation comparing shocked trajectories against baseline trajectories (Koop, Pesaran, & Potter 1996).
- **Module Location**: `src/stats_transformer/models/timeseries/nonlinear/girf.py`
- **Benchmark Target**: R `sstvars::GIRF`.

---

## 6. Counterfactuals & Diagnostic Suite

### 6.1 Historical Decompositions (`TimeSeriesDecompositions`) — **Implemented**

- **Description**: Decomposing historical series trajectories into cumulative structural shock contributions: $Y_t = \sum_{j=0}^{t-1} \Theta_j \epsilon_{t-j} + \text{Initial Conditions}$.
- **Output**: Long DataFrame suitable for stacked bar and faceted visualization.
- **Module Location**: `src/stats_transformer/models/timeseries/structural/decompositions.py`

### 6.2 Counterfactual Conditional Forecasting — *Planned*

- **Description**: Simulating system trajectories under hypothetical shock scenarios (e.g. setting specific structural shocks to zero or enforcing target path constraints).
- **Benchmark Target**: Waggoner & Zha (1999) / Bańbura et al. (2015).

---

## 7. Deferred Scope & Architecture Policy

### 7.1 Explicitly Deferred Scope

- **Bayesian MCMC & Hierarchical Estimation**: Gibbs sampling, Metropolis-Hastings, hierarchical/non-conjugate priors, and general Bayesian state-space estimation are explicitly deferred to preserve a lightweight codebase without a sampler dependency. Analytical conjugate BVAR (see §2.6) is in scope, since its Normal-Inverse-Wishart posterior is closed-form and requires no MCMC.
- **High-Dimensional Regularization**: Penalized VARs (LASSO, Ridge) are deferred to a future high-dimensional proposal.

### 7.2 Technical Backlog & Optimizations

- Fast $O(T \log T)$ distance covariance approximation (Huo & Székely 2016).
- Vectorized candidate grid search for TVAR threshold variable selection.
- Multi-threaded parallel Monte Carlo simulation engine for GIRF computations.

---

## 8. Missing Models Catalog & Implementation Triage

This section catalogs models present in external macroeconometric toolboxes ([`cacoleman16/tsecon`](https://github.com/cacoleman16/tsecon) and [`ambropo/VAR-Toolbox`](https://github.com/ambropo/VAR-Toolbox)) that are currently absent from `stats-transformer`. Models are triaged in ascending order of algorithmic complexity and dependency requirements.

```
  [ Tier 1: Low Complexity ] ────► Closed-Form / OLS / Basic Matrix Algebra
               │
               ▼
  [ Tier 2: Medium Complexity ] ──► Standard Solvers / Iterative 2-Step
               │
               ▼
  [ Tier 3: Medium-High ] ────────► Univariate MLE / GARCH / Structural Breaks
               │
               ▼
  [ Tier 4: High Complexity ] ────► Multivariate Dynamic Covariances / High-Dimensional
```

| Triage Tier | Complexity | Representative Methods | Key Mathematical / Algorithmic Requirements |
| --- | --- | --- | --- |
| **Tier 1: Low Complexity** | Low | VARX, LP Multiplier, Stambaugh Regression, Diebold-Yilmaz Connectedness, Nelson-Siegel, Fry-Pagan Selection, Survey Diagnostics | Closed-form OLS, 2SLS, QR/SVD linear algebra, analytic bias formulas |
| **Tier 2: Medium Complexity** | Medium | Max-Share SVAR, Proxy SVAR + Sign, State-Dependent LP, Smooth LP, HAR-RV, Quantile LP, Linear DSGE Solver, Panel Time Series (MG/PMG/CCE-MG) | Eigenvalue optimization, B-spline penalty bases, linear programming / IRLS, generalized Schur (QZ) |
| **Tier 3: Medium-High Complexity** | Medium / High | GARCH / EGARCH / GJR, GAS / DCS Volatility, Bai-Perron Breaks, Markov-Switching AR, Conditional Forecasting, IVX Test, Dynamic Probit/Logit | Constrained non-linear QMLE (BHHH/L-BFGS-B), dynamic programming break search, Hamilton filter recursion |
| **Tier 4: High Complexity** | High | CCC / DCC-GARCH, FAVAR, Ragged-Edge DFM Nowcasting + News, Growth-at-Risk, Prior-Robust SVAR Bounds, Functional LP (FLP / FVAR) | Multi-stage dynamic covariance MLE, EM missing-data Kalman smoothing, continuous FPCA, robust set optimization |

---

### 8.1 Tier 1: Low Complexity (Closed-Form / OLS / Basic Matrix Algebra)

#### 8.1.1 VARX & Exogenous Regressors
- **Origin**: `VAR-Toolbox` / `tsecon`
- **Description**: Vector Autoregressions augmented with exogenous deterministic and stochastic drivers $X_t$:
  $$Y_t = \sum_{i=1}^p A_i Y_{t-i} + B X_t + u_t$$
- **Implementation Path**: Extend `VARModel` regressor matrix assembly to append $X_t$ columns and track dynamic multiplier projections.
- **Target Benchmark**: `ambropo/VAR-Toolbox` `VARmodel.m` / R `vars::VAR(..., exogen=)`.

#### 8.1.2 LP Multiplier (`lp_multiplier`)
- **Origin**: `tsecon`
- **Description**: Cumulative and integral multiplier estimation using local projections:
  $$\sum_{j=0}^h Y_{t+j} = \beta_h \left(\sum_{j=0}^h X_{t+j}\right) + \Gamma_h W_t + \epsilon_{t+h}$$
  estimated via 2SLS instrumented by narrative shocks $Z_t$.
- **Implementation Path**: Direct calculation in `LocalProjectionsIVModel` with cumulative outcome and treatment sums.
- **Target Benchmark**: Ramey & Zubairy (2018) / `tsecon.lp_multiplier`.

#### 8.1.3 Stambaugh Bias-Corrected Predictive Regression
- **Origin**: `tsecon`
- **Description**: Analytical bias correction for predictive regressions with persistent regressors:
  $$r_{t+1} = \alpha + \beta x_t + u_{t+1}, \quad x_{t+1} = \mu + \rho x_t + v_{t+1}$$
  $$\mathbb{E}[\hat\beta - \beta] = \frac{\sigma_{uv}}{\sigma_v^2} \mathbb{E}[\hat\rho - \rho] = -\frac{\sigma_{uv}}{\sigma_v^2} \left(\frac{1 + 3\rho}{T}\right)$$
- **Implementation Path**: Closed-form coefficient adjustment following Stambaugh (1999).
- **Target Benchmark**: `tsecon.predictive_regression`.

#### 8.1.4 Diebold-Yilmaz Connectedness Index (`connectedness`)
- **Origin**: `tsecon`
- **Description**: Total, directional, and net spillover connectedness indices derived from generalized forecast error variance decompositions (GFEVD):
  $$S(H) = \frac{\sum_{i \ne j} \tilde\theta_{ij}(H)}{\sum_{i,j} \tilde\theta_{ij}(H)} \times 100$$
- **Implementation Path**: Post-processing function taking `VARModel` GFEVD matrices as input.
- **Target Benchmark**: Diebold & Yilmaz (2012) / `tsecon.connectedness`.

#### 8.1.5 Nelson-Siegel & Svensson Yield Curve Models
- **Origin**: `tsecon`
- **Description**: Parametric yield curve fitting based on level, slope, and curvature factor loadings:
  $$y(\tau) = \beta_0 + \beta_1 \left(\frac{1 - e^{-\lambda \tau}}{\lambda \tau}\right) + \beta_2 \left(\frac{1 - e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau}\right)$$
- **Implementation Path**: Closed-form OLS given fixed decay parameter $\lambda$ or simple 1D grid search.
- **Target Benchmark**: Nelson & Siegel (1987) / `tsecon.nelson_siegel`.

#### 8.1.6 Fry-Pagan Median Target Selection
- **Origin**: `VAR-Toolbox` / `tsecon`
- **Description**: Algorithm selecting the single structural identification draw $Q^*$ whose IRF profile minimizes the quadratic distance to the pointwise median IRF:
  $$Q^* = \arg\min_k \sum_{i,j,h} \left( \frac{\text{IRF}_{i,j,h}^{(k)} - \text{median}_{i,j,h}}{\text{std}_{i,j,h}} \right)^2$$
- **Implementation Path**: Post-draw selection helper inside `SignZeroSVARModel`.
- **Target Benchmark**: Fry & Pagan (2011) / `ambropo/VAR-Toolbox` `VARsign.m`.

#### 8.1.7 Survey Expectations Diagnostics
- **Origin**: `tsecon`
- **Description**: Diagnostic regression tests for forecast rationality and informational frictions:
  - Coibion-Gorodnichenko: $x_{t+h} - F_t x_{t+h} = \alpha + \beta (F_t x_{t+h} - F_{t-1} x_{t+h}) + \epsilon_{t+h}$
  - Mincer-Zarnowitz: $x_{t+h} = \alpha + \beta F_t x_{t+h} + \epsilon_{t+h}$
- **Implementation Path**: Simple OLS regression with Newey-West standard errors.
- **Target Benchmark**: Coibion & Gorodnichenko (2015) / `tsecon.cg_regression`.

---

### 8.2 Tier 2: Medium Complexity (Standard Solvers / Iterative 2-Step)

#### 8.2.1 Max-Share / Max-FEV SVAR
- **Origin**: `tsecon`
- **Description**: Identification of news and technology shocks by finding an impact rotation column $q$ that maximizes the forecast error variance contribution of variable $i$ at horizon $H$:
  $$\max_{q} q' V(H) q \quad \text{s.t.} \quad q' q = 1$$
- **Implementation Path**: Eigenvalue decomposition of the target horizon FEVD matrix.
- **Target Benchmark**: Barsky & Sims (2011) / Uhlig (2004) / `tsecon.max_share_svar`.

#### 8.2.2 Proxy SVAR Combined with Sign Restrictions
- **Origin**: `VAR-Toolbox`
- **Description**: Joint structural identification where an external instrument pins down the first column of the impact matrix, and remaining columns are restricted via sign inequalities.
- **Implementation Path**: Combine `ProxySVARModel` first-stage instrument regression with QR sign-rotation acceptance loop on the orthogonal subspace.
- **Target Benchmark**: `ambropo/VAR-Toolbox` `VARsign_proxy.m`.

#### 8.2.3 State-Dependent Local Projections (`lp_state`)
- **Origin**: `tsecon`
- **Description**: Local projections with dynamics interacted with continuous or binary regime indicator $F(z_t)$:
  $$Y_{t+h} = F(z_t) (\alpha_h^{(1)} + \beta_h^{(1)} X_t) + (1 - F(z_t)) (\alpha_h^{(2)} + \beta_h^{(2)} X_t) + \Gamma_h W_t + \epsilon_{t+h}$$
- **Implementation Path**: Extend `LocalProjectionsModel` with interaction terms and regime-specific Newey-West HAC inference.
- **Target Benchmark**: Ramey & Zubairy (2018) / `tsecon.lp_state`.

#### 8.2.4 Smooth Local Projections (`smooth_lp`)
- **Origin**: `tsecon`
- **Description**: Regularized local projections estimating impulse responses across all horizons simultaneously via penalized B-spline basis expansion:
  $$\min_{\theta} \sum_{h=0}^H \| Y_{t+h} - B_h \theta X_t \|^2 + \lambda \theta' D' D \theta$$
- **Implementation Path**: Ridge-regularized generalized least squares with cross-validated smoothing penalty $\lambda$.
- **Target Benchmark**: Barnichon & Brownlees (2019) / `tsecon.smooth_lp`.

#### 8.2.5 Heterogeneous Autoregressive Realized Volatility (HAR-RV)
- **Origin**: `tsecon`
- **Description**: Multi-component volatility model aggregating daily, weekly, and monthly realized variance:
  $$RV_{t+1}^{(d)} = \beta_0 + \beta_d RV_t^{(d)} + \beta_w RV_t^{(w)} + \beta_m RV_t^{(m)} + \epsilon_{t+1}$$
- **Implementation Path**: Rolling multi-frequency feature extraction followed by robust OLS.
- **Target Benchmark**: Corsi (2009) / `tsecon.har_rv`.

#### 8.2.6 Quantile Local Projections (`quantile_lp`)
- **Origin**: `tsecon`
- **Description**: Estimating horizon-dependent conditional quantile impulse responses:
  $$Q_{\tau}(Y_{t+h} | X_t, W_t) = \alpha_h(\tau) + \beta_h(\tau) X_t + \Gamma_h(\tau) W_t$$
- **Implementation Path**: Iteratively reweighted least squares (IRLS) Koenker-Bassett quantile solver across projection horizons.
- **Target Benchmark**: `tsecon.quantile_lp`.

#### 8.2.7 Linear Rational Expectations DSGE-Lite Solver (`dsge_solve`)
- **Origin**: `tsecon`
- **Description**: Solving linear first-order rational expectations model systems:
  $$A \mathbb{E}_t [y_{t+1}] = B y_t + C x_t$$
- **Implementation Path**: Generalized Schur / QZ matrix decomposition implementing the Blanchard-Kahn saddle-path stability condition.
- **Target Benchmark**: Blanchard & Kahn (1980) / Klein (2000) / `tsecon.dsge_solve`.

#### 8.2.8 Panel Time Series (MG, PMG, CCE-MG)
- **Origin**: `tsecon`
- **Description**: Heterogeneous and cross-sectionally dependent panel estimators:
  - Mean Group (MG): unweighted average of entity-specific time-series coefficients.
  - Pooled Mean Group (PMG): common long-run cointegrating vectors with entity-specific short-run adjustment.
  - Common Correlated Effects (CCE-MG): cross-sectional averages of dependent and explanatory variables proxying for unobserved common factors.
- **Implementation Path**: Multi-equation loop over entities with Pesaran (2006) cross-sectional augmentation.
- **Target Benchmark**: Pesaran & Smith (1995) / Pesaran (2006) / `tsecon.panel_mean_group`.

---

### 8.3 Tier 3: Medium-High Complexity (Numerical MLE / Nonlinear Optimization)

#### 8.3.1 Univariate GARCH Suite (GARCH, GJR-GARCH, EGARCH)
- **Origin**: `tsecon`
- **Description**: Autoregressive conditional heteroskedasticity models capturing volatility clustering and leverage asymmetries:
  $$\sigma_t^2 = \omega + \sum_{i=1}^q (\alpha_i + \gamma_i I(\epsilon_{t-i} < 0)) \epsilon_{t-i}^2 + \sum_{j=1}^p \beta_j \sigma_{t-j}^2$$
- **Implementation Path**: Constrained quasi-maximum likelihood estimation (QMLE) with analytical or numerical gradients and Bollerslev-Wooldridge robust standard errors.
- **Target Benchmark**: Engle (1982) / Bollerslev (1986) / Glosten-Jagannathan-Runkle (1993) / Nelson (1991) / Python `arch` / `tsecon.garch`.

#### 8.3.2 Generalized Autoregressive Score (GAS / DCS) Volatility
- **Origin**: `tsecon`
- **Description**: Observation-driven time-varying parameter model driven by the score of the conditional distribution:
  $$f_{t+1} = \omega + \sum_{i=1}^p A_i s_t + \sum_{j=1}^q B_j f_{t-j}, \quad s_t = S_t \nabla_t$$
- **Implementation Path**: Numerical MLE with Gaussian and Student-t conditional distributions.
- **Target Benchmark**: Creal, Koopman, & Lucas (2013) / Harvey (2013) / `tsecon.gas_volatility`.

#### 8.3.3 Bai-Perron Multiple Structural Breaks
- **Origin**: `tsecon`
- **Description**: Dynamic programming algorithm estimating $m$ unknown break dates in linear regression systems:
  $$\min_{(T_1, \dots, T_m)} \sum_{k=1}^{m+1} \sum_{t=T_{k-1}+1}^{T_k} (y_t - x_t' \beta_k)^2$$
- **Implementation Path**: Dynamic programming triangular sum-of-squares matrix search with Sup-F tests and BIC break selection.
- **Target Benchmark**: Bai & Perron (1998, 2003) / `tsecon.bai_perron`.

#### 8.3.4 Markov-Switching Autoregression (MS-AR)
- **Origin**: `tsecon`
- **Description**: Autoregressive system with discrete unobserved Markov state regimes $S_t \in \{1, \dots, M\}$:
  $$y_t = \mu(S_t) + \sum_{i=1}^p \phi_i(S_t) y_{t-i} + \sigma(S_t) \epsilon_t$$
- **Implementation Path**: Hamilton filter recursion with numerical log-likelihood optimization and smoothed regime probabilities.
- **Target Benchmark**: Hamilton (1989) / `statsmodels.tsa.regime_switching` / `tsecon.markov_switching_ar`.

#### 8.3.5 Conditional Forecasting & Scenario Projections
- **Origin**: `VAR-Toolbox`
- **Description**: Generating system forecasts conditional on specified future shock trajectories or fixed endogenous variable paths:
  $$\mathbb{E}[Y_{T+1:T+H} | Y_{1:T}, \mathcal{R} \epsilon_{T+1:T+H} = r]$$
- **Implementation Path**: Waggoner & Zha (1999) restricted least squares shock inversion.
- **Target Benchmark**: `ambropo/VAR-Toolbox` `VARcondfore.m`.

#### 8.3.6 Persistence-Robust IVX Estimation & Testing
- **Origin**: `tsecon`
- **Description**: Robust inference in predictive regressions with near-unit-root or persistent predictors via mildly integrated instrument construction:
  $$\tilde z_t = \sum_{j=1}^t (1 - c_z / T^\delta)^{t-j} \Delta x_j$$
- **Implementation Path**: IVX filter generation followed by instrumental variables estimation and chi-squared Wald testing.
- **Target Benchmark**: Kostakis, Magdalinos, & Stamatogiannis (2015) / `tsecon.ivx_test`.

#### 8.3.7 Dynamic Probit & Logit Models
- **Origin**: `tsecon`
- **Description**: Autoregressive binary classification models for recession forecasting:
  $$P(y_t = 1 | \Omega_{t-1}) = \Phi(\pi_t), \quad \pi_t = \omega + \alpha \pi_{t-1} + \beta' x_{t-1} + \delta y_{t-1}$$
- **Implementation Path**: Numerical maximum likelihood estimation under log-concave binary densities.
- **Target Benchmark**: Kauppi & Saikkonen (2008) / `tsecon.recession_probit`.

---

### 8.4 Tier 4: High Complexity (Multivariate Dynamic Covariances / High-Dimensional / Robust Bounds)

#### 8.4.1 Multivariate GARCH (CCC / DCC-GARCH)
- **Origin**: `tsecon`
- **Description**: Dynamic conditional correlation model decomposing covariance into time-varying standard deviations and correlations:
  $$\Sigma_t = D_t R_t D_t, \quad Q_t = (1 - \alpha - \beta) \bar Q + \alpha \epsilon_{t-1}^* {\epsilon_{t-1}^*}' + \beta Q_{t-1}$$
- **Implementation Path**: Two-stage quasi-maximum likelihood optimization with positive definiteness projection constraints.
- **Target Benchmark**: Bollerslev (1990) / Engle (2002) / `tsecon.dcc_garch`.

#### 8.4.2 Factor-Augmented VAR (FAVAR)
- **Origin**: `tsecon`
- **Description**: VAR augmenting observed macroeconomic variables $Y_t$ with unobserved latent factors $F_t$ extracted from a large data panel $X_t$:
  $$\begin{pmatrix} F_t \\ Y_t \end{pmatrix} = \Phi(L) \begin{pmatrix} F_{t-1} \\ Y_{t-1} \end{pmatrix} + v_t, \quad X_t = \Lambda^f F_t + \Lambda^y Y_t + e_t$$
- **Implementation Path**: Two-step principal components extraction with factor rotation and structural impulse response mapping.
- **Target Benchmark**: Bernanke, Boivin, & Eliasz (2005) / `tsecon.favar`.

#### 8.4.3 Ragged-Edge DFM Nowcasting with News Decomposition
- **Origin**: `tsecon`
- **Description**: Maximum likelihood dynamic factor model supporting unbalanced panel ragged edges and news decomposition:
  $$\mathcal{N}_{t} = \mathbb{E}[y_{t+h} | \Omega_{\text{new}}] - \mathbb{E}[y_{t+h} | \Omega_{\text{old}}] = \sum_i \beta_i (v_{i,t} - \mathbb{E}[v_{i,t} | \Omega_{\text{old}}])$$
- **Implementation Path**: Expectation-Maximization (EM) algorithm with missing-data Kalman smoother and Bańbura-Modugno shock attribution.
- **Target Benchmark**: Doz, Giannone, & Reichlin (2011) / Bańbura & Modugno (2014) / `tsecon.dfm_nowcast` / `tsecon.dfm_news`.

#### 8.4.4 Growth-at-Risk (GaR) Modeling
- **Origin**: `tsecon`
- **Description**: Macroeconomic downside risk evaluation by estimating conditional quantiles, fitting skewed-t parametric densities, and computing tail loss probabilities.
- **Implementation Path**: Koenker-Bassett quantile regressions, Azzalini skew-t density parameter inversion, and monotone quantile rearrangement.
- **Target Benchmark**: Adrian, Boyarchenko, & Giannone (2019) / `tsecon.growth_at_risk`.

#### 8.4.5 Giacomini-Kitagawa Prior-Robust SVAR Bounds
- **Origin**: `tsecon`
- **Description**: Set-identified structural VAR inference providing prior-robust bounds without requiring uniform Haar prior imposition over orthogonal rotation matrices $Q$.
- **Implementation Path**: Numerical global optimization minimizing and maximizing structural target responses over the identified set $\mathcal{Q}(B)$.
- **Target Benchmark**: Giacomini & Kitagawa (2021) / `tsecon.robust_svar_bounds`.

#### 8.4.6 Functional Local Projections & FVAR (`flp` / `fvar_scenario`)
- **Origin**: `tsecon`
- **Description**: Impulse response analysis for continuous curve-valued outcomes (such as high-frequency yield curves):
  $$Y_{t+h}(u) = \alpha_h(u) + \beta_h(u) X_t + \epsilon_{t+h}(u), \quad u \in [0, 1]$$
- **Implementation Path**: Functional Principal Component Analysis (FPCA) dimensionality reduction followed by multi-horizon curve projection.
- **Target Benchmark**: Inoue & Rossi (2021) / `tsecon.flp`.

