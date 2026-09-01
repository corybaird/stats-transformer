# Future Model Extension Roadmap

This roadmap catalogs future model extensions, advanced structural identification schemes, and econometric methods planned for upcoming releases of `stats-transformer`. 

For the complete catalog of models already implemented and available in the library, refer to the [Implemented Models Catalog](library/models.md).

---

## 1. Roadmap Architecture & Complexity Tiers

Planned models are triaged across four implementation tiers based on mathematical complexity, optimization routines, and algorithmic dependencies:

```text
  [ Tier 1: Low Complexity ] ────► Closed-Form / OLS / Basic Matrix Algebra
               │
               ▼
  [ Tier 2: Medium Complexity ] ──► Standard Solvers / Iterative 2-Step
               │
               ▼
  [ Tier 3: Medium-High ] ────────► Numerical MLE / GARCH / Structural Breaks
               │
               ▼
  [ Tier 4: High Complexity ] ────► Dynamic Covariances / High-Dimensional / Robust Bounds
```

| Triage Tier | Algorithmic Complexity | Representative Planned Methods | Primary Computational Requirements | Target Benchmark Software |
| --- | --- | --- | --- | --- |
| **Tier 1: Low Complexity** | Low | VARX, LP Multiplier, Stambaugh Regression, Diebold-Yilmaz Connectedness, Nelson-Siegel, Fry-Pagan Selection, Survey Diagnostics | Closed-form OLS, 2SLS, QR/SVD linear algebra, analytic bias formulas | `ambropo/VAR-Toolbox`, `cacoleman16/tsecon`, R `vars` |
| **Tier 2: Medium Complexity** | Medium | Max-Share SVAR, Proxy SVAR + Sign, State-Dependent LP, Smooth LP, HAR-RV, Quantile LP, Linear DSGE Solver, Panel Time Series (MG/PMG/CCE) | Generalized eigenvalue optimization, penalized B-splines, IRLS quantile solvers, QZ decomposition | `ambropo/VAR-Toolbox`, `cacoleman16/tsecon`, R `lpirfs` |
| **Tier 3: Medium-High Complexity** | Medium / High | GARCH / EGARCH / GJR-GARCH, GAS / DCS Volatility, Bai-Perron Breaks, Markov-Switching AR, Conditional Forecasting, IVX Test, Dynamic Probit/Logit | Constrained non-linear QMLE (BHHH/L-BFGS-B), dynamic programming break search, Hamilton filter recursion | Python `arch`, `statsmodels`, `ambropo/VAR-Toolbox`, `cacoleman16/tsecon` |
| **Tier 4: High Complexity** | High | CCC / DCC-GARCH, FAVAR, Ragged-Edge DFM Nowcasting + News, Growth-at-Risk (GaR), Giacomini-Kitagawa Prior-Robust SVAR Bounds, Functional LP (FLP / FVAR) | Multi-stage dynamic covariance MLE, EM missing-data Kalman smoothing, continuous FPCA, robust set optimization | `cacoleman16/tsecon`, R `rmgarch` |

---

## 2. Tier 1: Low Complexity (Closed-Form / OLS / Basic Matrix Algebra)

### 2.1 VARX & Exogenous Regressors
- **Description**: Vector Autoregressions augmented with exogenous deterministic and stochastic drivers $X_t$:
  $$Y_t = \sum_{i=1}^p A_i Y_{t-i} + B X_t + u_t$$
- **Implementation Strategy**: Extend regressor matrix assembly in `VARModel` to append exogenous columns and calculate dynamic multipliers.
- **Target Benchmark**: `ambropo/VAR-Toolbox` (`VARmodel.m`), R (`vars::VAR(..., exogen=)`)

### 2.2 Local Projections Multipliers (`lp_multiplier`)
- **Description**: Cumulative and integral multiplier estimation using local projections:
  $$\sum_{j=0}^h Y_{t+j} = \beta_h \left(\sum_{j=0}^h X_{t+j}\right) + \Gamma_h W_t + \epsilon_{t+h}$$
  estimated via 2SLS instrumented by narrative shocks $Z_t$.
- **Implementation Strategy**: Add cumulative summation transformations inside `LocalProjectionsIVModel`.
- **Target Benchmark**: Ramey & Zubairy (2018), `cacoleman16/tsecon` (`lp_multiplier`)

### 2.3 Stambaugh Bias-Corrected Predictive Regression
- **Description**: Analytical bias correction for predictive regressions with persistent regressors:
  $$r_{t+1} = \alpha + \beta x_t + u_{t+1}, \quad x_{t+1} = \mu + \rho x_t + v_{t+1}$$
  $$\mathbb{E}[\hat\beta - \beta] = -\frac{\sigma_{uv}}{\sigma_v^2} \left(\frac{1 + 3\rho}{T}\right)$$
- **Implementation Strategy**: Analytical post-estimation coefficient adjustment in `RegressionModel`.
- **Target Benchmark**: Stambaugh (1999), `cacoleman16/tsecon` (`predictive_regression`)

### 2.4 Diebold-Yilmaz Connectedness Index (`connectedness`)
- **Description**: Total, directional, and net spillover connectedness indices derived from generalized forecast error variance decompositions (GFEVD):
  $$S(H) = \frac{\sum_{i \ne j} \tilde\theta_{ij}(H)}{\sum_{i,j} \tilde\theta_{ij}(H)} \times 100$$
- **Implementation Strategy**: Post-processing utility consuming `VARModel` GFEVD matrices.
- **Target Benchmark**: Diebold & Yilmaz (2012), `cacoleman16/tsecon` (`connectedness`)

### 2.5 Nelson-Siegel & Svensson Yield Curve Models
- **Description**: Parametric yield curve fitting based on level, slope, and curvature factor loadings:
  $$y(\tau) = \beta_0 + \beta_1 \left(\frac{1 - e^{-\lambda \tau}}{\lambda \tau}\right) + \beta_2 \left(\frac{1 - e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau}\right)$$
- **Implementation Strategy**: Closed-form OLS given fixed decay parameter $\lambda$ or 1D grid search.
- **Target Benchmark**: Nelson & Siegel (1987), `cacoleman16/tsecon` (`nelson_siegel`)

### 2.6 Fry-Pagan Median Target Selection
- **Description**: Selection routine finding the single structural identification draw $Q^*$ whose IRF profile minimizes quadratic distance to the pointwise median IRF:
  $$Q^* = \arg\min_k \sum_{i,j,h} \left( \frac{\text{IRF}_{i,j,h}^{(k)} - \text{median}_{i,j,h}}{\text{std}_{i,j,h}} \right)^2$$
- **Implementation Strategy**: Post-draw selection helper in `SignZeroSVARModel`.
- **Target Benchmark**: Fry & Pagan (2011), `ambropo/VAR-Toolbox` (`VARsign.m`)

### 2.7 Survey Expectations Diagnostics
- **Description**: Diagnostic regression tests for forecast rationality and informational frictions:
  - Coibion-Gorodnichenko: $x_{t+h} - F_t x_{t+h} = \alpha + \beta (F_t x_{t+h} - F_{t-1} x_{t+h}) + \epsilon_{t+h}$
  - Mincer-Zarnowitz: $x_{t+h} = \alpha + \beta F_t x_{t+h} + \epsilon_{t+h}$
- **Implementation Strategy**: Linear regression helper with Newey-West standard errors.
- **Target Benchmark**: Coibion & Gorodnichenko (2015), `cacoleman16/tsecon` (`cg_regression`)

---

## 3. Tier 2: Medium Complexity (Standard Solvers / Iterative 2-Step)

### 3.1 Max-Share / Max-FEV SVAR
- **Description**: Identification of news and technology shocks by finding an impact rotation column $q$ that maximizes forecast error variance contribution at horizon $H$:
  $$\max_{q} q' V(H) q \quad \text{s.t.} \quad q' q = 1$$
- **Implementation Strategy**: Generalized eigenvalue decomposition of the target horizon FEVD matrix.
- **Target Benchmark**: Barsky & Sims (2011), Uhlig (2004), `cacoleman16/tsecon` (`max_share_svar`)

### 3.2 Proxy SVAR Combined with Sign Restrictions
- **Description**: Joint structural identification where an external instrument pins down the first column of the impact matrix and remaining columns are restricted via sign inequalities.
- **Implementation Strategy**: Combine `ProxySVARModel` first-stage instrument regression with QR sign-rotation acceptance loop on the orthogonal subspace.
- **Target Benchmark**: `ambropo/VAR-Toolbox` (`VARsign_proxy.m`)

### 3.3 State-Dependent Local Projections (`lp_state`)
- **Description**: Local projections with dynamics interacted with continuous or binary regime indicator $F(z_t)$:
  $$Y_{t+h} = F(z_t) (\alpha_h^{(1)} + \beta_h^{(1)} X_t) + (1 - F(z_t)) (\alpha_h^{(2)} + \beta_h^{(2)} X_t) + \Gamma_h W_t + \epsilon_{t+h}$$
- **Implementation Strategy**: Extend `LocalProjectionsModel` with interaction terms and regime-specific HAC inference.
- **Target Benchmark**: Ramey & Zubairy (2018), `cacoleman16/tsecon` (`lp_state`)

### 3.4 Smooth Local Projections (`smooth_lp`)
- **Description**: Regularized local projections estimating impulse responses across all horizons simultaneously via penalized B-spline basis expansion:
  $$\min_{\theta} \sum_{h=0}^H \| Y_{t+h} - B_h \theta X_t \|^2 + \lambda \theta' D' D \theta$$
- **Implementation Strategy**: Ridge-regularized generalized least squares with cross-validated penalty $\lambda$.
- **Target Benchmark**: Barnichon & Brownlees (2019), `cacoleman16/tsecon` (`smooth_lp`)

### 3.5 Heterogeneous Autoregressive Realized Volatility (HAR-RV)
- **Description**: Multi-component volatility model aggregating daily, weekly, and monthly realized variance:
  $$RV_{t+1}^{(d)} = \beta_0 + \beta_d RV_t^{(d)} + \beta_w RV_t^{(w)} + \beta_m RV_t^{(m)} + \epsilon_{t+1}$$
- **Implementation Strategy**: Rolling multi-frequency feature extraction followed by robust OLS.
- **Target Benchmark**: Corsi (2009), `cacoleman16/tsecon` (`har_rv`)

### 3.6 Quantile Local Projections (`quantile_lp`)
- **Description**: Estimating horizon-dependent conditional quantile impulse responses:
  $$Q_{\tau}(Y_{t+h} | X_t, W_t) = \alpha_h(\tau) + \beta_h(\tau) X_t + \Gamma_h(\tau) W_t$$
- **Implementation Strategy**: Iteratively reweighted least squares (IRLS) quantile solver across projection horizons.
- **Target Benchmark**: `cacoleman16/tsecon` (`quantile_lp`)

### 3.7 Linear Rational Expectations DSGE Solver (`dsge_solve`)
- **Description**: Solving linear first-order rational expectations systems:
  $$A \mathbb{E}_t [y_{t+1}] = B y_t + C x_t$$
- **Implementation Strategy**: Generalized Schur / QZ matrix decomposition implementing Blanchard-Kahn saddle-path stability conditions.
- **Target Benchmark**: Blanchard & Kahn (1980), Klein (2000), `cacoleman16/tsecon` (`dsge_solve`)

### 3.8 Panel Time Series (MG, PMG, CCE-MG)
- **Description**: Heterogeneous and cross-sectionally dependent panel estimators:
  - Mean Group (MG): unweighted average of entity-specific coefficients.
  - Pooled Mean Group (PMG): common long-run cointegrating vectors with heterogeneous short-run adjustment.
  - Common Correlated Effects (CCE-MG): cross-sectional averages proxying for unobserved common factors.
- **Implementation Strategy**: Multi-entity estimation loop with Pesaran (2006) cross-sectional augmentation.
- **Target Benchmark**: Pesaran (2006), `cacoleman16/tsecon` (`panel_mean_group`)

---

## 4. Tier 3: Medium-High Complexity (Numerical MLE / Nonlinear Optimization)

### 4.1 Univariate GARCH Suite (GARCH, GJR-GARCH, EGARCH)
- **Description**: Autoregressive conditional heteroskedasticity models capturing volatility clustering and leverage asymmetries:
  $$\sigma_t^2 = \omega + \sum_{i=1}^q (\alpha_i + \gamma_i \mathbb{I}(\epsilon_{t-i} < 0)) \epsilon_{t-i}^2 + \sum_{j=1}^p \beta_j \sigma_{t-j}^2$$
- **Implementation Strategy**: Constrained quasi-maximum likelihood estimation (QMLE) with analytical gradients and Bollerslev-Wooldridge standard errors.
- **Target Benchmark**: Engle (1982), Bollerslev (1986), Python `arch`, `cacoleman16/tsecon` (`garch`)

### 4.2 Generalized Autoregressive Score (GAS / DCS) Volatility
- **Description**: Observation-driven time-varying parameter model driven by the score of the conditional distribution:
  $$f_{t+1} = \omega + \sum_{i=1}^p A_i s_t + \sum_{j=1}^q B_j f_{t-j}, \quad s_t = S_t \nabla_t$$
- **Implementation Strategy**: Numerical MLE under Gaussian and Student-$t$ conditional densities.
- **Target Benchmark**: Creal, Koopman, & Lucas (2013), `cacoleman16/tsecon` (`gas_volatility`)

### 4.3 Bai-Perron Multiple Structural Breaks
- **Description**: Dynamic programming algorithm estimating $m$ unknown break dates in linear regression systems:
  $$\min_{(T_1, \dots, T_m)} \sum_{k=1}^{m+1} \sum_{t=T_{k-1}+1}^{T_k} (y_t - x_t' \beta_k)^2$$
- **Implementation Strategy**: Triangular sum-of-squares matrix search with Sup-F tests and BIC selection.
- **Target Benchmark**: Bai & Perron (1998, 2003), `cacoleman16/tsecon` (`bai_perron`)

### 4.4 Markov-Switching Autoregression (MS-AR)
- **Description**: Autoregressive system with discrete unobserved Markov state regimes $S_t \in \{1, \dots, M\}$:
  $$y_t = \mu(S_t) + \sum_{i=1}^p \phi_i(S_t) y_{t-i} + \sigma(S_t) \epsilon_t$$
- **Implementation Strategy**: Hamilton filter recursion with numerical log-likelihood optimization and smoothed probabilities.
- **Target Benchmark**: Hamilton (1989), `statsmodels.tsa.regime_switching`, `cacoleman16/tsecon` (`markov_switching_ar`)

### 4.5 Counterfactual Conditional Forecasting & Scenario Projections
- **Description**: Generating system forecasts conditional on specified future shock trajectories or fixed endogenous variable paths:
  $$\mathbb{E}[Y_{T+1:T+H} | Y_{1:T}, \mathcal{R} \epsilon_{T+1:T+H} = r]$$
- **Implementation Strategy**: Waggoner & Zha (1999) restricted least squares shock inversion.
- **Target Benchmark**: `ambropo/VAR-Toolbox` (`VARcondfore.m`), Bańbura et al. (2015)

### 4.6 Persistence-Robust IVX Testing
- **Description**: Robust inference in predictive regressions with near-unit-root or persistent predictors via mildly integrated instrument construction.
- **Implementation Strategy**: IVX filter generation followed by instrumental variables estimation and Wald tests.
- **Target Benchmark**: Kostakis, Magdalinos, & Stamatogiannis (2015), `cacoleman16/tsecon` (`ivx_test`)

### 4.7 Dynamic Probit & Logit Models
- **Description**: Autoregressive binary classification models for recession forecasting with lagged probability feedback.
- **Implementation Strategy**: Numerical maximum likelihood estimation under log-concave binary densities.
- **Target Benchmark**: Kauppi & Saikkonen (2008), `cacoleman16/tsecon` (`recession_probit`)

---

## 5. Tier 4: High Complexity (Dynamic Covariances / High-Dimensional / Robust Bounds)

### 5.1 Multivariate GARCH (CCC / DCC-GARCH)
- **Description**: Dynamic conditional correlation model decomposing covariance into time-varying standard deviations and correlations:
  $$\Sigma_t = D_t R_t D_t, \quad Q_t = (1 - \alpha - \beta) \bar Q + \alpha \epsilon_{t-1}^* {\epsilon_{t-1}^*}' + \beta Q_{t-1}$$
- **Implementation Strategy**: Two-stage quasi-maximum likelihood optimization with positive definiteness projection constraints.
- **Target Benchmark**: Bollerslev (1990), Engle (2002), `cacoleman16/tsecon` (`dcc_garch`)

### 5.2 Factor-Augmented VAR (FAVAR)
- **Description**: VAR augmenting observed macroeconomic variables $Y_t$ with unobserved latent factors $F_t$ extracted from large data panels $X_t$.
- **Implementation Strategy**: Two-step principal components extraction with factor rotation and structural impulse response mapping.
- **Target Benchmark**: Bernanke, Boivin, & Eliasz (2005), `cacoleman16/tsecon` (`favar`)

### 5.3 Ragged-Edge DFM Nowcasting with News Decomposition
- **Description**: Maximum likelihood dynamic factor model supporting unbalanced panel ragged edges and news attribution decomposition:
  $$\mathcal{N}_{t} = \mathbb{E}[y_{t+h} | \Omega_{\text{new}}] - \mathbb{E}[y_{t+h} | \Omega_{\text{old}}] = \sum_i \beta_i (v_{i,t} - \mathbb{E}[v_{i,t} | \Omega_{\text{old}}])$$
- **Implementation Strategy**: Expectation-Maximization (EM) algorithm with missing-data Kalman smoother and Bańbura-Modugno shock attribution.
- **Target Benchmark**: Doz, Giannone, & Reichlin (2011), Bańbura & Modugno (2014), `cacoleman16/tsecon` (`dfm_nowcast`, `dfm_news`)

### 5.4 Growth-at-Risk (GaR) Modeling
- **Description**: Macroeconomic downside risk evaluation estimating conditional quantiles, fitting skewed-$t$ parametric densities, and computing tail loss probabilities.
- **Implementation Strategy**: Koenker-Bassett quantile regressions, Azzalini skew-$t$ density parameter inversion, and monotone quantile rearrangement.
- **Target Benchmark**: Adrian, Boyarchenko, & Giannone (2019), `cacoleman16/tsecon` (`growth_at_risk`)

### 5.5 Giacomini-Kitagawa Prior-Robust SVAR Bounds
- **Description**: Set-identified structural VAR inference providing prior-robust bounds without requiring uniform Haar prior imposition over orthogonal rotation matrices $Q$.
- **Implementation Strategy**: Numerical global optimization minimizing and maximizing structural target responses over the identified set $\mathcal{Q}(B)$.
- **Target Benchmark**: Giacomini & Kitagawa (2021), `cacoleman16/tsecon` (`robust_svar_bounds`)

### 5.6 Functional Local Projections & FVAR (`flp`)
- **Description**: Impulse response analysis for continuous curve-valued outcomes such as high-frequency yield curves:
  $$Y_{t+h}(u) = \alpha_h(u) + \beta_h(u) X_t + \epsilon_{t+h}(u), \quad u \in [0, 1]$$
- **Implementation Strategy**: Functional Principal Component Analysis (FPCA) dimensionality reduction followed by multi-horizon curve projection.
- **Target Benchmark**: Inoue & Rossi (2021), `cacoleman16/tsecon` (`flp`)

---

## 6. Deferred Scope & Architecture Boundaries

### 6.1 Explicitly Deferred Scope
- **General Bayesian MCMC Sampling**: Gibbs samplers, Metropolis-Hastings chains, and hierarchical state-space priors are explicitly deferred to maintain a lightweight codebase without external C++/sampler dependencies. Analytical conjugate Bayesian estimation (see `BVARModel` in [Implemented Models Catalog](library/models.md#35-analytical-conjugate-bayesian-var-bvarmodel)) is in scope because its posterior moments are available in closed form.
- **High-Dimensional Penalized VARs**: Regularized L1/L2 lasso and elastic net VAR estimators are deferred to a dedicated high-dimensional module.

### 6.2 Model Registry and Utility Tooling Harmonization
- **Estimator and Utility Decoupling**: Separate standalone `ModelBase` statistical estimators from post-estimation computation engines (`GIRFEngine`) and batch runners (`SpecificationRunner`).
- **Restricted VAR Pipeline Integration**: Expose `RestrictedVAR` through `params.yaml` either via a dedicated `restricted_var` model key or through `model.var.restrictions` configuration in `VARModel`.
- **Multi-Specification Runner Relocation**: Relocate `SpecificationRunner` from `src/stats_transformer/models/regression/` to `src/stats_transformer/pipeline/` or `src/stats_transformer/reporting/` to maintain `models/` exclusively for single-model estimators.
- **Simulation Engine Relocation**: Relocate `GIRFEngine` to a dedicated time series diagnostics or IRF calculation module.

