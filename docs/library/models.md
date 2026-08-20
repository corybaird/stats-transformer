# Implemented Models Catalog

`stats-transformer` provides a comprehensive suite of econometric and statistical models. Every model exposes a unified API contract via `ModelBase` and can be executed either programmatically via Python or configured in YAML and orchestrated via `Pipeline`.

---

## 1. Master Model Inventory & Reference Tables

### Column Definitions
- **YAML Pipeline Key**: The string identifier passed to `model.model_type` in `params.yaml` (e.g. `ols`, `var`, `logit`). Classes marked as `None` are invoked exclusively via direct Python API.
- **Direct Python API**: The importable class name for programmatic Python invocation (`from stats_transformer.models import ...`).
- **Benchmark Dataset**: The packaged empirical dataset used to fit and verify the model.
- **Reference Target**: The reference function, package, or toolbox in R, Stata, or MATLAB against which the model is compared.
- **Verification Parity Status**:
  - **Verified**: Parameter point estimates, standard errors, or structural impact matrices match the reference target software (Stata, MATLAB, R) to machine precision in automated benchmark tests.
  - **Implemented**: Class is fully implemented and covered by unit/integration tests in Python, with cross-language benchmarks planned or in development.
  - **Utility Wrapper**: Helper or grid runner orchestrating underlying model fits.

### 1.1 Cross-Language Numerically Verified Models

These models have been verified to machine precision against native Stata, MATLAB, or R routines on empirical benchmark datasets.

| Family | Class Name | YAML Pipeline Key | Direct Python API | Benchmark Dataset | Reference Target | Verification Parity Status |
| --- | --- | --- | --- | --- | --- | --- |
| **Applied Regression** | `RegressionModel` | `ols` | `RegressionModel` | Longley (`data/examples/regression/longley.csv`) | R `stats::lm`, Stata `regress`, MATLAB `mldivide` | **Verified** ($0.00$ vs Stata) |
| **Applied Regression** | `PanelRegressionModel` | `panel_ols` | `PanelRegressionModel` | Grunfeld (`data/examples/regression/grunfeld.csv`) | R `plm::plm`, Stata `xtreg` | **Verified** (R `plm` parity) |
| **Discrete Choice** | `LogitModel` | `logit` | `LogitModel` | Spector & Mazzeo (`data/examples/discrete/spector.csv`) | R `stats::glm`, Stata `logit` | **Verified** ($0.00$ vs Stata) |
| **Unsupervised Learning** | `PCAModel` | `pca` | `PCAModel` | Longley (`data/examples/regression/longley.csv`) | R `stats::prcomp`, Stata `pca` | **Verified** ($0.00$ vs Stata) |
| **Reduced-Form Time Series** | `VARModel` | `var` | `VARModel` | US Macro (`data/examples/timeseries/macrodata.csv`) | R `vars::VAR`, Stata `var`, `statsmodels` | **Verified** ($5.77 \times 10^{-15}$ vs Stata) |
| **Reduced-Form Time Series** | `VECMModel` | `vecm` | `VECMModel` | US Macro (`data/examples/timeseries/macrodata.csv`) | R `urca::ca.jo`, Stata `vec`, MATLAB `vecm` | **Verified** ($2.00 \times 10^{-3}$ vs Stata) |
| **Structural Identification** | `BlanchardQuahModel` | `blanchard_quah` | `BlanchardQuahModel` | Blanchard & Quah (`data/examples/matlab_examples/BQ1989_Data.xlsx`) | MATLAB `VAR-Toolbox 4.0`, R `vars::BQ` | **Verified** ($2.22 \times 10^{-16}$ vs MATLAB) |
| **Structural Identification** | `SVARModel` | `svar` | `SVARModel` | Blanchard & Quah (`data/examples/matlab_examples/BQ1989_Data.xlsx`) | R `vars::SVAR`, Stata `svar`, VAR-Toolbox | **Verified** (R `svars` parity) |
| **Structural Identification** | `VolatilitySVARModel` | `volatility_svar` | `VolatilitySVARModel` | Rigobon Break Panel (`data/examples/regression/grunfeld.csv`) | R `svars::id.cv` | **Verified** (R `svars` parity) |
| **Nonlinear Dynamics** | `TVARModel` | `tvar` | `TVARModel` | US Macro (`data/examples/timeseries/macrodata.csv`) | R `tsDyn::TVAR`, Stata `threshold` | **Verified** (Stata threshold parity) |

### 1.2 Implemented Models (Python Tested, External Parity Benchmarks Planned)

These models are fully implemented and verified via automated Python unit and integration test suites. External cross-language parity comparators are currently planned or in development.

| Family | Class Name | YAML Pipeline Key | Direct Python API | Benchmark Dataset | Reference Target | Verification Parity Status |
| --- | --- | --- | --- | --- | --- | --- |
| **Applied Regression** | `RobustOLSModel` | `robust_ols` | `RobustOLSModel` | Longley (`data/examples/regression/longley.csv`) | R `sandwich::vcovHC`, Stata `regress, robust` | **Implemented** |
| **Applied Regression** | `IV2SLSModel` | `iv` (`iv_2sls`, `2sls`) | `IV2SLSModel` | Mroz (`data/examples/regression/mroz.csv`) | R `AER::ivreg`, Stata `ivregress 2sls` | **Implemented** |
| **Applied Regression** | `PanelIV2SLSModel` | `panel_iv` | `PanelIV2SLSModel` | Grunfeld (`data/examples/regression/grunfeld.csv`) | R `fixest::feols`, Stata `xtivreg` | **Implemented** |
| **Applied Regression** | `GMMModel` | `gmm` | `GMMModel` | Greenbook (`data/examples/academic/coibion_gorodnichenko_2012/`) | R `gmm::gmm`, Stata `gmm` | **Implemented** |
| **Applied Regression** | `DiDModel` | `did` | `DiDModel` | Policy Loans (`data/examples/academic/lane_2025/`) | R `did::att_gt`, Stata `csdid` | **Implemented** |
| **Applied Regression** | `SpecificationRunner` | `spec_runner` | `SpecificationRunner` | Longley (`data/examples/regression/longley.csv`) | Custom grid runner | **Utility Wrapper** |
| **Discrete Choice** | `ProbitModel` | `probit` | `ProbitModel` | Spector & Mazzeo (`data/examples/discrete/spector.csv`) | R `stats::glm(probit)`, Stata `probit` | **Implemented** |
| **Unsupervised Learning** | `KMeansModel` | `kmeans` | `KMeansModel` | Spector & Mazzeo (`data/examples/discrete/spector.csv`) | R `stats::kmeans`, Stata `cluster kmeans` | **Implemented** |
| **Reduced-Form Time Series** | `RestrictedVAR` | `None` | `RestrictedVAR` | US Macro (`data/examples/timeseries/macrodata.csv`) | R `vars::restrict`, Stata `var (constrained)` | **Implemented** |
| **Reduced-Form Time Series** | `ARIMAModel` | `arima` | `ARIMAModel` | US Macro (`data/examples/timeseries/macrodata.csv`) | R `forecast::auto.arima`, Stata `arima` | **Implemented** |
| **Reduced-Form Time Series** | `BVARModel` | `bvar` | `BVARModel` | FOMC Surprises (`data/examples/academic/jarocinski_karadi_2020/`) | R `BVAR::bvar`, Stata `bayes: var` | **Implemented** |
| **Reduced-Form Time Series** | `DynamicFactorModel` | `dynamic_factor` | `DynamicFactorModel` | Global Factor (`data/examples/academic/miranda_agrippino_rey_2020/`) | R `MARSS`, Stata `dfactor` | **Implemented** |
| **Reduced-Form Time Series** | `LocalProjectionsModel` | `local_projections` | `LocalProjectionsModel` | JT Macro (`data/examples/matlab_examples/JT2025_Data.xlsx`) | R `lpirfs::lp_lin`, Stata `jorda` | **Implemented** |
| **Reduced-Form Time Series** | `LocalProjectionsIVModel` | `lp_iv` | `LocalProjectionsIVModel` | JT Macro (`data/examples/matlab_examples/JT2025_Data.xlsx`) | R `lpirfs::lp_lin_iv`, Stata `lproj` | **Implemented** |
| **Structural Identification** | `ProxySVARModel` | `proxy_svar` | `ProxySVARModel` | GK Macro (`data/examples/matlab_examples/GK2015_Data.xlsx`) | R `svars`, MATLAB `VAR-Toolbox 4.0` | **Implemented** |
| **Structural Identification** | `SignZeroSVARModel` | `sign_restrictions` (`sign_zero`) | `SignZeroSVARModel` | Kilian & Lütkepohl (`data/examples/timeseries/macrodata.csv`) | MATLAB `VAR-Toolbox 4.0`, R `BMR` | **Implemented** |
| **Structural Identification** | `IndependenceSVARModel` | `independence_svar` | `IndependenceSVARModel` | FastICA Macro Panel (`data/examples/timeseries/macrodata.csv`) | R `svars::id.dc` | **Implemented** |
| **Structural Identification** | `CVMSVARModel` | `cvm_svar` (`cvm`) | `CVMSVARModel` | FastICA Macro Panel (`data/examples/timeseries/macrodata.csv`) | R `svars::id.cvm` | **Implemented** |
| **Structural Identification** | `NonGaussianSVARModel` | `non_gaussian_svar` (`non_gaussian`) | `NonGaussianSVARModel` | FastICA Macro Panel (`data/examples/timeseries/macrodata.csv`) | R `svars::id.ng` | **Implemented** |
| **Structural Identification** | `SVECModel` | `svec` | `SVECModel` | King et al. (`data/examples/matlab_examples/SW2001_Data.xlsx`) | R `vars::SVEC`, MATLAB `VAR-Toolbox 4.0` | **Implemented** |
| **Nonlinear Dynamics** | `TVECMModel` | `tvecm` | `TVECMModel` | Cointegration Panel (`data/examples/timeseries/ghysels_ch7/`) | R `tsDyn::TVECM` | **Implemented** |
| **Nonlinear Dynamics** | `STVARModel` | `stvar` | `STVARModel` | Smooth Transition Panel (`data/examples/timeseries/macrodata.csv`) | R `sstvars::STVAR` | **Implemented** |
| **Nonlinear Dynamics** | `GIRFEngine` | `None` | `GIRFEngine` | Smooth Transition Panel (`data/examples/timeseries/macrodata.csv`) | R `sstvars::GIRF` | **Implemented** |

---

## 2. Applied Regression Models

### 2.1 Ordinary Least Squares (`RegressionModel`)
- **Description**: Standard linear regression estimated via OLS, supporting entity fixed-effects dummy variables and constant intercepts.
- **Specification**: $y = X \beta + u$, where $\mathbb{E}[u | X] = 0$.
- **Module Location**: `src/stats_transformer/models/regression/regression.py`
- **Pipeline Access**: `ols` (direct instantiation: `RegressionModel`)
- **Benchmark Target**: R (`stats::lm`), Stata (`regress`), MATLAB (`mldivide`)
- **Verified Discrepancy**: Machine precision (Stata: $0.00$, MATLAB: $3.52 \times 10^{-6}$, R: $2.07 \times 10^{-6}$)

### 2.2 Robust OLS (`RobustOLSModel`)
- **Description**: OLS regression with heteroskedasticity-consistent (HC0, HC1, HC2, HC3) or autocorrelation-consistent (HAC / Newey-West) standard errors.
- **Specification**: $\text{Var}(\hat\beta) = (X'X)^{-1} X' \Omega X (X'X)^{-1}$
- **Module Location**: `src/stats_transformer/models/regression/robust_ols.py`
- **Pipeline Access**: `robust_ols` (direct instantiation: `RobustOLSModel`)
- **Benchmark Target**: R (`sandwich::vcovHC`), Stata (`regress, robust`)

### 2.3 Panel Regression (`PanelRegressionModel`)
- **Description**: Static panel data regression supporting entity fixed effects, two-way fixed effects, and random effects.
- **Specification**: $y_{it} = \alpha_i + \gamma_t + x_{it}' \beta + \epsilon_{it}$
- **Module Location**: `src/stats_transformer/models/regression/panel.py`
- **Pipeline Access**: `panel_ols` (direct instantiation: `PanelRegressionModel`)
- **Benchmark Target**: R (`plm::plm`), Stata (`xtreg`)
- **Verified Discrepancy**: Verified against R `plm` in integration test suite.

### 2.4 Instrumental Variables 2SLS (`IV2SLSModel`)
- **Description**: Two-Stage Least Squares (2SLS) instrumental variables estimation for endogenous regressors.
- **Specification**: First stage: $\hat X = Z (Z'Z)^{-1} Z' X$; Second stage: $y = \hat X \beta + u$.
- **Module Location**: `src/stats_transformer/models/regression/iv.py`
- **Pipeline Access**: `iv` (aliases: `iv_2sls`, `2sls`)
- **Benchmark Target**: R (`AER::ivreg`), Stata (`ivregress 2sls`), Python (`linearmodels.iv`)

### 2.5 Panel IV 2SLS (`PanelIV2SLSModel`)
- **Description**: Panel instrumental variables estimation combining fixed effects with two-stage least squares.
- **Specification**: $\tilde y_{it} = \tilde X_{it} \beta + \tilde u_{it}$, where regressors are projected on instruments $\tilde Z_{it}$ after within-transformation.
- **Module Location**: `src/stats_transformer/models/regression/panel_iv.py`
- **Pipeline Access**: `panel_iv` (direct instantiation: `PanelIV2SLSModel`)
- **Benchmark Target**: R (`fixest::feols`), Stata (`xtivreg`)

### 2.6 Generalized Method of Moments (`GMMModel`)
- **Description**: GMM estimation supporting one-step, two-step, iterated, and continuous updating estimator (CUE) formulations with HAC weighting matrices and Sargan-Hansen $J$-tests of overidentifying restrictions.
- **Specification**: $\mathbb{E}[g(w_t, \theta_0)] = 0$, minimizing $\hat g(\theta)' W \hat g(\theta)$.
- **Module Location**: `src/stats_transformer/models/regression/gmm.py`
- **Pipeline Access**: `gmm` (direct instantiation: `GMMModel`)
- **Benchmark Target**: R (`gmm::gmm`), Stata (`gmm`)

### 2.7 Difference-in-Differences (`DiDModel`)
- **Description**: Callaway and Sant'Anna staggered difference-in-differences estimator identifying group-time average treatment effects ($ATT(g, t)$).
- **Specification**: Identification under conditional parallel trends using not-yet-treated or never-treated comparison groups.
- **Module Location**: `src/stats_transformer/models/regression/did.py`
- **Pipeline Access**: `did` (direct instantiation: `DiDModel`)
- **Benchmark Target**: R (`did::att_gt`), Stata (`csdid`)

### 2.8 Specification Runner (`SpecificationRunner`)
- **Description**: Multi-specification grid runner executing and compiling regressions across multiple combinations of controls, fixed effects, and clustering rules.
- **Module Location**: `src/stats_transformer/models/regression/spec_runner.py`
- **Pipeline Access**: `spec_runner` (direct instantiation: `SpecificationRunner`)

---

## 3. Discrete Choice & Unsupervised Models

### 3.1 Binary Logit (`LogitModel`)
- **Description**: Maximum likelihood binary logit classification model with robust covariance options and average marginal effects.
- **Specification**: $P(y = 1 | x) = \Lambda(x' \beta) = \frac{1}{1 + \exp(-x' \beta)}$
- **Module Location**: `src/stats_transformer/models/discrete/logit.py`
- **Pipeline Access**: `logit` (direct instantiation: `LogitModel`)
- **Benchmark Target**: R (`stats::glm`), Stata (`logit`)
- **Verified Discrepancy**: Machine precision (Stata: $0.00$, R: $1.81 \times 10^{-9}$)

### 3.2 Binary Probit (`ProbitModel`)
- **Description**: Maximum likelihood binary probit classification model under standard normal cumulative density.
- **Specification**: $P(y = 1 | x) = \Phi(x' \beta) = \int_{-\infty}^{x' \beta} \frac{1}{\sqrt{2\pi}} e^{-z^2/2} dz$
- **Module Location**: `src/stats_transformer/models/discrete/probit.py`
- **Pipeline Access**: `probit` (direct instantiation: `ProbitModel`)
- **Benchmark Target**: R (`stats::glm(family=binomial(link="probit"))`), Stata (`probit`)

### 3.3 Principal Component Analysis (`PCAModel`)
- **Description**: Standardized PCA feature extraction producing factor loadings, scree variance shares, and orthogonal latent components.
- **Specification**: Spectral decomposition of correlation matrix $\Sigma_X = V \Lambda V'$.
- **Module Location**: `src/stats_transformer/models/unsupervised/pca.py`
- **Pipeline Access**: `pca` (direct instantiation: `PCAModel`)
- **Benchmark Target**: R (`stats::prcomp`), Stata (`pca`)
- **Verified Discrepancy**: Machine precision (Stata: $0.00$, R: $3.33 \times 10^{-16}$)

### 3.4 K-Means Clustering (`KMeansModel`)
- **Description**: Standardized K-means clustering algorithm partitioning observations into $K$ disjoint clusters by minimizing within-cluster sum of squares.
- **Specification**: $\arg\min_S \sum_{i=1}^K \sum_{x \in S_i} \| x - \mu_i \|^2$
- **Module Location**: `src/stats_transformer/models/unsupervised/kmeans.py`
- **Pipeline Access**: `kmeans` (direct instantiation: `KMeansModel`)
- **Benchmark Target**: R (`stats::kmeans`), Stata (`cluster kmeans`)

---

## 4. Reduced-Form Time Series Models

### 4.1 Vector Autoregression (`VARModel`)
- **Description**: Linear multivariate time series system estimating dynamic interactions across $K$ endogenous variables with $p$ lags.
- **Specification**: $Y_t = c + \sum_{i=1}^p A_i Y_{t-i} + u_t$, where $u_t \sim \mathcal{N}(0, \Sigma_u)$.
- **Module Location**: `src/stats_transformer/models/timeseries/reduced_form/var.py`
- **Pipeline Access**: `var` (direct instantiation: `VARModel`)
- **Benchmark Target**: R (`vars::VAR`), Stata (`var`), Python (`statsmodels`)
- **Verified Discrepancy**: Stata $5.77 \times 10^{-15}$, `statsmodels` $0.00$.

### 4.2 Vector Error Correction Model (`VECMModel`)
- **Description**: Cointegrated multivariate time series model capturing long-run equilibrium relationships alongside short-run adjustment dynamics.
- **Specification**: $\Delta Y_t = \Pi Y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta Y_{t-i} + c + u_t$, where $\Pi = \alpha \beta'$.
- **Module Location**: `src/stats_transformer/models/timeseries/reduced_form/vecm.py`
- **Pipeline Access**: `vecm` (direct instantiation: `VECMModel`)
- **Benchmark Target**: R (`urca::ca.jo`), Stata (`vec`), MATLAB (`vecm`)
- **Verified Discrepancy**: Stata $2.00 \times 10^{-3}$ parameter agreement.

### 4.3 Restricted VAR (`RestrictedVAR`)
- **Description**: VAR enforced with structural zero restrictions on lag coefficients at the equation level.
- **Specification**: Mask matrix $M \in \{0, 1\}^{K \times (K \cdot p + 1)}$ constraining specific coefficients to zero.
- **Module Location**: `src/stats_transformer/models/timeseries/reduced_form/restrictions.py`
- **Pipeline Access**: Direct instantiation via `RestrictedVAR`
- **Benchmark Target**: R (`vars::restrict`), Stata (`var, constraint`)

### 4.4 Univariate ARIMA (`ARIMAModel`)
- **Description**: Autoregressive Integrated Moving Average model for single-series time-series forecasting.
- **Specification**: $\Phi(L)(1-L)^d y_t = c + \Theta(L) \epsilon_t$
- **Module Location**: `src/stats_transformer/models/timeseries/arima.py`
- **Pipeline Access**: `arima` (direct instantiation: `ARIMAModel`)
- **Benchmark Target**: R (`forecast::auto.arima`), Stata (`arima`)

### 4.5 Analytical Conjugate Bayesian VAR (`BVARModel`)
- **Description**: Conjugate Normal-Inverse-Wishart prior BVAR with Minnesota prior moments, estimated analytically in closed form without MCMC dependencies.
- **Specification**: $B | \Sigma \sim \mathcal{MN}(\bar B, \Sigma \otimes \bar V)$, $\Sigma \sim \mathcal{IW}(\bar S, \bar\nu)$.
- **Module Location**: `src/stats_transformer/models/timeseries/reduced_form/bvar.py`
- **Pipeline Access**: `bvar` (direct instantiation: `BVARModel`)
- **Benchmark Target**: R (`BVAR::bvar`)

### 4.6 Dynamic Factor Model (`DynamicFactorModel`)
- **Description**: High-dimensional dynamic factor extraction estimated via Expectation-Maximization (EM) algorithm and Kalman filter/smoother.
- **Specification**: $X_t = \Lambda F_t + e_t$, $F_t = A F_{t-1} + u_t$.
- **Module Location**: `src/stats_transformer/models/timeseries/reduced_form/dynamic_factor.py`
- **Pipeline Access**: `dynamic_factor` (direct instantiation: `DynamicFactorModel`)
- **Benchmark Target**: R (`MARSS`), Stata (`dfactor`)

### 4.7 Local Projections (`LocalProjectionsModel`)
- **Description**: Jordà (2005) multi-horizon direct regression estimating impulse response functions robust to misspecification of lag dynamics.
- **Specification**: $y_{t+h} = \alpha_h + \beta_h x_t + \sum_{i=1}^p \gamma_{h,i} W_{t-i} + \epsilon_{t+h}$ for $h = 0, 1, \dots, H$.
- **Module Location**: `src/stats_transformer/models/timeseries/reduced_form/local_projections.py`
- **Pipeline Access**: `local_projections` (direct instantiation: `LocalProjectionsModel`)
- **Benchmark Target**: R (`lpirfs::lp_lin`), Stata (`jorda`)

### 4.8 Local Projections IV (`LocalProjectionsIVModel`)
- **Description**: Instrumented local projections (Stock & Watson 2018) using external narrative or high-frequency surprises as instruments for policy treatments.
- **Specification**: 2SLS estimation at each horizon $h$ with treatment $x_t$ instrumented by $z_t$.
- **Module Location**: `src/stats_transformer/models/timeseries/reduced_form/local_projections_iv.py`
- **Pipeline Access**: `lp_iv` (direct instantiation: `LocalProjectionsIVModel`)
- **Benchmark Target**: R (`lpirfs::lp_lin_iv`), Stata (`lproj`)

---

## 5. Structural Identification (SVAR & SVEC)

### 5.1 Short-Run SVAR (`SVARModel`)
- **Description**: Structural VAR identification enforcing linear recursive (Cholesky) or exact/over-identifying $AB$-model matrix restrictions.
- **Specification**: $A u_t = B \epsilon_t$, with $\epsilon_t \sim (0, I_K)$ and $A \Sigma_u A' = B B'$.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/svar.py`
- **Pipeline Access**: `svar` (direct instantiation: `SVARModel`)
- **Benchmark Target**: R (`vars::SVAR`), Stata (`svar`), MATLAB (`VAR-Toolbox 4.0`)

### 5.2 Blanchard-Quah Long-Run SVAR (`BlanchardQuahModel`)
- **Description**: Identification imposing cumulative long-run neutrality restrictions on structural shocks ($C(1) = (I - \sum A_i)^{-1} B$ is triangular).
- **Specification**: Supply and demand shock identification following Blanchard and Quah (1989).
- **Module Location**: `src/stats_transformer/models/timeseries/identification/blanchard_quah.py`
- **Pipeline Access**: `blanchard_quah` (direct instantiation: `BlanchardQuahModel`)
- **Benchmark Target**: MATLAB (`VAR-Toolbox 4.0`), R (`vars::BQ`)
- **Verified Discrepancy**: Machine precision ($2.22 \times 10^{-16}$ maximum absolute difference vs MATLAB).

### 5.3 Proxy SVAR / SVAR-IV (`ProxySVARModel`)
- **Description**: Structural shock identification using external instruments correlated with target shock but orthogonal to other structural shocks.
- **Specification**: Two-stage regression of reduced-form residuals on instrument $z_t$ identifying impact vector $b_1$.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/proxy_svar.py`
- **Pipeline Access**: `proxy_svar` (direct instantiation: `ProxySVARModel`)
- **Benchmark Target**: R (`svars`), MATLAB (`VAR-Toolbox 4.0`)

### 5.4 Sign, Zero, & Narrative Restrictions (`SignZeroSVARModel`)
- **Description**: Non-Bayesian set identification imposing directional inequality signs, exact zero impact constraints, and historical narrative restrictions (Antolín-Díaz & Rubio-Ramírez 2018).
- **Algorithm**: QR decomposition of random orthonormal rotation matrices $Q$ with acceptance rejection loop.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/sign_zero.py`
- **Pipeline Access**: `sign_restrictions` (alias: `sign_zero`)
- **Benchmark Target**: MATLAB (`VAR-Toolbox 4.0`), R (`BMR`)

### 5.5 Changes in Volatility SVAR (`VolatilitySVARModel`)
- **Description**: Heteroskedastic identification across discrete variance regimes (Rigobon 2003).
- **Specification**: $\Sigma_u^{(1)} = B B'$ and $\Sigma_u^{(2)} = B \Lambda B'$, where $\Lambda$ is a diagonal eigenvalue matrix.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/volatility.py`
- **Pipeline Access**: `volatility_svar` (direct instantiation: `VolatilitySVARModel`)
- **Benchmark Target**: R (`svars::id.cv`)
- **Verified Discrepancy**: Verified against R `svars` in integration test suite.

### 5.6 Distance Covariance & ICA SVAR (`IndependenceSVARModel`)
- **Description**: Data-driven structural identification by maximizing statistical independence across shock components via FastICA or distance covariance minimization.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/independence.py`
- **Pipeline Access**: `independence_svar` (direct instantiation: `IndependenceSVARModel`)
- **Benchmark Target**: R (`svars::id.dc`)

### 5.7 Cramér-von Mises Distance SVAR (`CVMSVARModel`)
- **Description**: Data-driven structural identification minimizing Cramér-von Mises distance of joint empirical copulas.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/cvm.py`
- **Pipeline Access**: `cvm_svar` (alias: `cvm`)
- **Benchmark Target**: R (`svars::id.cvm`)

### 5.8 Non-Gaussian Maximum Likelihood SVAR (`NonGaussianSVARModel`)
- **Description**: Structural identification assuming non-Gaussian (Student-$t$) shock distributions estimated via numerical maximum likelihood.
- **Module Location**: `src/stats_transformer/models/timeseries/identification/non_gaussian.py`
- **Pipeline Access**: `non_gaussian_svar` (alias: `non_gaussian`)
- **Benchmark Target**: R (`svars::id.ng`)

### 5.9 Structural VECM (`SVECModel`)
- **Description**: Combining long-run cointegrating vectors with structural short-run and permanent/transitory shock decompositions (King et al. 1991).
- **Module Location**: `src/stats_transformer/models/timeseries/structural/svec.py`
- **Pipeline Access**: `svec` (direct instantiation: `SVECModel`)
- **Benchmark Target**: R (`vars::SVEC`), MATLAB (`VAR-Toolbox 4.0`)

---

## 6. Non-Linear & Regime-Switching Models

### 6.1 Threshold Vector Autoregression (`TVARModel`)
- **Description**: Two-regime threshold VAR where dynamic coefficients switch based on whether a transition variable $y_{t-d}$ exceeds threshold $\gamma$.
- **Specification**: $Y_t = (A_0^{(1)} + \sum A_i^{(1)} Y_{t-i}) \mathbb{I}(s_{t-d} \le \gamma) + (A_0^{(2)} + \sum A_i^{(2)} Y_{t-i}) \mathbb{I}(s_{t-d} > \gamma) + u_t$.
- **Module Location**: `src/stats_transformer/models/timeseries/nonlinear/tvar.py`
- **Pipeline Access**: `tvar` (direct instantiation: `TVARModel`)
- **Benchmark Target**: R (`tsDyn::TVAR`), Stata (`threshold`)
- **Verified Discrepancy**: Stata threshold search and parameter parity verified.

### 6.2 Threshold VECM (`TVECMModel`)
- **Description**: Threshold cointegrated vector error correction model switching regimes based on error correction deviation magnitude.
- **Module Location**: `src/stats_transformer/models/timeseries/nonlinear/tvecm.py`
- **Pipeline Access**: `tvecm` (direct instantiation: `TVECMModel`)
- **Benchmark Target**: R (`tsDyn::TVECM`)

### 6.3 Smooth Transition VAR (`STVARModel`)
- **Description**: Continuous regime-switching model with logistic or exponential smooth transition weights.
- **Specification**: $Y_t = (1 - G(s_t; \gamma, c)) F_1(Y) + G(s_t; \gamma, c) F_2(Y) + u_t$.
- **Module Location**: `src/stats_transformer/models/timeseries/nonlinear/stvar.py`
- **Pipeline Access**: `stvar` (direct instantiation: `STVARModel`)
- **Benchmark Target**: R (`sstvars::STVAR`)

### 6.4 Generalized Impulse Response Functions (`GIRFEngine`)
- **Description**: History-dependent and shock-sign-dependent impulse response simulation engine for nonlinear multivariate models (Koop, Pesaran, & Potter 1996).
- **Module Location**: `src/stats_transformer/models/timeseries/nonlinear/girf.py`
- **Pipeline Access**: Direct utility instantiation via `GIRFEngine`
- **Benchmark Target**: R (`sstvars::GIRF`)

---

## 7. Time-Series Diagnostics & Utilities

| Class / Function | Module Location | Responsibility |
| --- | --- | --- |
| `GrangerCausalityTester` | `src/stats_transformer/models/timeseries/granger.py` | Bivariate and system Granger causality Wald F-tests |
| `ResidualDiagnostics` | `src/stats_transformer/models/timeseries/diagnostics/residuals.py` | Portmanteau Q-test, Jarque-Bera multivariate normality, ARCH-LM |
| `StabilityDiagnostics` | `src/stats_transformer/models/timeseries/diagnostics/stability.py` | Companion matrix eigenvalue modulus calculations and unit circle plots |
| `StationarityDiagnostics` | `src/stats_transformer/models/timeseries/diagnostics/stationarity.py` | Augmented Dickey-Fuller (ADF) and KPSS unit root testing |
| `VARLagSelector` | `src/stats_transformer/models/timeseries/reduced_form/lag_selection.py` | Information criteria evaluation across lag orders ($AIC, HQ, SC/BIC, FPE$) |
| `VARForecaster` | `src/stats_transformer/models/timeseries/reduced_form/forecasting.py` | Iterative point forecasting with analytical confidence intervals |
| `ForecastEvaluator` | `src/stats_transformer/models/timeseries/analysis/forecast_evaluation.py` | Out-of-sample RMSE, MAE, and directional forecast evaluation |
| `TimeSeriesDecompositions` | `src/stats_transformer/models/timeseries/structural/decompositions.py` | Historical shock decomposition and forecast error variance decomposition |
