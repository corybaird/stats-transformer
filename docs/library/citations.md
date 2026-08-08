# Academic Citations & Data Sources

This document catalogs the academic literature, econometric papers, benchmark datasets, and reference software packages cited and implemented within `stats-transformer`.

---

## 1. Primary Academic Literature

### 1.1 Structural VAR Identification & Cointegration

- **Blanchard, O. J., & Quah, D. (1989)**. The Dynamic Effects of Aggregate Demand and Supply Disturbances. *American Economic Review*, 79(4), 655–673.
  - *Usage*: Implemented in `BlanchardQuahModel` (`src/stats_transformer/models/timeseries/identification/blanchard_quah.py`) and benchmarked in `src/examples/academic/var/blanchard_quah_1989.py`.
- **King, R. G., Plosser, C. I., Stock, J. H., & Watson, M. W. (1991)**. Stochastic Trends and Economic Fluctuations. *American Economic Review*, 81(4), 819–840.
  - *Usage*: Structural VECM (SVEC) long-run cointegration decomposition (`src/stats_transformer/models/timeseries/structural/svec.py`).
- **Johansen, S. (1991)**. Estimation and Hypothesis Testing of Cointegration Vectors in Vector Autoregressive Models. *Econometrica*, 59(6), 1551–1580.
  - *Usage*: Cointegration rank tests and VECM estimation (`src/stats_transformer/models/timeseries/reduced_form/vecm.py`).

### 1.2 Sign, Zero, and Narrative Restrictions

- **Faust, J. (1998)**. The Robustness of Identified VAR Conclusions About Money. *Carnegie-Rochester Conference Series on Public Policy*, 49, 207–244.
  - *Usage*: Inequality sign restriction algorithms for structural identification.
- **Canova, F., & De Nicolo, G. (2002)**. Monetary Disturbances Matter for Business Fluctuations in the G-7. *Journal of Monetary Economics*, 49(6), 1131–1159.
  - *Usage*: Cross-correlation and sign-based structural identification.
- **Uhlig, H. (2005)**. What Are the Effects of Monetary Policy on Output? Results from an Agnostic Identification Procedure. *Journal of Monetary Economics*, 52(2), 381–419.
  - *Usage*: Sign restriction identification engine for monetary shocks.
- **Rubio-Ramírez, J. F., Waggoner, D. F., & Zha, T. (2010)**. Structural Vector Autoregressions: Theory of Identification and Algorithms for Inference. *Review of Economic Studies*, 77(2), 665–696.
  - *Usage*: Implemented in `SignZeroSVARModel` (`src/stats_transformer/models/timeseries/identification/sign_zero.py`).
- **Antolín-Díaz, J., & Rubio-Ramírez, J. F. (2018)**. Narrative Sign Restrictions for SVARs. *American Economic Review*, 108(10), 2802–2829.
  - *Usage*: Event-specific shock sign and magnitude constraints in structural identification routines.
- **Arias, J. F., Rubio-Ramírez, J. F., & Waggoner, D. F. (2018)**. Inference Based on Structural Vector Autoregressions Identified with Sign and Zero Restrictions: Theory and Applications. *Econometrica*, 86(2), 685–720.
  - *Usage*: Joint sign and zero restriction QR decomposition algorithms.

### 1.3 Proxy SVAR & High-Frequency Identification

- **Kuttner, K. N. (2001)**. Monetary Policy Surprises and Interest Rates: Evidence from the Fed Funds Futures Market. *Journal of Monetary Economics*, 47(3), 523–544.
  - *Usage*: High-frequency futures surprise identification.
- **Romer, C. D., & Romer, D. H. (2004)**. A New Measure of Monetary Shocks: Derivation and Implications. *American Economic Review*, 94(4), 1055–1084.
  - *Usage*: Narrative monetary policy shock series.
- **Gürkaynak, R. S., Sack, B., & Swanson, E. T. (2005)**. Do Actions Speak Louder Than Words? The Response of Asset Prices to Monetary Policy Actions and Statements. *International Journal of Central Banking*, 1(1), 55–93.
  - *Usage*: High-frequency central bank announcement surprise measures.
- **Mertens, K., & Ravn, M. O. (2013)**. The Dynamic Effects of Personal and Corporate Income Tax Changes in the United States. *American Economic Review*, 103(4), 1212–1247.
  - *Usage*: Implemented in `ProxySVARModel` (`src/stats_transformer/models/timeseries/identification/proxy_svar.py`).
- **Gertler, M., & Karadi, P. (2015)**. Monetary Policy Surprises, Credit Costs, and Economic Activity. *American Economic Journal: Macroeconomics*, 7(1), 44–76.
  - *Usage*: External-instrument SVAR benchmarked in `src/examples/academic/var/gertler_karadi_2015.py`.

### 1.4 Local Projections

- **Jordà, Ó. (2005)**. Estimation and Inference of Impulse Responses by Local Projections. *American Economic Review*, 95(1), 161–182.
  - *Usage*: Implemented in `LocalProjectionsModel` (`src/stats_transformer/models/timeseries/reduced_form/local_projections.py`).
- **Stock, J. H., & Watson, M. W. (2018)**. Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments. *Economic Journal*, 128(610), 917–948.
  - *Usage*: Implemented in `LocalProjectionsIVModel` (`src/stats_transformer/models/timeseries/reduced_form/local_projections_iv.py`).
- **Plagborg-Møller, M., & Wolf, C. K. (2021)**. Local Projections and VARs Estimate the Same Impulse Responses. *Econometrica*, 89(2), 955–980.
  - *Usage*: Equivalence mapping between Local Projections and VAR impulse response dynamics.

### 1.5 Data-Driven SVAR Identification

- **Rigobon, R. (2003)**. Identification Through Heteroskedasticity. *Review of Economics and Statistics*, 85(4), 777–792.
  - *Usage*: Implemented in `HeteroskedasticSVARModel` (`src/stats_transformer/models/timeseries/identification/volatility.py`).
- **Lanne, M., & Saikkonen, P. (2007)**. A Note on Identification of Statistical Vector Autoregressions. *Economics Letters*, 95(2), 249–253.
  - *Usage*: Statistical identification through GARCH and time-varying variance structures.
- **Matteson, D. S., & Tsay, R. S. (2017)**. Independent Component Analysis via Distance Covariance. *Journal of the American Statistical Association*, 112(518), 623–637.
  - *Usage*: Implemented in `IndependenceSVARModel` (`src/stats_transformer/models/timeseries/identification/independence.py`).

### 1.6 Nonlinear & Regime-Switching Models

- **Koop, G., Pesaran, M. H., & Potter, S. M. (1996)**. Impulse Response Analysis in Nonlinear Multivariate Models. *Journal of Econometrics*, 74(1), 119–147.
  - *Usage*: Generalized Impulse Response Functions (GIRF) simulation engine.
- **Hansen, B. E. (2011)**. Threshold Autoregression in Economics. *Statistics and Its Interface*, 4(2), 123–127.
  - *Usage*: Threshold VAR (TVAR) regime classification.
- **Teräsvirta, T., & Yang, Y. (2014)**. Specification, Estimation and Evaluation of Smooth Transition Structural Vector Autoregressive Models. *CREATES Research Paper*, 2014-04.
  - *Usage*: Smooth Transition VAR (STVAR) formulation.

---

## 2. Academic Datasets & Data Sources

### 2.1 US Macroeconomic Data (FRED-QD / FRED-MD)

- **Description**: High-dimensional quarterly and monthly macroeconomic datasets maintained by the Federal Reserve Bank of St. Louis.
- **Citation**: McCracken, M. W., & Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic Research. *Journal of Business & Economic Statistics*, 34(4), 574–589.
- **Location**: `data/raw/macrodb_gdp_inflation.parquet`, `data/examples/timeseries/`
- **Usage**: Used for VAR, VECM, LP-IV, and structural monetary shock benchmarks.

### 2.2 Stock & Watson Macro Data (2001)

- **Description**: Three-variable US macroeconomic dataset (Inflation, Unemployment, Federal Funds Rate) from 1960 to 2000.
- **Citation**: Stock, J. H., & Watson, M. W. (2001). Vector Autoregressions. *Journal of Economic Perspectives*, 15(4), 101–115.
- **Location**: `src/examples/academic/var/stock_watson_2001.py`
- **Usage**: Standard 3-variable reduced-form VAR demonstration and impulse response comparison.

### 2.3 Nakamura & Steinsson Monetary Surprises (2018)

- **Description**: High-frequency monetary policy surprise series constructed from Federal Funds futures around FOMC announcements.
- **Citation**: Nakamura, E., & Steinsson, J. (2018). High-Frequency Identification of Monetary Non-Neutrality: The Information Effect. *Quarterly Journal of Economics*, 133(3), 1283–1330.
- **Location**: `src/examples/academic/nakamura_steinsson.py`, `nakamura_steinsson_pca.py`
- **Usage**: External instrument for Proxy SVAR and PCA monetary shock extraction.

### 2.4 Bauer & Swanson Monetary Surprises (2023)

- **Description**: Updated high-frequency monetary policy surprise dataset orthogonalized against economic forecasts.
- **Citation**: Bauer, M. D., & Swanson, E. T. (2023). A Reassessment of Monetary Policy Surprises and High-Frequency Identification. *NBER Macroeconomics Annual*, 37, 87–155.
- **Location**: `src/examples/academic/bauer_swanson.py`
- **Usage**: Monetary shock transformation sanity checks.

### 2.5 Longley Benchmark Dataset (1967)

- **Description**: Highly collinear US economic dataset used for regression numerical stability testing.
- **Citation**: Longley, J. W. (1967). An Appraisal of Least Squares Programs for the Electronic Computer from the Point of View of the User. *Journal of the American Statistical Association*, 62(319), 819–841.
- **Location**: `src/examples/regression/longley.py`
- **Usage**: Robust OLS numerical benchmark.

### 2.6 Grunfeld Investment Panel Dataset (1958)

- **Description**: Microeconomic panel dataset of 10 US corporations over 20 years.
- **Citation**: Grunfeld, Y. (1958). *The Determinants of Corporate Investment*. Ph.D. dissertation, University of Chicago.
- **Location**: `src/examples/regression/grunfeld.py`
- **Usage**: Fixed-effects panel OLS regression benchmark.

### 2.7 Mroz Female Labor Supply Dataset (1987)

- **Description**: Microeconomic dataset for married women's labor force participation and wage equations.
- **Citation**: Mroz, T. A. (1987). The Sensitivity of an Empirical Model of Married Women's Hours of Work to Economic and Statistical Assumptions. *Econometrica*, 55(4), 765–799.
- **Location**: `src/examples/regression/mroz_iv.py`
- **Usage**: 2SLS Instrumental Variables regression benchmark.

### 2.8 Spector & Mazzeo Educational Logit Dataset (1980)

- **Description**: Binary outcome dataset assessing the effectiveness of a new teaching method (PSI).
- **Citation**: Spector, L. C., & Mazzeo, M. (1980). Probit Analysis and Economic Education. *Journal of Economic Education*, 11(2), 37–44.
- **Location**: `src/examples/discrete/spector_logit.py`
- **Usage**: Binary Logit classification benchmark.

---

## 3. Open-Source Benchmark Software

- **R `vars` Package**: Pfaff, B. (2008). VAR, SVAR and SVEC Models: Implementation Within R Package vars. *Journal of Statistical Software*, 27(4), 1–28.
- **R `tsDyn` Package**: Stigler, M. (2010). Threshold, Cointegration, and Non-Linearity: Introducing the R Package tsDyn. *R Journal*.
- **R `svars` Package**: Lange, N., Feldkircher, M., & Siklos, P. L. (2021). Data-Driven Identification of SVAR Models in R: The svars Package. *Journal of Statistical Software*, 97(5), 1–34.
- **R `sstvars` Package**: Savonen, M. (2024). sstvars: Smooth Transition Vector Autoregressive Models in R. *CRAN Package*.
- **MATLAB VAR-Toolbox**: Cesa-Bianchi, A. (2026). *VAR-Toolbox for MATLAB*. GitHub Repository: `https://github.com/ambropo/VAR-Toolbox`.
- **Python `statsmodels`**: Seabold, S., & Perktold, J. (2010). statsmodels: Econometric and Statistical Modeling in Python. *Proceedings of the 9th Python in Science Conference*.
- **Python `linearmodels`**: Sheppard, K. (2021). linearmodels: Linear Econometric Models in Python. *GitHub Repository*.
