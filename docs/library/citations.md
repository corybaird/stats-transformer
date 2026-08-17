# Academic Citations & Data Sources

This document catalogs the academic literature, econometric papers, benchmark datasets, and reference software packages cited and implemented within `stats-transformer`.

---

## 1. Primary Academic Literature

### 1.1 Structural VAR Identification & Cointegration

- **Blanchard, O. J., & Quah, D. (1989)**. 
	- Blanchard, O. J., & Quah, D. (1988). The dynamic effects of aggregate demand and supply disturbances.
	  - ***Usage*:** `BlanchardQuahModel`
			  - src/stats_transformer/models/timeseries/identification/blanchard_quah.py`
			  - Benchmarked: `src/examples/academic/var/blanchard_quah_1989.py`.
- **King, R. G., Plosser, C. I., Stock, J. H., & Watson, M. W. (1991)**. 
	- King, R. G., Plosser, C. I., Stock, J. H., & Watson, M. W. (1987). Stochastic trends and economic fluctuations.
	  - ***Usage*:** Structural VECM (SVEC) long-run cointegration decomposition
		  - `src/stats_transformer/models/timeseries/structural/svec.py`
- **Johansen, S. (1991)**.
	- Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in the presence of linear trend. _Econometrica_, _59_(6), 1551-1580.
	- ***Usage*:** Cointegration rank tests and VECM estimation
		- `src/stats_transformer/models/timeseries/reduced_form/vecm.py`
- **Kilian, L., & Lütkepohl, H. (2017)**.
	- Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*. Cambridge University Press.
	- ***Usage*:** Theoretical foundation for Structural VAR identification, restricted VAR estimation, historical decompositions, sign restrictions, and structural bootstrap methods.

### 1.2 Sign, Zero, and Narrative Restrictions

- **Faust, J. (1998)**. 
	- Faust, J. (1998, December). The robustness of identified VAR conclusions about money. In _Carnegie-Rochester conference series on public policy_ (Vol. 49, pp. 207-244). North-Holland.
	  - ***Usage*:** Inequality sign restriction algorithms for structural identification.
- **Canova, F., & De Nicolo, G. (2002)**. 
	- Canova, F., & De Nicolo, G. (2002). Monetary disturbances matter for business fluctuations in the G-7. _Journal of Monetary Economics_, _49_(6), 1131-1159.
	  - ***Usage*:** Cross-correlation and sign-based structural identification.
- **Uhlig, H. (2005)**. 
	- Uhlig, H. (2005). What are the effects of monetary policy on output? Results from an agnostic identification procedure. _Journal of Monetary Economics_, _52_(2), 381-419.
	  - ***Usage*:** Sign restriction identification engine for monetary shocks.
- **Rubio-Ramírez, J. F., Waggoner, D. F., & Zha, T. (2010)**. 
	- Rubio-Ramirez, J. F., Waggoner, D. F., & Zha, T. (2010). Structural vector autoregressions: Theory of identification and algorithms for inference. _The Review of Economic Studies_, _77_(2), 665-696.
	- ***Usage*:** Implemented in `SignZeroSVARModel`
		- `src/stats_transformer/models/timeseries/identification/sign_zero.py`
- **Antolín-Díaz, J., & Rubio-Ramírez, J. F. (2018)**. 
	- Antolín-Díaz, J., & Rubio-Ramírez, J. F. (2018). Narrative sign restrictions for SVARs. _American Economic Review_, _108_(10), 2802-2829.
	  - ***Usage*:** Event-specific shock sign and magnitude constraints in structural identification routines.
- **Arias, J. F., Rubio-Ramírez, J. F., & Waggoner, D. F. (2018)**. 
	- Arias, Jonas E., Juan F. Rubio‐Ramírez, and Daniel F. Waggoner. "Inference based on structural vector autoregressions identified with sign and zero restrictions: Theory and applications." _Econometrica_ 86.2 (2018): 685-720.
	  - *Usage*: Joint sign and zero restriction QR decomposition algorithms.

### 1.3 Proxy SVAR & High-Frequency Identification

- **Kuttner, K. N. (2001)**. 
	- Kuttner, Kenneth N. "Monetary policy surprises and interest rates: Evidence from the Fed funds futures market." _Journal of monetary economics_ 47.3 (2001): 523-544.
	  - *Usage*: High-frequency futures surprise identification.
- **Romer, C. D., & Romer, D. H. (2004)**.
	- Romer, Christina D., and David H. Romer. "A new measure of monetary shocks: Derivation and implications." _American economic review_ 94.4 (2004): 1055-1084.
	  - *Usage*: Narrative monetary policy shock series.
- **Gürkaynak, R. S., Sack, B., & Swanson, E. T. (2005)**. 
	- Gurkaynak, Refet S., Brian Sack, and Eric T. Swanson. "Do actions speak louder than words? The response of asset prices to monetary policy actions and statements." (2005): 55-93.
	  - *Usage*: High-frequency central bank announcement surprise measures.
- **Mertens, K., & Ravn, M. O. (2013)**. 
	- Mertens, K., & Ravn, M. O. (2013). The dynamic effects of personal and corporate income tax changes in the United States. _American economic review_, _103_(4), 1212-1247.
	- *Usage*: Implemented in `ProxySVARModel` 
		- `src/stats_transformer/models/timeseries/identification/proxy_svar.py`
- **Gertler, M., & Karadi, P. (2015)**. 
	- Gertler, M., & Karadi, P. (2015). Monetary policy surprises, credit costs, and economic activity. _American Economic Journal: Macroeconomics_, _7_(1), 44-76.
	  - *Usage*: External-instrument SVAR benchmarked 
		  - `src/examples/academic/var/gertler_karadi_2015.py`.

### 1.4 Local Projections

- **Jordà, Ó. (2005)**. 
	- Jordà, Ò. (2005). Estimation and inference of impulse responses by local projections. _American economic review_, _95_(1), 161-182.
	  - *Usage*: Implemented in `LocalProjectionsModel`
		  - `src/stats_transformer/models/timeseries/reduced_form/local_projections.py`
- **Stock, J. H., & Watson, M. W. (2018)**. 
	- Stock, J. H., & Watson, M. W. (2018). Identification and estimation of dynamic causal effects in macroeconomics using external instruments. _The Economic Journal_, _128_(610), 917-948.
	- *Usage*: Implemented in `LocalProjectionsIVModel`
		- `src/stats_transformer/models/timeseries/reduced_form/local_projections_iv.py`
- **Plagborg-Møller, M., & Wolf, C. K. (2021)**. 
	- Plagborg‐Møller, M., & Wolf, C. K. (2021). Local projections and VARs estimate the same impulse responses. _Econometrica_, _89_(2), 955-980.
	  - *Usage*: Equivalence mapping between Local Projections and VAR impulse response dynamics.

### 1.4a Dynamic Factor Models

- **Miranda-Agrippino, S., & Rey, H. (2020)**.
	- Miranda-Agrippino, S., & Rey, H. (2020). US monetary policy and the global financial cycle. _The Review of Economic Studies_, _87_(6), 2754-2776.
	- *Usage*: Illustrative EM-estimated `DynamicFactorModel` extraction, benchmarked against the paper's published Global Factor reference series in
		- `src/stats_transformer/models/timeseries/reduced_form/dynamic_factor.py`
		- `src/examples/academic/miranda_agrippino_rey_2020.py`

### 1.4b Generalized Method of Moments

- **Hansen, L. P. (1982)**.
	- Hansen, L. P. (1982). Large sample properties of generalized method of moments estimators. _Econometrica_, _50_(4), 1029-1054.
	- *Usage*: Implemented in `GMMModel` (one-step, two-step, iterated, CUE weighting; overidentification J-test)
		- `src/stats_transformer/models/regression/gmm.py`
- **Coibion, O., & Gorodnichenko, Y. (2012)**.
	- Coibion, O., & Gorodnichenko, Y. (2012). What can survey forecasts tell us about information rigidities? _Journal of Political Economy_, _120_(1), 116-159.
	- *Usage*: Simplified single-series forecast-error orthogonality test benchmarked in
		- `src/examples/academic/coibion_gorodnichenko_2012.py`

### 1.4c Staggered Difference-in-Differences

- **Callaway, B., & Sant'Anna, P. H. (2021)**.
	- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. _Journal of Econometrics_, _225_(2), 200-230.
	- *Usage*: Implemented in `DiDModel` (group-time ATT(g,t), never-treated / not-yet-treated control groups, event-study and simple aggregation, pre-trend testing)
		- `src/stats_transformer/models/regression/did.py`
- **Goodman-Bacon, A. (2021)**.
	- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. _Journal of Econometrics_, _225_(2), 254-277.
	- *Usage*: Motivates the negative-weighting failure mode of naive two-way fixed effects under dynamic treatment effects, tested against in `tests/test_models/test_did_model.py`.
- **Lane, N. (2025)**.
	- Lane, N. (2025). Manufacturing Revolutions: Industrial Policy and Industrialization in South Korea. _Quarterly Journal of Economics_ (forthcoming).
	- *Usage*: Illustrative `DiDModel` fit on a tariff-liberalization cohort derived from the available extract (not the paper's own treatment definition), in
		- `src/examples/academic/lane_2025.py`

### 1.5 Data-Driven SVAR Identification

- **Rigobon, R. (2003)**. 
	- Rigobon, R. (2003). Identification through heteroskedasticity. _Review of Economics and Statistics_, _85_(4), 777-792.
	- *Usage*: Implemented in `HeteroskedasticSVARModel` 
		- `src/stats_transformer/models/timeseries/identification/volatility.py`
- **Lanne, M., & Saikkonen, P. (2007)**. 
	- Lanne, M., Meitz, M., & Saikkonen, P. (2017). Identification and estimation of non-Gaussian structural vector autoregressions. _Journal of Econometrics_, _196_(2), 288-304.
	- *Usage*: Statistical identification through GARCH and time-varying variance structures.
- **Matteson, D. S., & Tsay, R. S. (2017)**.
	- Matteson, D. S., & Tsay, R. S. (2017). Independent component analysis via distance covariance. _Journal of the American Statistical Association_, _112_(518), 623-637.
	  - *Usage*: Implemented in `IndependenceSVARModel`
		  - `src/stats_transformer/models/timeseries/identification/independence.py`

### 1.6 Nonlinear & Regime-Switching Models

- **Koop, G., Pesaran, M. H., & Potter, S. M. (1996)**. 
	- Koop, G., Pesaran, M. H., & Potter, S. M. (1996). Impulse response analysis in nonlinear multivariate models. _Journal of econometrics_, _74_(1), 119-147.
	  - *Usage*: Generalized Impulse Response Functions (GIRF) simulation engine.
- **Hansen, B. E. (2011)**. 
	- Hansen, B. E. (2011). Threshold autoregression in economics. _Statistics and its Interface_, _4_(2), 123-127.
	  - *Usage*: Threshold VAR (TVAR) regime classification.
- **Teräsvirta, T., & Yang, Y. (2014)**. 
	- Teräsvirta, T., & Yang, Y. (2014). Specification, estimation and evaluation of vector smooth transition autoregressive models with applications.
	  - *Usage*: Smooth Transition VAR (STVAR) formulation.

### 1.7 Applied Time-Series Forecasting & Textbooks

- **Ghysels, E., & Marcellino, M. (2018)**.
	- Ghysels, E., & Marcellino, M. (2018). *Applied Economic Forecasting Using Time Series Methods*. Oxford University Press.
	  - *Usage*: Applied regression, VAR, VECM, and forecasting textbook demonstrations in `src/examples/regression/ghysels_chap1.py`, `ghysels_chap2.py`, `src/examples/timeseries/ghysels_chap6.py`, and `ghysels_chap7.py`.

---

## 2. Academic Datasets & Data Sources

### 2.1 US Macroeconomic Data (FRED-QD / FRED-MD)

- **Description**: Quarterly and monthly macroeconomic datasets maintained by the Federal Reserve Bank of St. Louis.
- **Citation**: McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. _Journal of Business & Economic Statistics_, _34_(4), 574-589.
- **Location**: `data/examples/macrodb_gdp_inflation.parquet`, `data/examples/timeseries/`
- **Usage**: Used for VAR, VECM, LP-IV, and structural monetary shock benchmarks.

### 2.2 Stock & Watson Macro Data (2001)

- **Description**: Three-variable US macroeconomic dataset (Inflation, Unemployment, Federal Funds Rate) from 1960 to 2000.
- **Citation**: Stock, J. H., & Watson, M. W. (2001). Vector autoregressions. _Journal of Economic perspectives_, _15_(4), 101-115.
- **Location**: `src/examples/academic/var/stock_watson_2001.py`
- **Usage**: Standard 3-variable reduced-form VAR demonstration and impulse response comparison.

### 2.3 Nakamura & Steinsson Monetary Surprises (2018)

- **Description**: High-frequency monetary policy surprise series constructed from Federal Funds futures around FOMC announcements.
- **Citation**: Nakamura, E., & Steinsson, J. (2018). High-frequency identification of monetary non-neutrality: the information effect. _The Quarterly Journal of Economics_, _133_(3), 1283-1330.
- **Location**: `src/examples/academic/nakamura_steinsson.py`, `nakamura_steinsson_pca.py`
- **Usage**: External instrument for Proxy SVAR and PCA monetary shock extraction.

### 2.4 Bauer & Swanson Monetary Surprises (2023)

- **Description**: Updated high-frequency monetary policy surprise dataset orthogonalized against economic forecasts.
- **Citation**: Bauer, M. D., & Swanson, E. T. (2023). A reassessment of monetary policy surprises and high-frequency identification. _NBER Macroeconomics Annual_, _37_(1), 87-155.
- **Location**: `src/examples/academic/bauer_swanson.py`
- **Usage**: Monetary shock transformation sanity checks.

### 2.5 Longley Benchmark Dataset (1967)

- **Description**: Highly collinear US economic dataset used for regression numerical stability testing.
- **Citation**: Longley, J. W. (1967). An appraisal of least squares programs for the electronic computer from the point of view of the user. _Journal of the American Statistical association_, _62_(319), 819-841.
- **Location**: `src/examples/regression/longley.py`
- **Usage**: Robust OLS numerical benchmark.

### 2.6 Grunfeld Investment Panel Dataset (1958)

- **Description**: Microeconomic panel dataset of 10 US corporations over 20 years.
- **Citation**: Grunfeld, Y. (1958). _The Determinants of Corporate Investment: A Study of a Number of Large Corporations in the United States_ (Doctoral dissertation, Department of Photoduplication, University of Chicago Library).
- **Location**: `src/examples/regression/grunfeld.py`
- **Usage**: Fixed-effects panel OLS regression benchmark.

### 2.7 Mroz Female Labor Supply Dataset (1987)

- **Description**: Microeconomic dataset for married women's labor force participation and wage equations.
- **Citation**: Mroz, T. A. (1984). _The sensitivity of an empirical model of married women's hours of work to economic and statistical assumptions_. Stanford University.
- **Location**: `src/examples/regression/mroz_iv.py`
- **Usage**: 2SLS Instrumental Variables regression benchmark.

### 2.8 Spector & Mazzeo Educational Logit Dataset (1980)

- **Description**: Binary outcome dataset assessing the effectiveness of a new teaching method (PSI).
- **Citation**: Spector, L. C., & Mazzeo, M. (1980). Probit analysis and economic education. _The Journal of Economic Education_, _11_(2), 37-44.
- **Location**: `src/examples/discrete/spector_logit.py`
- **Usage**: Binary Logit classification benchmark.

### 2.9 Ghysels & Marcellino Time-Series Datasets (2018)

- **Description**: Textbook replication data for US macro monetary series, EU GDP growth, UK term structure, and US leading economic indicators (LEI).
- **Citation**: Ghysels, E., & Marcellino, M. (2018). *Applied Economic Forecasting Using Time Series Methods*. Oxford University Press.
- **Location**: `data/examples/timeseries/ghysels_ch6/`, `data/examples/timeseries/ghysels_ch7/`
- **Usage**: Applied VAR and VECM textbook demonstration scripts (`ghysels_chap1.py`, `ghysels_chap2.py`, `ghysels_chap6.py`, `ghysels_chap7.py`).

---

## 3. Open-Source Benchmark Software

- **R `vars` Package**: Pfaff, B. (2008). VAR, SVAR and SVEC models: Implementation within R package vars. _Journal of statistical software_, _27_, 1-32.
- **R `tsDyn` Package**: Stigler, Matthieu. "Threshold cointegration: overview and implementation in R." _R package version 0.7-2. URL http://stat. ethz. ch/CRAN/web/packages/tsDyn/vignettes/ThCointOverview. pdf_ (2010).
- **R `svars` Package**: Lange, A., Dalheimer, B., Herwartz, H., & Maxand, S. (2021). svars: An R package for data-driven identification in multivariate time series analysis. _Journal of Statistical Software_, _97_,
- **R `sstvars` Package**: 
- **MATLAB VAR-Toolbox**: Cesa-Bianchi, Ambrogio. (2026). *VAR-Toolbox for MATLAB*. GitHub Repository:
	- `https://github.com/ambropo/VAR-Toolbox`.
- **Python `statsmodels`**: Seabold, S., & Perktold, J. (2010). Statsmodels: econometric and statistical modeling with python. _scipy_, _7_(1), 92-96.
- **Python `linearmodels`**: Kevin Sheppard, Snyk bot, Joon Ro, Brian Lewis, Christian Clauss, Guangyi, Jeff, Jerry Qinghui Yu, Jiageng, Kevin Wilson, LGTM Migrator, Thrasibule, WilliamRoyNelson, Xavier RENE-CORAIL& vikjam. (2025).

---

## 4. Citations for Review

### 4.1 Macroeconomics Replication Catalog Literature

- **Acosta, M., Brennan, K., & Jacobson, M. (2024)**. *High-Frequency Identification of Monetary Policy Shocks with SOFR Futures*. Federal Reserve Board Finance and Economics Discussion Series (FEDS 2024-034).
- **Aruoba, S. B., & Drechsel, T. (2024)**. *Identifying Monetary Policy Shocks: A Natural Language Approach*. National Bureau of Economic Research (NBER Working Paper No. 32252).
- **Bauer, M. D., Bernanke, B. S., & Milstein, E. (2023)**. *Risk Appetite and the Transmission of Monetary Policy*. Federal Reserve Bank of San Francisco Working Paper 2023-22.
- **Bauer, M. D., & Swanson, E. T. (2023)**. A reassessment of monetary policy surprises and high-frequency identification. *NBER Macroeconomics Annual*, 37(1), 87–155.
- **Choi, S., Willems, T., & Yoo, J. (2024)**. *Revisiting the Monetary Transmission Mechanism Through the Lens of High-Frequency Identification*. International Monetary Fund Working Paper.
- **Christiano, L. J., Motto, R., & Trabandt, M. (2014)**. Risk shocks. *American Economic Review*, 104(1), 27–65.
- **Cieslak, A., Hansen, S., McMahon, M., & Xiao, S. (2024)**. *Policymakers' Uncertainty*. Becker Friedman Institute for Economics Working Paper.
- **Coibion, O., & Gorodnichenko, Y. (2012)**. What can survey forecasts tell us about information rigidities? *Journal of Political Economy*, 120(1), 116–159.
- **Cormun, B., & De Leo, P. (2024)**. *Exchange Rate Dynamics and Monetary Policy in Open Economies*. Working Paper.
- **Erik, B., Lombardi, M. J., Mihaljek, D., & Shin, H. S. (2020)**. *The Dollar, Bank Leverage and Real Economic Activity: An Evolving Relationship*. BIS Working Papers No. 847.
- **Federal Reserve Bank of San Francisco (2025)**. *Monetary Policy Surprises and Macroeconomic Dynamics Data Series*. Economic Research Data Releases.
- **Fukui, M., Nakamura, E., & Steinsson, J. (2025)**. *The Empirical Trilemma*. National Bureau of Economic Research Working Paper.
- **di Giovanni, J., Levchenko, A. A., & Mejean, I. (2017)**. Large firms and international business cycle comovement. *American Economic Review*, 107(5), 598–602.
- **Jacobson, M., Matthes, C., & Walker, T. (2026)**. *Comparing High-Frequency Shock Identification and Empirical DSGE Models*. Working Paper.
- **Jarociński, M. (2024)**. *Central Bank Information Shocks and the Zero Lower Bound*. European Central Bank Working Paper.
- **Jarociński, M., & Karadi, P. (2020)**. Deconstructing monetary policy surprises: The role of information shocks. *American Economic Journal: Macroeconomics*, 12(2), 1–43.
- **Karadi, P., Schönle, R., & Wursten, J. (2024)**. *Price Setting in a Micro-to-Macro Menu Cost Model*. European Central Bank Working Paper.
- **Kerssenfischer, M., & Schmeling, M. (2024)**. What moves markets? *Journal of Financial Economics*, 156, 103844.
- **Lane, N. (2025)**. *Manufacturing Revolutions: Industrial Policy and Industrialization in South Korea*. Quarterly Journal of Economics (forthcoming).
- **Morrison, J. (2025)**. *The Distributional Effects of Monetary Policy Across Occupations and Skill Groups*. Working Paper.
- **Nakamura, E., & Steinsson, J. (2018)**. High-frequency identification of monetary non-neutrality: the information effect. *The Quarterly Journal of Economics*, 133(3), 1283–1330.
- **Nakamura, E., & Steinsson, J. (2018b)**. Identification in macroeconomics. *Journal of Economic Perspectives*, 32(3), 59–86.
- **Patterson, C., Del Negro, M., & Giannoni, M. P. (2023)**. *The Forward Guidance Puzzle and Survey Expectations*. Federal Reserve Bank of New York Staff Reports.
- **Reis, R. (2009)**. A sticky-information general-equilibrium model for policy analysis. In *Central Banking, Analysis, and Economic Policies Book Series* (Vol. 13, pp. 227–283). Central Bank of Chile.
- **Renault, T. (2017)**. Intraday online investor sentiment and return predictability in the financial markets. *Journal of Banking & Finance*, 84, 25–40.
- **Miranda-Agrippino, S., & Rey, H. (2020)**. US monetary policy and the global financial cycle. *The Review of Economic Studies*, 87(6), 2754–2776.
- **Shapiro, A. H., Sudhof, M., & Wilson, D. J. (2022)**. Measuring news sentiment. *Journal of Econometrics*, 228(2), 221–243.
- **Zhou, X. (2024)**. *Monetary Policy and the Banking System: Micro Evidence from Call Reports*. Working Paper.
