# Planned Extensions: Econometric Models

https://sites.google.com/site/lkilian2019/textbook/code?authuser=0

This document outlines the planned extensions for the `stats-transformer` library, specifically drawn from the simulated datasets and model templates provided in the `data/examples/Ghysels/` chapters.

## Implementation Principles

- Prefer proven numerical backends (`statsmodels`, `linearmodels`, `arch`, or focused optional dependencies) over hand-rolled estimators.
- Keep model outputs compatible with `get_model_metadata()` so visualization and report generation remain consistent.
- Mark new estimators experimental until they have deterministic tests against known examples.
- Add config examples before promoting a model to stable status.

## Priority Order

1. **ARIMA and forecasting utilities:** broad user demand, strong backend support, and natural fit with existing time-series examples.
2. **Structural break and recursive forecasting diagnostics:** improves existing regression and VAR workflows without a large new API.
3. **MIDAS:** high value for macro data because the package already emphasizes frequency alignment.
4. **BVAR and Markov switching:** useful, but should arrive after metadata and plotting conventions are stable.
5. **TVP, TAR/STAR, factor models, and GARCH:** important specialized models, best treated as optional or experimental modules at first.

## 1. ARMA / ARIMA Models
* **Source:** Chapter 5
* **Target Script:** `Chap5.R`
* **Data Input:** N/A (Uses dynamically simulated data via `arima.sim` like `arma.sim$y1`)
* **Background:** Autoregressive Moving Average (ARMA) models are a staple of univariate time series forecasting. They are used to model the relationship between a variable and its own past values and random shocks. Adding ARIMA (with integration) will serve as a foundational building block for the library.
* **Planned API:** `ARIMAModel(order=(p, d, q), seasonal_order=None, date_column="date")`.
* **Validation:** compare coefficients and forecasts against `statsmodels.tsa.arima.model.ARIMA` examples.

## 2. Bayesian VAR (BVAR)
* **Source:** Chapter 8
* **Target Script:** `Chap8.R`
* **Data Input:** `bvar_simulated_ch8_sec4.csv`
* **Background:** BVAR applies Bayesian methods to estimate Vector Autoregressive models. By introducing prior distributions on the model coefficients, BVARs can effectively handle over-parameterization issues common in high-dimensional standard VAR models.
* **Planned API:** `BayesianVARModel(prior="minnesota", lags=4, horizons=20)`.
* **Validation:** simulated impulse responses with known signs and shrinkage behavior.

## 3. Non-Linear Time Series (TAR & STAR)
* **Source:** Chapter 9
* **Target Script:** `Chap9.R`
* **Data Input:** Simulated dynamically via Gaussian noise.
* **Background:** Threshold Autoregressive (TAR) and Smooth Transition Autoregressive (STAR) models allow for non-linear behavior in time series. They assume the series exhibits different dynamics depending on whether it falls above or below a certain threshold.
* **Planned API:** `ThresholdARModel(threshold_variable=None, threshold=None, regimes=2)`.
* **Validation:** simulated regime-switching series with known threshold.

## 4. Markov Switching Models
* **Source:** Chapter 10
* **Target Script:** `Chap10.R`
* **Data Input:** `simulated_ch10_sec5.csv`
* **Background:** Markov Switching models assume that the time series is governed by a hidden state (or regime) that transitions according to a Markov process. This is particularly useful for modeling business cycles (e.g., expansion vs. recession).
* **Planned API:** `MarkovSwitchingModel(regimes=2, switching_variance=True)`.
* **Validation:** compare smoothed regime probabilities against `statsmodels` examples.

## 5. Time-Varying Parameter (TVP) Regression
* **Source:** Chapter 11
* **Target Script:** `Chap11.R`
* **Data Input:** `tvp_simulated_ch11_sec4.csv`
* **Background:** TVP models allow the regression coefficients to change over time, typically modeled using a state-space formulation and the Kalman filter. It captures structural changes dynamically rather than with static dummy variables.
* **Planned API:** `TimeVaryingParameterModel(state_covariance="estimated")`.
* **Validation:** synthetic coefficients that drift according to a known state process.

## 6. MIDAS (Mixed Data Sampling)
* **Source:** Chapter 12
* **Target Script:** `Chap12.R`
* **Data Input:** `simulated_data_ch12_sec5.csv`
* **Background:** MIDAS models enable regression analysis when variables are sampled at different frequencies (e.g., predicting quarterly GDP using daily stock market returns). This avoids the need for ad-hoc temporal aggregation.
* **Planned API:** `MIDASRegressionModel(low_frequency_target, high_frequency_features, lag_weights="beta")`.
* **Validation:** mixed-frequency simulation plus an applied GDP nowcasting example.

## 7. Factor Models / Three-Pass Regression Filter (TPRF)
* **Source:** Chapter 13
* **Target Script:** `Chap13.R`
* **Data Input:** Real-world panels or simulated factor data.
* **Background:** Factor models reduce the dimensionality of large datasets into a few unobserved components. TPRF is an advanced method to extract factors that are specifically relevant to forecasting a given target.
* **Planned API:** `DynamicFactorModel(n_factors=1)` and `ThreePassRegressionFilter(n_factors=1)`.
* **Validation:** simulated latent factors with known loadings.

## 8. ARCH / GARCH Volatility Models
* **Source:** Chapter 14
* **Target Script:** `Chap14.R`
* **Data Input:** `OxfordManRealizedVolatility.csv`
* **Background:** Autoregressive Conditional Heteroskedasticity (ARCH) models and their generalized forms (GARCH) are used to model time-varying volatility in high-frequency financial series, where volatility clusters.
* **Planned API:** `GARCHModel(p=1, q=1, distribution="normal")`.
* **Validation:** use the optional `arch` package as backend and compare volatility paths on known examples.

## 9. Structural Break Tests & Recursive Forecasting
* **Source:** Chapters 3 & 4
* **Target Script:** `Chap3.R`, `Chap4.R`
* **Data Input:** `simulated_datac3.csv`, `simulated_datac4.csv`
* **Background:** Enhances existing OLS and VAR models with diagnostic tests for regime shifts, Chow tests, and recursive forecasting validation loops.
* **Planned API:** `StructuralBreakDiagnostics` and `RecursiveForecastEvaluator`.
* **Validation:** simulated break dates and rolling-origin forecast backtests.
