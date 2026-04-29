# Planned Extensions: Econometric Models

https://sites.google.com/site/lkilian2019/textbook/code?authuser=0

This document outlines the planned extensions for the `stats-transformer` library, specifically drawn from the simulated datasets and model templates provided in the `data/examples/Ghysels/` chapters.

## 1. ARMA / ARIMA Models
* **Source:** Chapter 5
* **Target Script:** `Chap5.R`
* **Data Input:** N/A (Uses dynamically simulated data via `arima.sim` like `arma.sim$y1`)
* **Background:** Autoregressive Moving Average (ARMA) models are a staple of univariate time series forecasting. They are used to model the relationship between a variable and its own past values and random shocks. Adding ARIMA (with integration) will serve as a foundational building block for the library.

## 2. Bayesian VAR (BVAR)
* **Source:** Chapter 8
* **Target Script:** `Chap8.R`
* **Data Input:** `bvar_simulated_ch8_sec4.csv`
* **Background:** BVAR applies Bayesian methods to estimate Vector Autoregressive models. By introducing prior distributions on the model coefficients, BVARs can effectively handle over-parameterization issues common in high-dimensional standard VAR models.

## 3. Non-Linear Time Series (TAR & STAR)
* **Source:** Chapter 9
* **Target Script:** `Chap9.R`
* **Data Input:** Simulated dynamically via Gaussian noise.
* **Background:** Threshold Autoregressive (TAR) and Smooth Transition Autoregressive (STAR) models allow for non-linear behavior in time series. They assume the series exhibits different dynamics depending on whether it falls above or below a certain threshold.

## 4. Markov Switching Models
* **Source:** Chapter 10
* **Target Script:** `Chap10.R`
* **Data Input:** `simulated_ch10_sec5.csv`
* **Background:** Markov Switching models assume that the time series is governed by a hidden state (or regime) that transitions according to a Markov process. This is particularly useful for modeling business cycles (e.g., expansion vs. recession).

## 5. Time-Varying Parameter (TVP) Regression
* **Source:** Chapter 11
* **Target Script:** `Chap11.R`
* **Data Input:** `tvp_simulated_ch11_sec4.csv`
* **Background:** TVP models allow the regression coefficients to change over time, typically modeled using a state-space formulation and the Kalman filter. It captures structural changes dynamically rather than with static dummy variables.

## 6. MIDAS (Mixed Data Sampling)
* **Source:** Chapter 12
* **Target Script:** `Chap12.R`
* **Data Input:** `simulated_data_ch12_sec5.csv`
* **Background:** MIDAS models enable regression analysis when variables are sampled at different frequencies (e.g., predicting quarterly GDP using daily stock market returns). This avoids the need for ad-hoc temporal aggregation.

## 7. Factor Models / Three-Pass Regression Filter (TPRF)
* **Source:** Chapter 13
* **Target Script:** `Chap13.R`
* **Data Input:** Real-world panels or simulated factor data.
* **Background:** Factor models reduce the dimensionality of large datasets into a few unobserved components. TPRF is an advanced method to extract factors that are specifically relevant to forecasting a given target.

## 8. ARCH / GARCH Volatility Models
* **Source:** Chapter 14
* **Target Script:** `Chap14.R`
* **Data Input:** `OxfordManRealizedVolatility.csv`
* **Background:** Autoregressive Conditional Heteroskedasticity (ARCH) models and their generalized forms (GARCH) are used to model time-varying volatility in high-frequency financial series, where volatility clusters.

## 9. Structural Break Tests & Recursive Forecasting
* **Source:** Chapters 3 & 4
* **Target Script:** `Chap3.R`, `Chap4.R`
* **Data Input:** `simulated_datac3.csv`, `simulated_datac4.csv`
* **Background:** Enhances existing OLS and VAR models with diagnostic tests for regime shifts, Chow tests, and recursive forecasting validation loops.
