# Documentation Overview

`stats-transformer` unites feature engineering, frequency-aware data alignment, econometric estimation, and automated publication reporting into a cohesive pipeline. The documentation is organized into focused, modular guides to help researchers and developers navigate from system architecture to mathematical specifications, software benchmarks, and future roadmaps.

---

## 1. Quick Navigation Hub

| If you want to... | Start here |
| --- | --- |
| Understand pipeline workflows, package boundaries, and repository layout | [System architecture & repository structure](library/architecture.md) |
| Inspect mathematical formulas, parameters, and access modes for implemented models | [Implemented models catalog](library/models.md) |
| Browse packaged datasets, data layouts, and example panel inputs | [Data directory guide](library/data.md) |
| Review academic literature citations and methodology foundations | [Academic citations](library/citations.md) |
| Inspect cross-language software verification against Stata, MATLAB, and R | [Cross-language software benchmarks](validation/benchmarks.md) |
| Run automated unit and integration tests | [Testing suite](validation/testing_suite.md) |
| Review planned model extensions, complexity tiers, and future milestones | [Future model roadmap](roadmap.md) |

---

## 2. Documentation Map

| Section | Page | Description | Key Audience |
| --- | --- | --- | --- |
| **Project Entry** | [README](../README.md) | Quickstart guide, installation, and YAML pipeline demo | All users |
| **Architecture** | [Architecture](library/architecture.md) | Pipeline stages, package tree, and repository file structure | Developers & pipeline users |
| **Models** | [Implemented Models](library/models.md) | Complete reference of all 30 implemented model classes | Econometricians & modelers |
| **Data** | [Data Guide](library/data.md) | Data directory organization and packaged example datasets | Data engineers & analysts |
| **Citations** | [Citations](library/citations.md) | Literature references, data sources, and software citations | Academic researchers |
| **Benchmarks** | [Benchmarks](validation/benchmarks.md) | Cross-language numerical benchmarks (Stata, MATLAB, R) and replications | Empirical validators |
| **Testing** | [Testing Suite](validation/testing_suite.md) | Test runner guide and pytest test categorization | Contributors & QA |
| **Roadmap** | [Roadmap](roadmap.md) | Triaged planned model extensions (Tiers 1 to 4) | Research collaborators |
| **Tutorials** | `notebooks/07_chart_components.ipynb` | Composable visualization components walkthrough | Visualizers |
| **Tutorials** | `notebooks/08_structural_timeseries_models.ipynb` | Direct-use structural VAR, LP, and decomposition APIs | Advanced time-series users |

---

## 3. Library Model Summary

`stats-transformer` implements 30 specialized model classes and diagnostic engines across five functional domains. Every model is accessible via YAML pipeline configuration (`model.model_type`) or direct Python class instantiation.

| Functional Domain | Key Implemented Classes | Primary Capabilities |
| --- | --- | --- |
| **Applied Regression** | `RegressionModel`, `RobustOLSModel`, `PanelRegressionModel`, `IV2SLSModel`, `PanelIV2SLSModel`, `GMMModel`, `DiDModel`, `SpecificationRunner` | OLS, robust covariance (HC/HAC), fixed/random effects panel, 2SLS, GMM, staggered DiD |
| **Discrete Choice** | `LogitModel`, `ProbitModel` | Maximum-likelihood binary classification and marginal effects |
| **Unsupervised Learning** | `PCAModel`, `KMeansModel` | Standardized PCA factor extraction and K-means clustering |
| **Reduced-Form Time Series** | `VARModel`, `VECMModel`, `RestrictedVAR`, `ARIMAModel`, `BVARModel`, `DynamicFactorModel`, `LocalProjectionsModel`, `LocalProjectionsIVModel` | Multivariate dynamics, cointegration, analytical BVAR, Kalman factor extraction, Jordà local projections |
| **Structural Identification** | `SVARModel`, `BlanchardQuahModel`, `ProxySVARModel`, `SignZeroSVARModel`, `VolatilitySVARModel`, `IndependenceSVARModel`, `CVMSVARModel`, `NonGaussianSVARModel`, `SVECModel` | Short-run $AB$, long-run $C(1)$, external instruments (SVAR-IV), sign/zero/narrative restrictions, volatility breaks, ICA, Copula distance, Student-$t$ QMLE, structural VECM |
| **Nonlinear Dynamics** | `TVARModel`, `TVECMModel`, `STVARModel`, `GIRFEngine` | Two-regime threshold VAR/VECM, smooth transition VAR, generalized impulse responses |
| **Diagnostics & Utilities** | `GrangerCausalityTester`, `ResidualDiagnostics`, `StabilityDiagnostics`, `StationarityDiagnostics`, `VARLagSelector`, `VARForecaster`, `ForecastEvaluator`, `TimeSeriesDecompositions` | Causality tests, companion stability, unit root tests, lag selection criteria, point forecasting, historical shock decomposition |

For full mathematical definitions, estimation algorithms, parameter tables, and benchmark status for every class, consult the [Implemented Models Catalog](library/models.md).

---

## 4. Verification & Validation Standards

The library maintains strict protocols for empirical and numerical validation:

- **Automated Unit & Integration Testing**: Over 350 test cases running under continuous integration. See [Testing suite](validation/testing_suite.md).
- **Cross-Language Software Parity**: Automated benchmark suites verifying machine precision against StataNow 19.5, MATLAB 2025b, and containerized R. See [Cross-language software benchmarks](validation/benchmarks.md).
- **Academic Replications**: Runnable Python translations of empirical macroeconomic papers with bundled datasets.
