# stats-transformer: Planned Extensions Roadmap

This document catalogs the triaged roadmap for future extensions, focusing on the frequentist VAR family, structural identification, and nonlinear dynamics.

---

## 1. Planned Extensions (Triaged)

The following extensions are planned and triaged by implementation difficulty.

### Low Effort / Near-Term
These extensions rely on existing estimation loops and require mostly specification masking, diagnostic utilities, or minor algebraic extensions.
- **Restricted VAR**: Equation-level coefficient constraints via masking matrix. *(Benchmark: R `vars` / Kilian Ch. 3)*
- **Residual Diagnostics Suite**: Serial correlation (Portmanteau), normality (skew/kurtosis), ARCH-LM, and stability roots tests integrated into a unified `diagnostics` module. *(Benchmark: R `vars`)*
- **SVEC (Structural VECM)**: Combining cointegration terms with short-run and long-run structural restrictions. *(Benchmark: R `vars`)*

### Medium Effort
These models introduce new identification paradigms, custom sampling/rotations, or multi-start optimizers.
- **Sign & Zero Restrictions (`SignZeroSVARModel`)**: Configurable YAML schema for shock responses, QR orthogonal rotations, and accepted-draw inference (non-Bayesian set identification). *(Benchmark: VAR-Toolbox `Uhlig2005`)*
- **Narrative Restrictions**: Constraining the sign or magnitude of structural shocks and historical contributions during specific historical events. *(Benchmark: VAR-Toolbox `ADRR2018`)*
- **Data-Driven SVAR (Independence/ICA)**: Identification via non-Gaussianity or distance-covariance metrics, requiring permutation alignment and optimizer diagnostics. *(Benchmark: R `svars`)*

### High Effort / Non-Linear
These models depart from linear global dynamics and require complex threshold grid-searches, logistic transition optimizations, and simulation-based inference.
- **Threshold VAR (TVAR)**: Two-regime threshold search, regime classification, and regime-specific forecasting. *(Benchmark: R `tsDyn`)*
- **Threshold VECM (TVECM)**: TVAR logic applied to cointegrated systems. *(Benchmark: R `tsDyn`)*
- **Smooth Transition VAR (STVAR)**: Bounded logistic transition function optimization. *(Benchmark: R `sstvars`)*
- **Generalized IRF (GIRF)**: Simulation-based state-dependent impulse responses for non-linear models. *(Benchmark: R `sstvars`)*

---

## 2. Currently Implemented Models

The list of currently implemented models (e.g., `VARModel`, `SVARModel`, `ProxySVARModel`, `PanelRegressionModel`) and their exact numerical validation targets against MATLAB, R, or Stata is maintained separately. 

Please refer to the **[Replications Registry](../validation/replications.md)** for the active matrix of verified models and test paths.
