# Phase 3 Deferred Scope

During the initial implementation of Phase 3 (Data-Driven SVAR Identification), we focused on the two primary estimators to reach Milestone D parity quickly:
1. Changes in Volatility (Rigobon 2003)
2. Distance Covariance (Matteson and Tsay 2017)

The following items were deferred to a future iteration:

- **Cramér-von Mises (CVM) Identification**: Deferred to ensure the core optimization logic for independence-based estimators is stable first.
- **Non-Gaussian Maximum Likelihood**: Deferred to limit scope creep; requires explicit distributional choices and convergence diagnostics.
- **Distance Covariance O(T log T) Fast Approximation**: The current implementation of Distance Covariance uses the exact $O(T^2)$ calculation to perfectly match R `svars` parity. A fast $O(T \log T)$ approximation (e.g., Huo and Székely 2016) could be introduced later for performance on large samples, but requires careful benchmarking to ensure numerical deviations are acceptable.
