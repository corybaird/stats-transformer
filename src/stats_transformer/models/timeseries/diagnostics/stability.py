import numpy as np
import pandas as pd

class StabilityDiagnostics:
    """
    Stability diagnostics for VAR models.
    Matches conventions for companion matrix roots and OLS-CUSUM tests.
    """
    def __init__(self, model_result):
        self.res = model_result
        self.k_ar = model_result.k_ar
        self.neqs = len(model_result.names)
        self.params = np.asarray(model_result.params)
        self.k_trend = getattr(model_result, "k_trend", 1)

    def companion_matrix(self):
        """
        Constructs the companion matrix of the VAR.
        """
        if self.k_trend > 0:
            lag_params = self.params[self.k_trend:]
        else:
            lag_params = self.params
            
        dim = self.neqs * self.k_ar
        companion = np.zeros((dim, dim))
        
        # Top block: A_1, A_2, ..., A_p
        # lag_params shape is (p*K, K), where rows are L1.y1, L1.y2..., L2.y1...
        for i in range(self.k_ar):
            companion[0:self.neqs, i*self.neqs:(i+1)*self.neqs] = lag_params[i*self.neqs:(i+1)*self.neqs].T
            
        # Lower blocks: Identity matrices
        if self.k_ar > 1:
            companion[self.neqs:, 0:(self.k_ar-1)*self.neqs] = np.eye((self.k_ar-1)*self.neqs)
            
        return companion
        
    def roots(self):
        """
        Computes the roots of the characteristic polynomial (eigenvalues of companion matrix).
        VAR is stable if all moduli are strictly less than 1.
        """
        comp = self.companion_matrix()
        eigenvalues = np.linalg.eigvals(comp)
        return eigenvalues
        
    def is_stable(self):
        """
        Checks if the VAR system is stable.
        """
        eigenvalues = self.roots()
        moduli = np.abs(eigenvalues)
        return np.all(moduli < 1.0)
        
    def ols_cusum(self):
        """
        Computes the OLS-CUSUM empirical fluctuation process.
        Returns the process and 95% critical bounds.
        """
        # A full OLS-CUSUM requires recursive residuals and the empirical
        # fluctuation process, which is not implemented. Must fail rather
        # than silently return a zero process with fixed bounds (see
        # stats-transformer#47).
        raise NotImplementedError("OLS-CUSUM is not implemented.")
