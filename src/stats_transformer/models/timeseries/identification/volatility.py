import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.api import VAR
from stats_transformer.models.base import ModelBase
from stats_transformer.models.timeseries.identification.alignment import align_to_cholesky

class VolatilitySVARModel(ModelBase):
    """
    Data-driven SVAR identification via Changes in Volatility (Rigobon 2003).
    Assumes structural impact matrix B is constant, but structural shock variances change across regimes.
    """
    _is_multivariate = True

    def __init__(self, target_variables=None, regime_column=None, maxlags=1, **kwargs):
        super().__init__(**kwargs)
        
        self.target_variables = target_variables or []
        self.regime_column = regime_column
        self.maxlags = maxlags
        
        self.var_result = None
        self.structural_impact = None
        self.lambda_diag = None # Structural variances in regime 2 relative to regime 1
        
    def build_model(self):
        if getattr(self, 'df_clean', None) is None:
            raise ValueError("No cleaned data available")
        if not self.regime_column or self.regime_column not in self.df_clean.columns:
            raise ValueError("A regime column must be specified for Volatility identification.")
            
        self.y = self.df_clean[self.target_variables]
        var_model = VAR(self.y)
        self.var_result = var_model.fit(maxlags=self.maxlags)
        self.model = self.var_result
        
        self._identify_shocks()
        return self.var_result
        
    def _identify_shocks(self):
        """
        Identifies the structural impact matrix B using changes in volatility.
        Let Sigma_1 be the residual covariance in regime 1, and Sigma_2 in regime 2.
        Sigma_1 = B B' (Assuming unit variance in regime 1)
        Sigma_2 = B Lambda B' (Where Lambda is diagonal variance in regime 2)
        """
        # Get residuals and align with data
        resid = self.var_result.resid.values if isinstance(self.var_result.resid, pd.DataFrame) else self.var_result.resid
        regimes = self.df_clean[self.regime_column].values[-resid.shape[0]:]
        
        unique_regimes = np.unique(regimes)
        if len(unique_regimes) != 2:
            raise NotImplementedError("VolatilitySVARModel currently only supports exactly 2 regimes.")
            
        r1, r2 = unique_regimes
        u1 = resid[regimes == r1]
        u2 = resid[regimes == r2]
        
        Sigma1 = np.cov(u1.T)
        Sigma2 = np.cov(u2.T)
        
        # Identification via joint diagonalization (or generalized eigenvalue problem)
        # Sigma1^{-1} Sigma2 = (B B')^{-1} B Lambda B' = (B')^{-1} Lambda B'
        # Thus, the columns of (B')^{-1} are the right eigenvectors of Sigma1^{-1} Sigma2
        # and the eigenvalues are the diagonal elements of Lambda.
        
        S1_inv_S2 = np.linalg.inv(Sigma1) @ Sigma2
        
        eigenvalues, eigenvectors = np.linalg.eig(S1_inv_S2)
        
        # B' = eigenvectors^{-1}  => B = (eigenvectors^{-1})'
        B_unscaled = np.linalg.inv(eigenvectors).T
        
        # We need to scale B such that B B' = Sigma1
        # B_unscaled B_unscaled' might not equal Sigma1 because eigenvectors are arbitrarily scaled
        # Let B = B_unscaled * c (where c is a diagonal scaling matrix)
        # B B' = B_unscaled c^2 B_unscaled' = Sigma1
        # c^2 = diag( (B_unscaled)^{-1} Sigma1 (B_unscaled')^{-1} )
        
        B_inv = np.linalg.inv(B_unscaled)
        c2 = np.diag(B_inv @ Sigma1 @ B_inv.T)
        c = np.sqrt(np.abs(c2)) # absolute value for safety, should be positive
        
        B = B_unscaled @ np.diag(c)
        
        # Resolve permutation and sign indeterminacy by aligning to the Cholesky factor
        p_chol = np.linalg.cholesky(Sigma1)
        self.structural_impact = align_to_cholesky(B, p_chol)
        
        # Store relative variances
        # We need to reorder eigenvalues (Lambda) according to the permutation used for B
        # But align_to_cholesky doesn't return the permutation. We compute it from structural_impact
        # To find lambda for each column of the aligned matrix:
        # B_aligned = B_unscaled * c * P * S
        # So lambda for the i-th column of B_aligned is the variance of the i-th structural shock in regime 2
        # E(e_2 e_2') = Lambda_aligned
        # e_2 = B_aligned^{-1} u_2
        e2 = np.linalg.inv(self.structural_impact) @ u2.T
        self.lambda_diag = np.var(e2, axis=1)

    def get_summary(self):
        if self.structural_impact is None:
            raise ValueError("Model not trained")
            
        return (
            f"Changes in Volatility SVAR (Rigobon 2003)\n"
            f"Regime relative variances (Lambda): {np.round(self.lambda_diag, 3)}\n\n"
            f"VAR Summary:\n{self.var_result.summary()}"
        )

    def get_model_metrics(self):
        if self.structural_impact is None:
            raise ValueError("Model not trained")
        return {
            "nobs": int(self.var_result.nobs),
            "lambda_diag": self.lambda_diag.tolist(),
        }

    def run(self, data):
        self.fit(data)
        return self.get_model_metrics()
