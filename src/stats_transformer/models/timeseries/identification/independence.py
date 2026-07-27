import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.api import VAR
from stats_transformer.models.base import ModelBase
from stats_transformer.models.timeseries.identification.alignment import align_to_cholesky

class IndependenceSVARModel(ModelBase):
    """
    Data-driven SVAR identification via Distance Covariance (Matteson and Tsay 2017).
    Minimizes the distance covariance between the recovered structural shocks.
    Uses the exact O(T^2) calculation for empirical distance covariance.
    """
    def __init__(self, target_variables=None, maxlags=1, n_starts=5, **kwargs):
        target = target_variables[0] if target_variables else "dummy"
        indep = target_variables[1:] if target_variables and len(target_variables) > 1 else ["dummy"]
        super().__init__(target=target, independent_variables=indep, **kwargs)
        
        self.target_variables = target_variables or []
        self.maxlags = maxlags
        self.n_starts = n_starts
        
        self.var_result = None
        self.structural_impact = None
        self.optimization_status = None
        
    def build_model(self):
        if getattr(self, 'df_clean', None) is None:
            raise ValueError("No cleaned data available")
            
        self.y = self.df_clean[self.target_variables]
        var_model = VAR(self.y)
        self.var_result = var_model.fit(maxlags=self.maxlags)
        self.model = self.var_result
        
        self._identify_shocks()
        return self.var_result
        
    def _distance_covariance(self, x, y):
        """
        Computes the empirical distance covariance between two 1D vectors x and y.
        Exact O(T^2) method to match R svars.
        """
        T = len(x)
        
        # Pairwise distance matrices
        a = np.abs(x[:, None] - x[None, :])
        b = np.abs(y[:, None] - y[None, :])
        
        # Double centering
        A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
        B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
        
        # Empirical distance covariance squared
        dcov2 = np.sum(A * B) / (T ** 2)
        return dcov2 if dcov2 > 0 else 0.0

    def _objective(self, theta, p_chol, resid):
        """
        Objective function to minimize the sum of pairwise distance covariances 
        between the recovered structural shocks.
        theta are the generalized Euler angles for the orthogonal matrix.
        """
        K = resid.shape[1]
        
        # Construct orthogonal matrix from angles
        # We handle K=2 or K=3 explicitly for simplicity, generalizing requires K(K-1)/2 angles
        # Using a general Givens rotation builder
        Q = np.eye(K)
        idx = 0
        for i in range(K):
            for j in range(i + 1, K):
                G = np.eye(K)
                G[i, i] = np.cos(theta[idx])
                G[j, j] = np.cos(theta[idx])
                G[i, j] = -np.sin(theta[idx])
                G[j, i] = np.sin(theta[idx])
                Q = Q @ G
                idx += 1
                
        # Candidate impact
        B = p_chol @ Q
        
        # Structural shocks e_t = B^{-1} u_t
        B_inv = np.linalg.inv(B)
        e_t = (B_inv @ resid.T).T # (T, K)
        
        # Sum of pairwise distance covariances
        dcov_sum = 0.0
        for i in range(K):
            for j in range(i + 1, K):
                dcov_sum += self._distance_covariance(e_t[:, i], e_t[:, j])
                
        return dcov_sum

    def _identify_shocks(self):
        resid = self.var_result.resid.values if isinstance(self.var_result.resid, pd.DataFrame) else self.var_result.resid
        K = resid.shape[1]
        n_angles = int(K * (K - 1) / 2)
        
        Sigma = np.cov(resid.T)
        p_chol = np.linalg.cholesky(Sigma)
        
        best_obj = np.inf
        best_theta = None
        best_res = None
        
        np.random.seed(42)
        
        # Multi-start optimization
        for _ in range(self.n_starts):
            theta_init = np.random.uniform(0, 2 * np.pi, size=n_angles)
            
            res = minimize(
                self._objective, 
                theta_init, 
                args=(p_chol, resid),
                method='L-BFGS-B',
                bounds=[(0, 2 * np.pi)] * n_angles
            )
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_theta = res.x
                best_res = res
                
        # Construct final best Q
        Q = np.eye(K)
        idx = 0
        for i in range(K):
            for j in range(i + 1, K):
                G = np.eye(K)
                G[i, i] = np.cos(best_theta[idx])
                G[j, j] = np.cos(best_theta[idx])
                G[i, j] = -np.sin(best_theta[idx])
                G[j, i] = np.sin(best_theta[idx])
                Q = Q @ G
                idx += 1
                
        B = p_chol @ Q
        
        self.structural_impact = align_to_cholesky(B, p_chol)
        self.optimization_status = best_res

    def get_summary(self):
        if self.structural_impact is None:
            raise ValueError("Model not trained")
            
        opt_success = self.optimization_status.success
        
        return (
            f"Distance Covariance SVAR (Matteson and Tsay 2017)\n"
            f"Optimization Success: {opt_success}\n\n"
            f"VAR Summary:\n{self.var_result.summary()}"
        )

    def get_model_metrics(self):
        if self.structural_impact is None:
            raise ValueError("Model not trained")
        return {
            "nobs": int(self.var_result.nobs),
            "opt_success": bool(self.optimization_status.success),
        }

    def run(self, data):
        self.fit(data)
        return self.get_model_metrics()
