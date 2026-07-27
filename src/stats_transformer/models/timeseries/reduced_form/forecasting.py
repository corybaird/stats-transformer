import numpy as np
import pandas as pd
from scipy import stats

class VARForecaster:
    """
    Forecasting engine for VAR models.
    Supports point forecasts and analytic confidence intervals.
    """
    def __init__(self, model_result):
        """
        model_result: Can be statsmodels VARResults or our custom RestrictedVARResults.
        Requires attributes: params, sigma_u, k_ar, names, k_trend.
        """
        self.res = model_result
        self.k_ar = model_result.k_ar
        self.names = model_result.names
        self.params = np.asarray(model_result.params)
        self.sigma_u = np.asarray(model_result.sigma_u)
        self.k_trend = getattr(model_result, "k_trend", 1) # Assumes constant by default if missing

    def forecast(self, y, steps=10, alpha=0.05):
        """
        y: numpy array of shape (obs, neqs), the historical data to forecast from.
           Must have at least k_ar observations.
        steps: number of steps ahead to forecast.
        alpha: significance level for the confidence intervals.
        
        Returns:
            point_forecast: (steps, neqs)
            lower: (steps, neqs)
            upper: (steps, neqs)
        """
        if len(y) < self.k_ar:
            raise ValueError(f"Need at least {self.k_ar} observations to forecast.")
            
        y = np.asarray(y)
        neqs = len(self.names)
        
        # We need the last p observations
        y_history = y[-self.k_ar:].copy()
        
        point_forecast = np.zeros((steps, neqs))
        
        # Extract intercept (if any) and lag matrices
        # params shape is (k_trend + neqs * k_ar, neqs)
        if self.k_trend > 0:
            intercept = self.params[0]
            lag_params = self.params[self.k_trend:]
        else:
            intercept = np.zeros(neqs)
            lag_params = self.params
            
        # lag_params shape is (neqs * k_ar, neqs). 
        # The rows are L1.y1, L1.y2..., L2.y1, L2.y2...
        
        # 1. Point Forecasts
        for h in range(steps):
            # Construct the regressor row for the next step
            # It needs [y_{t}, y_{t-1}, ..., y_{t-p+1}] flattened
            # y_history is ordered [y_{t-p+1}, ..., y_t]
            # So we reverse it to [y_t, y_{t-1}, ...]
            lagged_y = y_history[::-1].flatten()
            
            # Predict
            pred = intercept + lagged_y @ lag_params
            point_forecast[h] = pred
            
            # Update history
            y_history = np.vstack([y_history[1:], pred])
            
        # 2. Analytic Intervals (MSE of forecasts)
        # We need the MA representation of the VAR model (Phi matrices)
        phis = self._compute_ma_rep(steps)
        
        # MSE matrices for each step h
        # Sigma(h) = sum_{i=0}^{h-1} Phi_i Sigma_u Phi_i'
        sigma_h = np.zeros((steps, neqs, neqs))
        current_sigma = np.zeros((neqs, neqs))
        
        for h in range(steps):
            phi = phis[h]
            current_sigma += phi @ self.sigma_u @ phi.T
            sigma_h[h] = current_sigma
            
        # Standard errors for each variable and horizon
        se = np.sqrt(np.diagonal(sigma_h, axis1=1, axis2=2))
        
        # Confidence intervals
        z = stats.norm.ppf(1 - alpha / 2)
        lower = point_forecast - z * se
        upper = point_forecast + z * se
        
        return point_forecast, lower, upper
        
    def _compute_ma_rep(self, steps):
        """
        Computes the MA representation matrices (Phi_i) up to `steps`.
        Phi_0 = I
        Phi_s = sum_{j=1}^s Phi_{s-j} A_j  (where A_j are the VAR coefficient matrices)
        """
        neqs = len(self.names)
        
        if self.k_trend > 0:
            lag_params = self.params[self.k_trend:]
        else:
            lag_params = self.params
            
        # Extract A_j matrices. lag_params is (p*K, K)
        # A_j is (K, K), but in lag_params it's block-wise stacked vertically
        A = []
        for i in range(self.k_ar):
            A.append(lag_params[i*neqs:(i+1)*neqs].T)
            
        phis = np.zeros((steps, neqs, neqs))
        phis[0] = np.eye(neqs)
        
        for s in range(1, steps):
            phi_s = np.zeros((neqs, neqs))
            for j in range(1, min(s + 1, self.k_ar + 1)):
                phi_s += phis[s - j] @ A[j - 1]
            phis[s] = phi_s
            
        return phis
