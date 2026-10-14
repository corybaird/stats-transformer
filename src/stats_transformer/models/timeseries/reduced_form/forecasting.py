import numpy as np
import pandas as pd
from scipy import stats

def compute_ma_rep(params, k_ar, k_trend, neqs, steps):
    """
    Computes the Wold MA representation matrices (Phi_s) of a VAR.
    Phi_0 = I
    Phi_s = sum_{j=1}^s Phi_{s-j} A_j  (where A_j are the VAR coefficient matrices)

    Shared by VARForecaster and RestrictedVARResults.ma_rep so both slice the
    lag coefficients identically. params is (k_trend + p*K, K), with the
    deterministic rows first.
    """
    lag_params = np.asarray(params)[k_trend:] if k_trend > 0 else np.asarray(params)

    # A_j is (K, K); in lag_params the blocks are stacked vertically by lag.
    A = [lag_params[i * neqs:(i + 1) * neqs].T for i in range(k_ar)]

    phis = np.zeros((steps, neqs, neqs))
    phis[0] = np.eye(neqs)
    for s in range(1, steps):
        phi_s = np.zeros((neqs, neqs))
        for j in range(1, min(s + 1, k_ar + 1)):
            phi_s += phis[s - j] @ A[j - 1]
        phis[s] = phi_s
    return phis

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
        return compute_ma_rep(self.params, k_ar=self.k_ar, k_trend=self.k_trend, neqs=len(self.names), steps=steps)

    @staticmethod
    def forecast_direct(
        data: pd.DataFrame,
        target_variables: list[str],
        lags: int = 1,
        steps: int = 10,
        trend: str = "c",
    ) -> pd.DataFrame:
        """Direct multi-step forecasting for each variable and horizon.

        For each horizon h in 1..steps and each variable y_i:
            y_{i, t+h} = alpha_h + sum_{j=1}^K sum_{l=0}^{p-1} beta_{h,j,l} y_{j, t-l} + e_{i, t+h}
        Returns DataFrame of point forecasts for t+1 .. t+steps indexed by horizon.
        """
        import statsmodels.api as sm

        df = data[target_variables].dropna().copy()
        T, K = df.shape
        if T <= lags + steps:
            raise ValueError(f"Data length ({T}) must exceed lags ({lags}) + steps ({steps}).")

        # Build design matrix of contemporaneous and lagged values: [y_t, y_{t-1}, ..., y_{t-p+1}]
        X_cols = []
        X_df = pd.DataFrame(index=df.index)
        for p in range(lags):
            for col in target_variables:
                name = f"{col}_lag{p}" if p > 0 else col
                X_df[name] = df[col].shift(p)
                X_cols.append(name)

        if trend == "c":
            X_with_const = sm.add_constant(X_df[X_cols])
        else:
            X_with_const = X_df[X_cols]

        latest_X = X_with_const.iloc[-1:]

        forecasts = np.zeros((steps, K))
        for h in range(1, steps + 1):
            for i, target in enumerate(target_variables):
                y_h = df[target].shift(-h)
                valid = y_h.notna() & X_with_const.notna().all(axis=1)
                model = sm.OLS(y_h[valid], X_with_const[valid]).fit()
                pred = model.predict(latest_X)
                forecasts[h - 1, i] = pred.iloc[0]

        return pd.DataFrame(forecasts, index=range(1, steps + 1), columns=target_variables)
