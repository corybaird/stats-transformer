import numpy as np
import pandas as pd

class TimeSeriesDecompositions:

    def __init__(self, var_result, B_0=None):
        self.var_result = var_result
        self.k = var_result.neqs
        self.p = var_result.k_ar
        sigma_u = var_result.sigma_u
        if type(sigma_u) == pd.DataFrame:
            sigma_u = sigma_u.values
        self.sigma_u = sigma_u
        self.B_0 = B_0 if B_0 is not None else np.linalg.cholesky(self.sigma_u)

    def compute_fevd(self, steps=20):
        ma_coefs = self.var_result.ma_rep(maxn=steps)
        k = self.k
        fevd = np.zeros((steps, k, k))
        mse = np.zeros((steps, k))
        structural_ma = np.zeros((steps, k, k))
        for h in range(steps):
            structural_ma[h] = ma_coefs[h] @ self.B_0
        for h in range(steps):
            for i in range(k):
                for j in range(k):
                    variance_contrib = np.sum([structural_ma[tau, i, j]**2 for tau in range(h + 1)])
                    fevd[h, i, j] = variance_contrib
                mse[h, i] = np.sum(fevd[h, i, :])
                if mse[h, i] > 0:
                    fevd[h, i, :] = fevd[h, i, :] / mse[h, i]
        return fevd

    def compute_hd(self):
        residuals = self.var_result.resid
        if type(residuals) == pd.DataFrame:
            residuals = residuals.values
        t_obs = len(residuals)
        k = self.k
        inv_B0 = np.linalg.inv(self.B_0)
        structural_shocks = (inv_B0 @ residuals.T).T
        ma_coefs = self.var_result.ma_rep(maxn=t_obs)
        hd = np.zeros((t_obs, k, k))
        for t in range(t_obs):
            for j in range(k):
                for tau in range(t + 1):
                    impact = ma_coefs[t - tau] @ self.B_0[:, j]
                    hd[t, :, j] += impact * structural_shocks[tau, j]
        return hd, structural_shocks

    def run(self, steps=20):
        fevd = self.compute_fevd(steps=steps)
        hd, shocks = self.compute_hd()
        return {"fevd": fevd, "hd": hd, "shocks": shocks}
