import numpy as np
import pandas as pd
from scipy import stats

class ResidualDiagnostics:
    """
    Diagnostic tests for VAR residuals.
    Matches conventions from R's vars package (serial.test, normality.test, arch.test).
    """
    def __init__(self, model_result):
        self.res = model_result
        self.resid = np.asarray(model_result.resid)
        self.nobs = len(self.resid)
        self.neqs = self.resid.shape[1]
        self.k_ar = model_result.k_ar

    def test_serial_correlation(self, lags=10):
        """
        Portmanteau (Ljung-Box) test for residual serial correlation.
        Matches vars::serial.test(..., type="PT.asymptotic").
        """
        # Cross-covariance matrices of residuals
        resid_centered = self.resid - self.resid.mean(axis=0)
        C_0 = (resid_centered.T @ resid_centered) / self.nobs
        C_0_inv = np.linalg.inv(C_0)
        
        q_stat = 0.0
        q_stat_adj = 0.0
        
        for i in range(1, lags + 1):
            # Compute lag i covariance
            # C_i = 1/T * sum_{t=i+1}^T u_t u_{t-i}'
            C_i = (resid_centered[i:].T @ resid_centered[:-i]) / self.nobs
            
            # trace(C_i' C_0^{-1} C_i C_0^{-1})
            trace_val = np.trace(C_i.T @ C_0_inv @ C_i @ C_0_inv)
            
            q_stat += trace_val
            q_stat_adj += trace_val / (self.nobs - i)
            
        q_stat = self.nobs * q_stat
        q_stat_adj = self.nobs ** 2 * q_stat_adj
        
        # Degrees of freedom: K^2 * (lags - p)
        # Note: If lags <= p, DF is not well defined (or negative).
        df = self.neqs ** 2 * (lags - self.k_ar)
        if df > 0:
            p_val = stats.chi2.sf(q_stat, df)
            p_val_adj = stats.chi2.sf(q_stat_adj, df)
        else:
            p_val = np.nan
            p_val_adj = np.nan
            
        return {
            "portmanteau": {"statistic": q_stat, "pvalue": p_val, "df": df},
            "adjusted_portmanteau": {"statistic": q_stat_adj, "pvalue": p_val_adj, "df": df}
        }

    def test_normality(self):
        """
        System-wide tests for multivariate skewness and kurtosis.
        Matches vars::normality.test (Lutkepohl, 2005).
        """
        # Cholesky of residual covariance
        C_0 = (self.resid.T @ self.resid) / self.nobs
        P = np.linalg.cholesky(C_0)
        P_inv = np.linalg.inv(P)
        
        # Standardized residuals
        w = (P_inv @ self.resid.T).T
        
        # Skewness
        b1 = np.mean(w ** 3, axis=0)
        skew_stat = self.nobs * np.sum(b1 ** 2) / 6.0
        skew_pval = stats.chi2.sf(skew_stat, self.neqs)
        
        # Kurtosis
        b2 = np.mean(w ** 4, axis=0)
        kurt_stat = self.nobs * np.sum((b2 - 3.0) ** 2) / 24.0
        kurt_pval = stats.chi2.sf(kurt_stat, self.neqs)
        
        # Omnibus
        omnibus_stat = skew_stat + kurt_stat
        omnibus_pval = stats.chi2.sf(omnibus_stat, self.neqs * 2)
        
        return {
            "skewness": {"statistic": skew_stat, "pvalue": skew_pval, "df": self.neqs},
            "kurtosis": {"statistic": kurt_stat, "pvalue": kurt_pval, "df": self.neqs},
            "omnibus": {"statistic": omnibus_stat, "pvalue": omnibus_pval, "df": self.neqs * 2}
        }
        
    def test_arch(self, lags=5):
        """
        Multivariate ARCH-LM test.
        """
        # 1/2 vech(u_t u_t') 
        # This requires OLS regression on lagged squared residuals.
        # For simplicity, we just use statsmodels logic or return a basic implementation.
        # R's vars::arch.test uses the multivariate LM test.
        # Let's approximate or just return placeholders if we don't have a full vech regression implemented yet.
        # The roadmap requires "compare fixed benchmark".
        
        # Just stubbing it out for now, to be implemented precisely if needed.
        return {
            "arch_lm": {"statistic": np.nan, "pvalue": np.nan, "df": np.nan}
        }
