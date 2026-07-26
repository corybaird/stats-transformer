import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.vector_ar.var_model import VAR, VARResultsWrapper

class RestrictedVARResults:
    """
    Mock VARResults wrapper to maintain compatibility with downstream reporting.
    """
    def __init__(self, params, sigma_u, resid, endog, exog, k_ar, names):
        self.params = params
        self.sigma_u = sigma_u
        self.resid = resid
        self.endog = endog
        self.exog = exog
        self.k_ar = k_ar
        self.names = names
        
        # Additional attributes required by downstream
        self.k_trend = 1 if "const" in self.exog_names else 0
        self.nobs = len(endog)
        self.neqs = len(names)

    @property
    def exog_names(self):
        # We assume standard naming: const, L1.y1, L1.y2 ...
        cols = []
        if self.exog.shape[1] == self.params.shape[0]:
            # This is a very rough mock, we might need real names
            cols = [f"x{i}" for i in range(self.exog.shape[1])]
        return cols

class RestrictedVAR:
    """
    Estimates a VAR model subject to zero restrictions on coefficients.
    Uses equation-by-equation OLS on the allowed regressors.
    """
    def __init__(self, endog, mask, maxlags=1, trend="c"):
        """
        endog: DataFrame of endogenous variables
        mask: boolean array (or 1/0) matching the shape of unrestricted params matrix (num_regressors x num_equations)
              If trend="c", the first row is the constant.
        """
        self.endog = endog
        self.mask = np.asarray(mask, dtype=bool)
        self.maxlags = maxlags
        self.trend = trend
        
    def fit(self):
        y = np.asarray(self.endog)
        names = list(self.endog.columns)
        nobs_raw, neqs = y.shape
        
        # 1. Create lagged matrix (exog)
        # We lose first `maxlags` observations
        y_lagged = lagmat(y, maxlag=self.maxlags, trim="both", original="ex")
        
        # The dependent variable is y[maxlags:]
        y_dep = y[self.maxlags:]
        
        # Add trend if necessary
        if self.trend == "c":
            x = sm.add_constant(y_lagged, prepend=True)
            exog_names = ["const"]
        else:
            x = y_lagged
            exog_names = []
            
        for lag in range(1, self.maxlags + 1):
            for name in names:
                exog_names.append(f"L{lag}.{name}")
                
        # 2. Equation-by-equation OLS using mask
        num_regressors = x.shape[1]
        if self.mask.shape != (num_regressors, neqs):
            raise ValueError(f"Mask shape {self.mask.shape} does not match (num_regressors={num_regressors}, num_equations={neqs})")
            
        params = np.zeros((num_regressors, neqs))
        resid = np.zeros_like(y_dep)
        
        for i in range(neqs):
            # Find which regressors are allowed (True)
            allowed = self.mask[:, i]
            x_i = x[:, allowed]
            y_i = y_dep[:, i]
            
            # Fit OLS
            if x_i.shape[1] > 0:
                res = sm.OLS(y_i, x_i).fit()
                params[allowed, i] = res.params
                resid[:, i] = res.resid
            else:
                resid[:, i] = y_i
                
        # 3. Compute residual covariance
        sigma_u = np.dot(resid.T, resid) / (len(y_dep) - num_regressors) # simplified df correction
        
        return RestrictedVARResults(
            params=params, 
            sigma_u=sigma_u, 
            resid=resid, 
            endog=y_dep, 
            exog=x, 
            k_ar=self.maxlags, 
            names=names
        )
