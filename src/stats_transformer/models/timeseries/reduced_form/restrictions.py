import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.vector_ar.var_model import VAR, VARResultsWrapper
from stats_transformer.models.timeseries.reduced_form.forecasting import compute_ma_rep

class RestrictedVARResults:
    """
    Mock VARResults wrapper to maintain compatibility with downstream reporting.
    """
    def __init__(self, params, sigma_u, resid, endog, exog, k_ar, names, exog_names=None, trend="c"):
        self.params = params
        self.sigma_u = sigma_u
        self.resid = resid
        self.endog = endog
        self.exog = exog
        self.k_ar = k_ar
        self.names = names
        self.trend = trend

        # Real regressor names are built by RestrictedVAR.fit and passed in.
        # Falling back to placeholders would make k_trend wrong, which silently
        # misaligns every lag-coefficient slice downstream.
        self._exog_names = list(exog_names) if exog_names is not None else [f"x{i}" for i in range(exog.shape[1])]

        # Additional attributes required by downstream
        self.k_trend = 1 if "const" in self._exog_names else 0
        self.nobs = len(endog)
        self.neqs = len(names)

    @property
    def exog_names(self):
        return self._exog_names

    def ma_rep(self, maxn=10):
        return compute_ma_rep(self.params, k_ar=self.k_ar, k_trend=self.k_trend, neqs=self.neqs, steps=maxn + 1)

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
            names=names,
            exog_names=exog_names,
            trend=self.trend
        )
