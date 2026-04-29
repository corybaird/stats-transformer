import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import f
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson, jarque_bera

class RegressionDiagnostics:
    def __init__(self, model_fit):
        self.model_fit = model_fit

    def breusch_pagan_test(self):
        bp_test = het_breuschpagan(self.model_fit.resid, self.model_fit.model.exog)
        return {"statistic": bp_test[0], "p_value": bp_test[1]}

    def jarque_bera_test(self):
        jb_test = jarque_bera(self.model_fit.resid)
        return {"statistic": jb_test[0], "p_value": jb_test[1]}

    def white_test(self):
        white_test = het_white(self.model_fit.resid, self.model_fit.model.exog)
        return {"statistic": white_test[0], "p_value": white_test[1]}

    def durbin_watson_test(self):
        dw_test = durbin_watson(self.model_fit.resid)
        return {"statistic": dw_test}

    def chow_test(self, data, formula, split_point):
        data1 = data.iloc[:split_point]
        data2 = data.iloc[split_point:]
        ols1 = smf.ols(formula=formula, data=data1).fit()
        ols2 = smf.ols(formula=formula, data=data2).fit()
        ols_full = smf.ols(formula=formula, data=data).fit()
        rss1 = ols1.ssr
        rss2 = ols2.ssr
        rss_full = ols_full.ssr
        k = ols_full.df_model + 1
        n1 = len(data1)
        n2 = len(data2)
        chow_stat = ((rss_full - (rss1 + rss2)) / k) / ((rss1 + rss2) / (n1 + n2 - 2 * k))
        p_value = 1 - f.cdf(chow_stat, k, n1 + n2 - 2 * k)
        return {"statistic": chow_stat, "p_value": p_value}

    def run(self):
        pass
