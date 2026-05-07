import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from stats_transformer.models.regression.regression import RegressionModel

class IV2SLSModel(RegressionModel):
    def __init__(self, target=None, independent_variables=None, instruments=None, endogenous=None, cov_type="robust", **kwargs):
        super().__init__(target=target, independent_variables=independent_variables, **kwargs)
        self.instruments = instruments or []
        self.endogenous = endogenous or []
        self.cov_type = cov_type

    def _get_required_columns(self):
        columns = super()._get_required_columns()
        for col in self.endogenous + self.instruments:
            if col not in columns:
                columns.append(col)
        return columns
    def build_model(self, drop_na=True):
        if self.X is None or self.y is None:
            # We skip RegressionModel's split_xy because linearmodels needs endogenous and instruments separate
            pass
            
        x_numeric = self.df_clean.select_dtypes(include=[np.number])
        if np.isnan(x_numeric.values).any() or np.isinf(x_numeric.values).any():
            if drop_na:
                self.df_clean = self.df_clean.replace([np.inf, -np.inf], np.nan).dropna()
            else:
                raise ValueError("Data contains inf or nans in IV2SLS")

        self.y = self.df_clean[self.target]
        self.X_exog = sm.add_constant(self.df_clean[self.independent_variables]) if self.independent_variables else sm.add_constant(pd.DataFrame(index=self.df_clean.index))
        self.X_endog = self.df_clean[self.endogenous] if self.endogenous else None
        self.Z = self.df_clean[self.instruments] if self.instruments else None

        self.iv_model_spec = IV2SLS(
            dependent=self.y,
            exog=self.X_exog,
            endog=self.X_endog,
            instruments=self.Z
        )
        self.model = self.iv_model_spec.fit(cov_type=self.cov_type)
        return self.model

    def get_summary(self):
        if self.model is None:
            raise ValueError("Model not trained")
        return str(self.model.summary)

    def get_model_metrics(self):
        if self.model is None:
            raise ValueError("Model not trained")
        return {
            "r_squared": self.model.rsquared,
            "f_statistic": self.model.f_statistic.stat,
            "f_pvalue": self.model.f_statistic.pval,
            "wu_hausman_stat": self.model.wu_hausman().stat if hasattr(self.model, 'wu_hausman') else None,
            "num_observations": self.model.nobs,
        }
