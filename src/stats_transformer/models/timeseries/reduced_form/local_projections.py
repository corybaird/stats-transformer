import pandas as pd
import numpy as np
import statsmodels.api as sm
from stats_transformer.models.base import ModelBase

class LocalProjectionsModel(ModelBase):
    def __init__(
        self,
        target,
        shock_var,
        horizon=8,
        controls=None,
        date_column=None,
        cumulative=False,
        lags=0,
        cov_type="HC3",
        cov_kwds=None,
        **kwargs
    ):
        if cumulative not in (False, None, True, "cumulative", "baseline", "diff", "change"):
            raise ValueError(
                f"cumulative must be True, None, False, 'cumulative', 'baseline', 'diff' or 'change', "
                f"got {cumulative!r}"
            )
        super().__init__(target=target, independent_variables=[shock_var] + (controls or []), **kwargs)
        self.shock_var = shock_var
        self.horizon = horizon
        self.controls = controls or []
        self.date_column = date_column
        self.time_column = date_column
        self.cumulative = cumulative
        self.lags = lags
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds
        self.irf_results = []

    def _build_design_matrix(self):
        """Constructs regressor DataFrame including shock, controls, and optional lags."""
        x = self.df_clean[self.independent_variables].copy()
        if self.lags > 0:
            for lag in range(1, self.lags + 1):
                # Lag of target
                x[f"{self.target}_lag{lag}"] = self.df_clean[self.target].shift(lag)
                # Lag of shock
                x[f"{self.shock_var}_lag{lag}"] = self.df_clean[self.shock_var].shift(lag)
                # Lags of controls
                for c in self.controls:
                    x[f"{c}_lag{lag}"] = self.df_clean[c].shift(lag)
        return sm.add_constant(x)

    def _target_at_horizon(self, y, h):
        """Outcome at horizon h: levels, cumulative (vs pre-shock baseline y_{t-1}),
        or diff/change (vs contemporaneous y_t)."""
        if self.cumulative is True or self.cumulative in ("cumulative", "baseline"):
            return y.shift(-h) - y.shift(1)
        if self.cumulative in ("diff", "change"):
            return y.shift(-h) - y
        return y.shift(-h)

    def build_model(self):
        self.irf_results = []
        if self.date_column and self.date_column in self.df_clean.columns:
            self.df_clean = self.df_clean.sort_values(self.date_column)
        y = self.df_clean[self.target]
        x = self._build_design_matrix()
        
        for h in range(self.horizon + 1):
            y_h = self._target_at_horizon(y, h)

            valid_idx = y_h.notna() & x.notna().all(axis=1)
            y_valid = y_h[valid_idx]
            x_valid = x.loc[valid_idx]
            
            # Resolve covariance type
            cov_type_norm = self.cov_type.upper() if isinstance(self.cov_type, str) else str(self.cov_type)
            if cov_type_norm in ("HAC", "NW", "NEWEY_WEST", "NEWEYWEST"):
                # Newey-West: bandwidth defaults to h + 1; user-supplied maxlags applies to every horizon
                cov_type_actual = "HAC"
                cov_kwds_h = {"maxlags": h + 1, **(self.cov_kwds or {})}
            elif cov_type_norm == "STANDARD":
                cov_type_actual = "nonrobust"
                cov_kwds_h = None
            else:
                cov_type_actual = self.cov_type
                cov_kwds_h = dict(self.cov_kwds) if self.cov_kwds else None

            model = sm.OLS(y_valid, x_valid)
            if cov_kwds_h is not None:
                res = model.fit(cov_type=cov_type_actual, cov_kwds=cov_kwds_h)
            else:
                res = model.fit(cov_type=cov_type_actual)
            
            self.irf_results.append({
                'horizon': h,
                'effect': res.params[self.shock_var],
                'stderr': res.bse[self.shock_var],
                'lower_ci': res.conf_int().loc[self.shock_var, 0],
                'upper_ci': res.conf_int().loc[self.shock_var, 1],
                'pvalue': res.pvalues[self.shock_var]
            })

    def compute_irf(self):
        if not self.irf_results:
            self.build_model()
        return pd.DataFrame(self.irf_results)

    def compute_vd(self):
        vd_results = []
        y = self.df_clean[self.target]
        x_full = self._build_design_matrix()
        
        # Baseline model without the contemporaneous shock_var
        x_controls = x_full.drop(columns=[self.shock_var], errors="ignore")
        
        for h in range(self.horizon + 1):
            y_h = self._target_at_horizon(y, h)

            valid_idx = y_h.notna() & x_full.notna().all(axis=1)
            y_valid = y_h[valid_idx]
            
            # Baseline model (controls + lags only)
            if not x_controls.empty and len(x_controls.columns) > 1:
                res_base = sm.OLS(y_valid, x_controls.loc[valid_idx]).fit()
                r2_base = res_base.rsquared
            else:
                r2_base = 0
                
            # Full model (controls + lags + contemporaneous shock)
            res_full = sm.OLS(y_valid, x_full.loc[valid_idx]).fit()
            r2_full = res_full.rsquared
            
            # VD is the marginal R-squared of the shock
            vd = r2_full - r2_base
            
            vd_results.append({
                'horizon': h,
                'variance_explained': max(0, vd)  # Avoid tiny negative numerical errors
            })
            
        return pd.DataFrame(vd_results)

    def summary_df(self):
        return self.compute_irf()

    def get_summary(self):
        return self.summary_df().to_string()

    def get_model_metrics(self):
        irf = self.compute_irf()
        return {
            "horizon": self.horizon,
            "shock_var": self.shock_var,
            "irf_h0": float(irf["effect"].iloc[0]) if not irf.empty else None,
            "irf_h_max": float(irf["effect"].iloc[-1]) if not irf.empty else None
        }

    def run(self, data):
        self.fit(data)
        return self.get_model_metadata()
