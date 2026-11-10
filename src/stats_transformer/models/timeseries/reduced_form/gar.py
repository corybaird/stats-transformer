import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import least_squares
from scipy.special import gammaln
from scipy.stats import t
from statsmodels.regression.quantile_regression import QuantReg
from stats_transformer.models.base import ModelBase


def _skewed_t_ppf(u, mu, sigma, lam, nu):
    c = np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2)))
    a = 4 * lam * c * (nu - 2) / (nu - 1)
    b = np.sqrt(1 + 3 * lam**2 - a**2)
    left = (1 - lam) * t.ppf(u / (1 - lam), nu)
    right = (1 + lam) * t.isf((1 - u) / (1 + lam), nu)
    z = (np.where(u < (1 - lam) / 2, left, right) * np.sqrt(1 - 2 / nu) - a) / b
    return mu + sigma * z


class GaRModel(ModelBase):

    def __init__(self, target=None, independent_variables=None, quantiles=(0.05, 0.25, 0.50, 0.75, 0.95), horizon=1, date_column=None, **kwargs):
        super().__init__(target=target, independent_variables=independent_variables, **kwargs)
        model_params = self.params.get("model", {})
        self.quantiles = tuple(sorted(model_params.get("quantiles", quantiles)))
        self.horizon = model_params.get("horizon", horizon)
        self.date_column = date_column
        self.time_column = date_column
        self.quantile_results = {}
        self.skew_t_params = None
        self.X = None
        self.y = None

    def _require_fit(self):
        if not self.quantile_results or self.X is None or self.y is None:
            raise ValueError("Model must be fitted before calling this method")

    def _quantile_residuals(self, p, taus, observed):
        theoretical = _skewed_t_ppf(taus, p[0], np.exp(p[1]), np.clip(p[2], -0.99, 0.99), p[3])
        return np.nan_to_num(theoretical - observed, nan=1e12, posinf=1e12, neginf=-1e12)

    def _fit_skewed_t_rows(self, qf):
        taus = np.array(self.quantiles)
        observed = qf.to_numpy(dtype=float)
        rows = []
        for i in range(observed.shape[0]):
            row = observed[i]
            mid = row[len(row) // 2]
            spread = max((row[-1] - row[0]) / 3.29, 1e-6)
            res = least_squares(
                self._quantile_residuals,
                np.array([mid, np.log(spread), 0.0, 4.0]),
                args=(taus, row),
                bounds=([-np.inf, np.log(1e-8), -0.99, 2.0 + 1e-6], [np.inf, 20.0, 0.99, 500.0]),
            )
            rows.append({"mu": res.x[0], "sigma": np.exp(res.x[1]), "alpha": res.x[2], "nu": res.x[3]})
        return pd.DataFrame(rows, index=qf.index)

    def _rearrange_quantiles(self, qf):
        return pd.DataFrame(np.sort(qf.to_numpy(dtype=float), axis=1), index=qf.index, columns=qf.columns)

    def _skew_t_params_for_row(self):
        self._require_fit()
        if self.skew_t_params is None:
            raise ValueError("Skewed-t density not fitted")
        return self.skew_t_params

    def build_model(self):
        if self.df_clean is None:
            raise ValueError("No cleaned data available")
        work = self.df_clean
        if self.date_column and self.date_column in work.columns:
            work = work.sort_values(self.date_column)
        y_h = work[self.target].shift(-self.horizon)
        x = sm.add_constant(work[self.independent_variables], has_constant="add")
        valid = y_h.notna() & x.notna().all(axis=1)
        self.y = y_h[valid]
        self.X = x.loc[valid]
        if len(self.y) < max(10, len(self.X.columns) * 2):
            raise ValueError("Not enough observations to fit quantile regressions")
        self.quantile_results = {tau: QuantReg(self.y, self.X).fit(q=tau, max_iter=10000) for tau in self.quantiles}
        middle = self.quantiles[len(self.quantiles) // 2]
        self.model = self.quantile_results[middle]
        self.skew_t_params = self._fit_skewed_t_rows(self.predict_quantiles())

    def predict_quantiles(self, new_data=None):
        self._require_fit()
        if new_data is None:
            x = self.X
        else:
            df = new_data.copy()
            if self.date_column and self.date_column in df.columns:
                df = df.sort_values(self.date_column)
            x = sm.add_constant(df[self.independent_variables], has_constant="add").reindex(columns=self.X.columns)
            if x.isna().any().any():
                raise ValueError("new_data contains NaN values in the model columns")
        qf = pd.DataFrame(
            {f"q{int(round(t * 100)):02d}": res.predict(x) for t, res in self.quantile_results.items()},
            index=x.index,
        )
        return self._rearrange_quantiles(qf)

    def compute_gar(self, alpha=0.05):
        p = self._skew_t_params_for_row()
        q = _skewed_t_ppf(
            alpha,
            p["mu"].to_numpy(),
            p["sigma"].to_numpy(),
            p["alpha"].to_numpy(),
            p["nu"].to_numpy(),
        )
        return pd.Series(np.asarray(q, dtype=float), index=p.index, name=f"gar_{alpha}")

    def compute_expected_shortfall(self, alpha=0.05):
        p = self._skew_t_params_for_row()
        grid = np.linspace(alpha / 200.0, alpha - alpha / 200.0, 100)
        tail = _skewed_t_ppf(
            grid[None, :],
            p["mu"].to_numpy()[:, None],
            p["sigma"].to_numpy()[:, None],
            p["alpha"].to_numpy()[:, None],
            p["nu"].to_numpy()[:, None],
        )
        return pd.Series(np.nanmean(tail, axis=1), index=p.index, name=f"es_{alpha}")

    def get_model_metrics(self):
        metrics = {"nobs": int(len(self.y)) if self.y is not None else 0, "horizon": self.horizon, "quantiles": list(self.quantiles)}
        if self.skew_t_params is not None and len(self.skew_t_params):
            last = self.skew_t_params.iloc[-1]
            metrics.update({
                "skewed_t_location": float(last["mu"]),
                "skewed_t_scale": float(last["sigma"]),
                "skewed_t_skewness": float(last["alpha"]),
                "skewed_t_df": float(last["nu"]),
            })
        return metrics

    def get_summary(self):
        self._require_fit()
        lines = [f"Growth-at-Risk model: QuantReg at {list(self.quantiles)} for y(t+{self.horizon}) on {self.independent_variables}", f"nobs={len(self.y)}"]
        if self.skew_t_params is not None and len(self.skew_t_params):
            last = self.skew_t_params.iloc[-1]
            lines.append(f"skewed-t fit (last period): mu={last['mu']:.4f}, sigma={last['sigma']:.4f}, alpha={last['alpha']:.4f}, nu={last['nu']:.4f}")
        return "\n".join(lines)
