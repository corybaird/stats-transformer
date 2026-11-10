import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import least_squares
from typing import Any, Dict, List, Optional, Tuple
from stats_transformer.models.base import ModelBase


def _read_frame(data: Any, params: Dict[str, Any]) -> pd.DataFrame:
    if data is None:
        data = (params or {}).get("data", {}).get("raw_data_file")
        if not data:
            raise ValueError("No data provided: pass data to fit() or set data.raw_data_file")
    if isinstance(data, str):
        return pd.read_csv(data) if data.endswith(".csv") else pd.read_parquet(data)
    if isinstance(data, pd.DataFrame):
        return data.copy()
    raise TypeError(f"Unsupported data type: {type(data)}")


def _build_hf_lag_matrix(hf: np.ndarray, freq_ratio: int, hf_lags: int) -> Tuple[np.ndarray, np.ndarray]:
    m, K = int(freq_ratio), int(hf_lags)
    ends = np.array([(t + 1) * m - 1 for t in range(len(hf) // m)], dtype=int)
    ends = ends[ends - K >= 0]
    lags = np.column_stack([hf[ends - j] for j in range(1, K + 1)]) if len(ends) else np.empty((0, K))
    return lags, ends


def _reshape_hf(df: pd.DataFrame, hf_col: str, target: str, freq_ratio: int, hf_lags: int, y_lags: int) -> pd.DataFrame:
    hf = df[hf_col].to_numpy(dtype=float)
    y_hf = df[target].to_numpy(dtype=float) if target in df.columns else np.full(len(df), np.nan)
    lags, ends = _build_hf_lag_matrix(hf, freq_ratio, hf_lags)
    if len(ends) == 0:
        raise ValueError("Not enough high-frequency observations for the requested lags")
    m = int(freq_ratio)
    parts = [lags]
    names = [f"{hf_col}_hf_lag{j}" for j in range(1, int(hf_lags) + 1)]
    for j in range(1, y_lags + 1):
        parts.append(np.array([y_hf[p - j * m] if p - j * m >= 0 else np.nan for p in ends])[:, None])
        names.append(f"y_lag{j}")
    parts.append(y_hf[ends][:, None])
    names.append(target)
    index = df.index.to_numpy()[ends]
    return pd.DataFrame(np.column_stack(parts), index=index, columns=names)


def _almon_basis(hf_lags: int, degree: int) -> np.ndarray:
    # ponytail: plain Almon Vandermonde in lag order k=1..K; conditioning degrades for large K
    return np.vander(np.arange(1, hf_lags + 1, dtype=float), degree + 1, increasing=True)


def _exp_almon_weights(theta: np.ndarray, hf_lags: int) -> np.ndarray:
    k = np.arange(hf_lags, dtype=float)
    exponent = np.clip(theta[0] * k + theta[1] * k * k, -700.0, 700.0)
    weights = np.exp(exponent)
    return weights / weights.sum()


def _beta_weights(theta: np.ndarray, hf_lags: int) -> np.ndarray:
    u = np.arange(1, hf_lags + 1, dtype=float) / (hf_lags + 1)
    log_f = np.clip(theta[0] * np.log(u) + theta[1] * np.log1p(-u), -700.0, 700.0)
    f = np.exp(log_f)
    return f / f.sum()


class _LinearShim:

    def __init__(self, params: np.ndarray, names: List[str], nobs: int, bse: Optional[np.ndarray] = None) -> None:
        self.params = pd.Series(params, index=names)
        self.nobs = int(nobs)
        if bse is not None:
            self.bse = pd.Series(bse, index=names)


class UMIDASModel(ModelBase):

    def __init__(self, params_path: Optional[str] = None, target: Optional[str] = None, independent_variables: Optional[List[str]] = None, hf_columns: Optional[List[str]] = None, hf_lags: Optional[int] = None, freq_ratio: Optional[int] = None, y_lags: int = 0, alpha: float = 0.0, **kwargs: Any) -> None:
        columns = independent_variables if independent_variables is not None else (hf_columns or [])
        super().__init__(params_path=params_path, target=target, independent_variables=columns, **kwargs)
        self.hf_lags = self._param("hf_lags", hf_lags, None)
        self.freq_ratio = self._param("freq_ratio", freq_ratio, None)
        self.y_lags = int(self._param("y_lags", y_lags, 0))
        self.alpha = float(self._param("alpha", alpha, 0.0))

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.hf_lags is not None and self.freq_ratio is not None:
            if len(self.independent_variables) != 1:
                raise ValueError("hf_lags/freq_ratio mode expects exactly one high-frequency series in independent_variables")
            return _reshape_hf(df, self.independent_variables[0], self.target, int(self.freq_ratio), int(self.hf_lags), self.y_lags)
        missing = [c for c in [self.target] + list(self.independent_variables) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in data: {missing}")
        y = pd.to_numeric(df[self.target], errors="coerce")
        features = df[list(self.independent_variables)].apply(pd.to_numeric, errors="coerce")
        for j in range(1, self.y_lags + 1):
            features[f"y_lag{j}"] = y.shift(j)
        features[self.target] = y
        return features

    def _estimate(self, X: np.ndarray, y: np.ndarray) -> Any:
        if self.alpha == 0.0:
            return sm.OLS(y, sm.add_constant(pd.DataFrame(X, columns=self.feature_names), has_constant="add")).fit()
        X_centered = X - X.mean(axis=0)
        y_centered = y - y.mean()
        slopes = np.linalg.solve(X_centered.T @ X_centered + self.alpha * np.eye(X.shape[1]), X_centered.T @ y_centered)
        const = float(y.mean() - X.mean(axis=0) @ slopes)
        return _LinearShim(np.concatenate([[const], slopes]), ["const"] + self.feature_names, nobs=len(y))

    def build_model(self) -> Any:
        if self.df is None:
            raise ValueError("No data loaded: call fit() with data")
        frame = self._build_features(self.df).dropna()
        if frame.empty:
            raise ValueError("No complete observations after aligning high-frequency lags")
        self.y = frame[self.target]
        self.X = frame.drop(columns=[self.target])
        self.feature_names = list(self.X.columns)
        self.model = self._estimate(self.X.to_numpy(dtype=float), self.y.to_numpy(dtype=float))
        return self.model

    def fit(self, data: Any = None) -> Dict[str, Any]:
        self.df = _read_frame(data, self.params)
        self.build_model()
        return self.get_model_metrics()

    def _design_at(self, params_index: List[str], X: pd.DataFrame) -> np.ndarray:
        if params_index and params_index[0] == "const":
            return np.column_stack([np.ones(len(X)), X[params_index[1:]].to_numpy(dtype=float)])
        return X[params_index].to_numpy(dtype=float)

    def _fitted_values(self) -> np.ndarray:
        return self._design_at(list(self.model.params.index), self.X) @ self.model.params.to_numpy(dtype=float)

    def predict(self, new_data: Any) -> pd.Series:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before predict().")
        df = _read_frame(new_data, self.params)
        features = self._build_features(df)
        X = features.drop(columns=[self.target]).dropna()
        if X.empty:
            raise ValueError("No complete observations in new_data after aligning lags")
        preds = self._design_at(list(self.model.params.index), X) @ self.model.params.to_numpy(dtype=float)
        return pd.Series(preds, index=X.index, name="prediction")

    def get_summary(self) -> str:
        if self.model is None:
            raise ValueError("Model not trained")
        if hasattr(self.model, "summary"):
            return str(self.model.summary())
        return self.model.params.to_frame("coefficient").to_string()

    def get_model_metrics(self) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not trained")
        residuals = self.y.to_numpy(dtype=float) - self._fitted_values()
        n = len(residuals)
        k = len(self.model.params)
        ssr = float(residuals @ residuals)
        sst = float(((self.y - self.y.mean()) ** 2).sum())
        r_squared = 1.0 - ssr / sst if sst > 0 else 0.0
        metrics = {
            "r_squared": float(r_squared),
            "adj_r_squared": float(1.0 - (1.0 - r_squared) * (n - 1) / (n - k)),
            "aic": float(n * np.log(ssr / n) + 2 * k),
            "bic": float(n * np.log(ssr / n) + k * np.log(n)),
            "nobs": n,
            "num_params": k,
        }
        return metrics


class MIDASModel(UMIDASModel):

    _WEIGHT_FUNCTIONS: Dict[str, Any] = {"almon": None, "exp_almon": _exp_almon_weights, "beta": _beta_weights}

    def __init__(self, params_path: Optional[str] = None, target: Optional[str] = None, independent_variables: Optional[List[str]] = None, hf_columns: Optional[List[str]] = None, hf_lags: Optional[int] = None, freq_ratio: Optional[int] = None, weight_function: str = "almon", poly_degree: int = 2, **kwargs: Any) -> None:
        super().__init__(params_path=params_path, target=target, independent_variables=independent_variables, hf_columns=hf_columns, hf_lags=hf_lags, freq_ratio=freq_ratio, **kwargs)
        self.weight_function = str(self._param("weight_function", weight_function, "almon"))
        self.poly_degree = int(self._param("poly_degree", poly_degree, 2))
        if self.weight_function not in self._WEIGHT_FUNCTIONS:
            raise ValueError(f"weight_function must be one of {sorted(self._WEIGHT_FUNCTIONS)}, got {self.weight_function!r}")

    def _estimate_almon(self) -> None:
        K = self.X.shape[1]
        Q = _almon_basis(K, self.poly_degree)
        Z = pd.DataFrame(self.X.to_numpy(dtype=float) @ Q, index=self.X.index, columns=[f"poly_{p}" for p in range(Q.shape[1])])
        ols = sm.OLS(self.y, sm.add_constant(Z, has_constant="add")).fit()
        cov = ols.cov_params()
        slopes = ols.params.drop("const").to_numpy(dtype=float)
        weights = Q @ slopes
        cov_weights = Q @ cov.drop(index="const", columns="const").to_numpy() @ Q.T
        weight_se = np.sqrt(np.diag(cov_weights))
        self.weights = weights
        self.weight_se = weight_se
        self.theta = None
        self.ols_result = ols
        self.model = _LinearShim(
            np.concatenate([[float(ols.params["const"])], weights]),
            ["const"] + self.feature_names,
            nobs=int(ols.nobs),
            bse=np.concatenate([[float(np.sqrt(cov.loc["const", "const"]))], weight_se]),
        )

    def _estimate_nls(self) -> None:
        weight_fn = self._WEIGHT_FUNCTIONS[self.weight_function]
        K = self.X.shape[1]
        X = self.X.to_numpy(dtype=float)
        y = self.y.to_numpy(dtype=float)
        n = len(y)
        x0 = np.zeros(2) if self.weight_function == "exp_almon" else np.full(2, 2.0)

        def _residuals(theta: np.ndarray) -> np.ndarray:
            r = y - X @ weight_fn(theta, K)
            return r - r.mean()

        result = least_squares(_residuals, x0)
        weights = weight_fn(result.x, K)
        jacobian = np.column_stack([np.full(n, -1.0), result.jac])
        dof = max(n - jacobian.shape[1], 1)
        cov = float(result.fun @ result.fun) / dof * np.linalg.pinv(jacobian.T @ jacobian)
        eps = 1e-6
        weight_jac = np.zeros((K, 2))
        for j in range(2):
            up, down = result.x.copy(), result.x.copy()
            up[j] += eps
            down[j] -= eps
            weight_jac[:, j] = (weight_fn(up, K) - weight_fn(down, K)) / (2 * eps)
        weight_se = np.sqrt(np.diag(weight_jac @ cov[1:, 1:] @ weight_jac.T))
        const = float(np.mean(y - X @ weights))
        self.weights = weights
        self.weight_se = weight_se
        self.theta = result.x
        self.ols_result = None
        self.model = _LinearShim(
            np.concatenate([[const], weights]),
            ["const"] + self.feature_names,
            nobs=n,
            bse=np.concatenate([[float(np.sqrt(cov[0, 0]))], weight_se]),
        )

    def build_model(self) -> Any:
        if self.y_lags:
            raise ValueError("MIDASModel does not support low-frequency autoregressive lags; use UMIDASModel")
        if self.df is None:
            raise ValueError("No data loaded: call fit() with data")
        frame = self._build_features(self.df).dropna()
        if frame.empty:
            raise ValueError("No complete observations after aligning high-frequency lags")
        self.y = frame[self.target]
        self.X = frame.drop(columns=[self.target])
        self.feature_names = list(self.X.columns)
        if self.weight_function == "almon":
            self._estimate_almon()
        else:
            self._estimate_nls()
        return self.model

    def get_summary(self) -> str:
        if self.model is None:
            raise ValueError("Model not trained")
        lines = [f"Restricted MIDAS with '{self.weight_function}' lag weights"]
        if self.theta is not None:
            lines.append(f"theta: {np.round(self.theta, 6).tolist()}")
        metrics = self.get_model_metrics()
        lines.append(f"r_squared: {metrics['r_squared']:.4f}  nobs: {metrics['nobs']}")
        table = pd.DataFrame({"coefficient": self.model.params, "std_err": self.model.bse})
        return "\n".join(lines) + "\n" + table.to_string()

    def get_model_metrics(self) -> Dict[str, Any]:
        metrics = super().get_model_metrics()
        metrics["weight_function"] = self.weight_function
        if getattr(self, "theta", None) is not None:
            metrics["theta"] = [float(t) for t in self.theta]
        return metrics
