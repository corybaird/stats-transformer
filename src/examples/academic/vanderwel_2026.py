"""Replication of Hoogteijling, Martens & van der Wel (2026).

Paper: "A Simple New Benchmark for Forecasting Bond Risk Premia"
(International Journal of Forecasting, 2026)

Replicates Table 2, row "Diebold and Li":
Extracts the first 3 principal components of annual yield curve maturities (1y..10y)
using demean-only PCAModel (standardize=False), forecasts the factors with
direct 12-month-ahead AR(1) factor equations via RegressionModel, maps the
forecast factors back to yields, and converts to predicted 12-month excess bond returns.
Out-of-sample evaluation uses an expanding window (T_in = 221, OOS 1990-01 to 2018-12)
benchmarked against the historical expanding mean.
"""
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from stats_transformer.data import load_example
from stats_transformer.models.regression.regression import RegressionModel
from stats_transformer.models.unsupervised.unsupervised import PCAModel


class Vanderwel2026Replication:
    """Replication benchmark for Vanderwel et al. (2026) Table 2 Diebold-Li row."""

    BEGIN: str = "1971-08"
    END: str = "2018-12"
    T_IN: int = 221
    HP: int = 12
    NCOMP: int = 3
    MATURITIES: List[int] = [24, 36, 48, 60, 84, 120]
    PUBLISHED_R2: List[float] = [-0.236, -0.192, -0.165, -0.147, -0.141, -0.059]
    TOLERANCE: float = 0.0015

    def __init__(self, data_path: Path | str | None = None) -> None:
        self.data_path = Path(data_path) if data_path else None

    def _load_yields(self) -> pd.DataFrame:
        if self.data_path:
            yr = pd.read_csv(self.data_path, encoding="utf-8-sig").set_index("Year").astype(float)
        else:
            yr = load_example("vanderwel_2026").set_index("Year").astype(float)
        yr.index = pd.to_datetime(list(yr.index.astype(int).astype(str)), format="%Y%m")
        yr.columns = [i for i in range(1, len(yr.columns) + 1)]
        return yr.loc[self.BEGIN : self.END]

    def _excess_returns(self, yields: pd.DataFrame) -> pd.DataFrame:
        xr = pd.DataFrame(index=yields.index, columns=self.MATURITIES, dtype=float)
        hp = self.HP
        for mat in self.MATURITIES:
            xr.loc[xr.index[hp:], mat] = (
                mat / 12 * yields.iloc[:-hp, mat - 1].values
                - (mat - hp) / 12 * yields.iloc[hp:, mat - hp - 1].values
                - hp / 12 * yields.iloc[:-hp, hp - 1].values
            )
        return xr

    def _forecast_t(self, yields_reduced: pd.DataFrame, t: int) -> np.ndarray:
        sub_df = yields_reduced.iloc[0 : t - self.HP + 1].copy()
        feat_cols = [f"y_{c}" for c in sub_df.columns]
        sub_df.columns = feat_cols

        pca = PCAModel(features=feat_cols, n_components=self.NCOMP, standardize=False)
        pca.fit(sub_df)
        x_pc = pca.transformed_X  # (T_t, 3)

        x_pc_hat = np.empty_like(x_pc)
        for c in range(self.NCOMP):
            df_reg = pd.DataFrame({f"pc{c}": x_pc[: -self.HP, c], "pc_lead": x_pc[self.HP :, c]})
            reg = RegressionModel(target="pc_lead", independent_variables=[f"pc{c}"])
            reg.fit(df_reg, compute_metrics=False)
            pred = reg.predict(pd.DataFrame({f"pc{c}": x_pc[:, c]}))
            x_pc_hat[:, c] = pred.values

        # Inverse PCA of last in-sample row
        mean = pca.scaler.mean_
        loadings = pca.model.components_
        yield_pred = mean + x_pc_hat[-self.HP :] @ loadings

        yields_comb = np.concatenate([sub_df.values, yield_pred], axis=0)

        excess = np.empty((len(yields_comb), len(self.MATURITIES)))
        for i, mat in enumerate([m_ // 12 for m_ in self.MATURITIES]):
            excess[self.HP :, i] = (
                mat * yields_comb[: -self.HP, mat - 1]
                - (mat * 12 - self.HP) / 12 * yields_comb[self.HP :, mat - self.HP // 12 - 1]
                - self.HP / 12 * yields_comb[: -self.HP, self.HP // 12 - 1]
            )
        return np.concatenate([excess[-1, :], excess[-1, :].mean().reshape(1)], axis=0)

    def run(self) -> Dict[str, Any]:
        yields = self._load_yields()
        yields_reduced = yields[[c for c in range(self.HP, 132, 12)]]
        excess = self._excess_returns(yields)
        n = len(excess)

        pred = pd.DataFrame(index=excess.index, columns=self.MATURITIES + ["EW"], dtype=float)
        for t in range(self.T_IN, n):
            pred.iloc[t] = self._forecast_t(yields_reduced, t)

        xs = excess[self.MATURITIES].astype(float)
        pred_cols = pred[self.MATURITIES].astype(float)
        bench = xs.shift(self.HP).expanding().mean()
        mask = xs.index >= xs.index[self.T_IN]

        r2_oos = []
        for mat in self.MATURITIES:
            ok = (~xs[mat].isna() & ~pred_cols[mat].isna()).values * mask
            model_mspe = ((xs[mat] - pred_cols[mat]) ** 2)[ok].sum()
            bench_mspe = ((xs[mat] - bench[mat]) ** 2)[ok].sum()
            r2 = 1.0 - model_mspe / bench_mspe
            r2_oos.append(round(float(r2), 5))

        max_err = max(abs(a - b) for a, b in zip(r2_oos, self.PUBLISHED_R2, strict=False))
        passed = all(err <= self.TOLERANCE for err in [abs(a - b) for a, b in zip(r2_oos, self.PUBLISHED_R2, strict=False)])

        return {
            "maturities": self.MATURITIES,
            "r2_oos_replicated": r2_oos,
            "r2_oos_published": self.PUBLISHED_R2,
            "max_error": max_err,
            "passed": passed,
        }


if __name__ == "__main__":
    rep = Vanderwel2026Replication()
    res = rep.run()
    print("Maturities (months):", res["maturities"])
    print("Replicated R2 OOS:  ", res["r2_oos_replicated"])
    print("Published R2:       ", res["r2_oos_published"])
    print("Max error:          ", f"{res['max_error']:.6f}")
    assert res["passed"], f"Assertion failed: max error {res['max_error']} exceeds tolerance {rep.TOLERANCE}"
    print("ASSERT PASS: Successfully replicated Table 2 Diebold-Li row.")
