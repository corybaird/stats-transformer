"""Replication of Bauer & Swanson (2023).

Paper: "A Reassessment of Monetary Policy Surprises and High-Frequency Identification"
(NBER Macroeconomics Annual, 2023)

Replicates the paper's macro news regression: monthly FOMC monetary policy
surprises (orthogonalized, `mps_orth`) regressed on pre-meeting macro news
(nonfarm payroll surprise, monthly change in the unemployment rate, and monthly
CPI inflation), following the paper's Table (macro news predictors of policy
surprises). Estimated with RobustOLSModel(cov_type="HC3").

NOTE: replication outputs/PDF with the published coefficient table are not
available in the source replication repository (the shipped folder is
confidential/empty), so the benchmark asserts the library's RobustOLSModel
against a direct statsmodels HC3 computation on the identical sample instead of
published numbers.
"""
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from stats_transformer.data import load_example
from stats_transformer.models.regression.robust_ols import RobustOLSModel


class BauerSwanson2023Replication:
    """Replication of Bauer & Swanson (2023) macro news regression."""

    TARGET: str = "mps_orth"
    NEWS: list[str] = ["nfp_surp", "unemp_change", "cpi_inflation"]
    TOLERANCE: float = 1e-4

    def __init__(self, data_path: Path | str | None = None) -> None:
        self.data_path = Path(data_path) if data_path else None

    def _load_data(self) -> pd.DataFrame:
        if self.data_path:
            return pd.read_parquet(self.data_path)
        return load_example("bauer_swanson_2023", member="fomc_surprises_monthly")

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["unemp_change"] = df["unemp"].diff()
        df["cpi_inflation"] = np.log(df["cpi"]).diff()
        return df

    def run(self) -> Dict[str, Any]:
        import statsmodels.api as sm

        df = self._prepare(self._load_data())

        model = RobustOLSModel(target=self.TARGET, independent_variables=self.NEWS, cov_type="HC3")
        model.fit(df, compute_metrics=False)
        params = model.model.params
        r2 = float(model.model.rsquared)
        bse = model.model.bse

        # Reference computation: statsmodels HC3 on the identical estimation sample
        sub = df[[self.TARGET] + self.NEWS].dropna()
        ref = sm.OLS(sub[self.TARGET], sm.add_constant(sub[self.NEWS])).fit(cov_type="HC3")

        max_coef_err = float(np.max(np.abs(params - ref.params)))
        max_se_err = float(np.max(np.abs(bse - ref.bse)))
        r2_err = abs(r2 - float(ref.rsquared))
        passed = max(max_coef_err, r2_err) <= self.TOLERANCE

        return {
            "n_observations": int(model.model.nobs),
            "coefficients": {k: round(float(v), 6) for k, v in params.items()},
            "hc3_standard_errors": {k: round(float(v), 6) for k, v in bse.items()},
            "r_squared": round(r2, 6),
            "reference_coefficients": {k: float(v) for k, v in ref.params.items()},
            "max_coef_error": max_coef_err,
            "max_se_error": max_se_err,
            "r2_error": r2_err,
            "passed": passed,
        }


if __name__ == "__main__":
    rep = BauerSwanson2023Replication()
    res = rep.run()
    print(f"Observations: {res['n_observations']}")
    print("Coefficients (HC3):", res["coefficients"])
    print(f"R^2: {res['r_squared']}")
    print(f"Max coefficient error vs statsmodels: {res['max_coef_error']:.2e}")
    assert res["passed"], f"Assertion failed: coefficient error {res['max_coef_error']} exceeds {rep.TOLERANCE}"
    print("ASSERT PASS: RobustOLSModel HC3 matches direct statsmodels computation.")