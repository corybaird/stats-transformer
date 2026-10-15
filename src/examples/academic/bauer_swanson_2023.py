"""Replication of Bauer & Swanson (2023).

Paper: "A Reassessment of Monetary Policy Surprises and High-Frequency Identification"
(NBER Macroeconomics Annual, 2023)

Replicates the paper's central finding that RAW FOMC monetary policy surprises
are predictable from pre-meeting macro news: the monthly surprise (`mps`) is
regressed on the nonfarm payroll surprise (`nfp_surp`), 12-month payroll
growth (`nfp_12m`), and the 3-month log change of the CRB commodity price
index, estimated with RobustOLSModel(cov_type="HC3").

The paper's orthogonalized surprise (`mps_orth`, constructed by regressing
`mps` on macro news) is by construction unpredictable from news, so the same
regression with `mps_orth` as the dependent variable serves as a placebo:
its R^2 should be near zero. The paper's remaining predictors (S&P 500
pre-return, Treasury-skewness measures) are not part of the packaged monthly
dataset, so the news set here is the subset available in the data.

NOTE: replication outputs/PDF with the published coefficient table are not
available in the source replication repository (the shipped folder is
confidential/empty), so the benchmark asserts the library's RobustOLSModel
against a direct statsmodels HC3 computation on the identical sample and
checks the paper's predictability finding economically; no published numbers
are asserted.
"""
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from stats_transformer.data import load_example
from stats_transformer.models.regression.robust_ols import RobustOLSModel


class BauerSwanson2023Replication:
    """Replication of Bauer & Swanson (2023) macro news regression."""

    TARGET: str = "mps"
    PLACEBO_TARGET: str = "mps_orth"
    NEWS: list[str] = ["nfp_surp", "nfp_12m", "crb_comm_3m"]
    TOLERANCE: float = 1e-4
    # Economic thresholds, set with margin around measured values
    # (raw mps R2 ~ 0.11, mps_orth placebo R2 ~ 0.002)
    MIN_NEWS_R2: float = 0.05
    MAX_PLACEBO_R2: float = 0.02
    SIGNIFICANCE: float = 0.05

    def __init__(self, data_path: Path | str | None = None) -> None:
        self.data_path = Path(data_path) if data_path else None

    def _load_data(self) -> pd.DataFrame:
        if self.data_path:
            return pd.read_parquet(self.data_path)
        return load_example("bauer_swanson_2023", member="fomc_surprises_monthly")

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["crb_comm_3m"] = np.log(df["crb_pcomm"]).diff(3)
        return df

    def _fit(self, df: pd.DataFrame, target: str) -> RobustOLSModel:
        model = RobustOLSModel(target=target, independent_variables=self.NEWS, cov_type="HC3")
        model.fit(df, compute_metrics=False)
        return model

    def run(self) -> Dict[str, Any]:
        import statsmodels.api as sm

        df = self._prepare(self._load_data())
        model = self._fit(df, self.TARGET)
        params = model.model.params
        r2 = float(model.model.rsquared)
        bse = model.model.bse

        # Reference computation: statsmodels HC3 on the identical estimation sample
        sub = df[[self.TARGET] + self.NEWS].dropna()
        ref = sm.OLS(sub[self.TARGET], sm.add_constant(sub[self.NEWS])).fit(cov_type="HC3")

        max_coef_err = float(np.max(np.abs(params - ref.params)))
        r2_err = abs(r2 - float(ref.rsquared))

        # Economic checks: raw mps is predictable from news, mps_orth is not.
        placebo = self._fit(df, self.PLACEBO_TARGET)
        placebo_r2 = float(placebo.model.rsquared)
        sig_news = [n for n in self.NEWS if float(ref.pvalues[n]) < self.SIGNIFICANCE]

        passed = (
            max_coef_err <= self.TOLERANCE
            and r2_err <= self.TOLERANCE
            and r2 >= self.MIN_NEWS_R2
            and placebo_r2 < self.MAX_PLACEBO_R2
            and bool(sig_news)
        )

        return {
            "n_observations": int(model.model.nobs),
            "coefficients": {k: round(float(v), 6) for k, v in params.items()},
            "hc3_standard_errors": {k: round(float(v), 6) for k, v in bse.items()},
            "r_squared": round(r2, 6),
            "reference_coefficients": {k: float(v) for k, v in ref.params.items()},
            "max_coef_error": max_coef_err,
            "r2_error": r2_err,
            "placebo_r_squared": round(placebo_r2, 6),
            "significant_news": sig_news,
            "passed": passed,
        }


if __name__ == "__main__":
    rep = BauerSwanson2023Replication()
    res = rep.run()
    print(f"Observations: {res['n_observations']}")
    print("Coefficients (HC3):", res["coefficients"])
    print(f"R^2 (raw mps):        {res['r_squared']}")
    print(f"R^2 (mps_orth placebo): {res['placebo_r_squared']}")
    print(f"News significant at 5%: {res['significant_news']}")
    print(f"Max coefficient error vs statsmodels: {res['max_coef_error']:.2e}")
    assert res["passed"], f"Assertion failed: coef error {res['max_coef_error']}, r2 {res['r_squared']}, placebo r2 {res['placebo_r_squared']}"
    print("ASSERT PASS: RobustOLSModel HC3 matches statsmodels and reproduces the news-predictability finding.")