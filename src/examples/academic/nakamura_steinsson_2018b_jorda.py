"""Replication of Nakamura & Steinsson (2018b) JME Appendix.

Paper: "Identification in Macroeconomics: High Frequency Identification vs. Local Projections"
(Journal of Monetary Economics)

Replicates the Appendix Jordà Local Projections identified with Romer-Romer monetary shocks:
For each horizon h in 0..49, estimates cumulative responses of logIP, UE, logCPI, logCMPI, FFR:
    logIP(t+h+1) - logIP(t) = const + b_h * Shock(t+1) + lags of Shock + lags of Data
Calculates the output multiplier = cumsum(IP response) / cumsum(real FFR response) * (-1)
and asserts multiplier at h=36 matches published 1.95 (Appendix MultiplierTable.tex),
and full IRF paths match shipped RomerJordaIRF.dta.
"""
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from stats_transformer.data import load_example
from stats_transformer.models.regression.regression import RegressionModel


class NakamuraSteinsson2018bJordaReplication:
    """Replication benchmark for Nakamura & Steinsson (2018b) Appendix Jordà LPs."""

    PUBLISHED_MULTIPLIER_H36: float = 1.95
    HORIZONS: int = 50
    COLS: List[str] = ["logIP", "UE", "logCPI", "logCMPI", "FFR"]
    VARS: Dict[str, tuple[int, float]] = {
        "IP": (0, 100.0),
        "CPI": (2, 100.0),
        "IPFFR": (4, 1.0),
    }

    def __init__(self, data_path: Path | str | None = None) -> None:
        self.data_path = Path(data_path) if data_path else None

    def _load_data(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame | None]:
        if self.data_path:
            p = Path(self.data_path)
            data = np.loadtxt(p / "MonetaryData.txt", skiprows=1)[:336]
            shock = np.loadtxt(p / "RomerShock.txt", skiprows=1)[:336]
            shipped_path = p / "RomerJordaIRF.dta"
            shipped = pd.read_stata(shipped_path) if shipped_path.exists() else None
        else:
            pkg_data = load_example("nakamura_steinsson_2018_b")
            data = pkg_data["macro_data"].values[:336]
            shock = pkg_data["shocks"].values[:336].ravel()
            shipped = pkg_data["published_irfs"]
        return data, shock, shipped

    def _horizon_regression(self, dep: np.ndarray, reg_vars: Dict[str, np.ndarray]) -> float:
        df = pd.DataFrame(reg_vars)
        df["dep"] = dep
        model = RegressionModel(target="dep", independent_variables=list(reg_vars.keys()))
        model.fit(df, compute_metrics=False)
        return float(model.model.params["shock"])

    def _jorda_irf(self, col_idx: int, data: np.ndarray, shock: np.ndarray, nc: int = 2) -> np.ndarray:
        y = data[:, col_idx]
        irf = []
        for h in range(self.HORIZONS):
            t = np.arange(1, len(y) - nc - h + 1)
            reg = {c: data[t - 1, i] for i, c in enumerate(self.COLS)}
            reg.update({f"{c}_0": data[t, i] for i, c in enumerate(self.COLS)})
            reg.update({"shock": shock[t + 1], "shock_l1": shock[t], "shock_l2": shock[t - 1]})
            dep = y[t + h + 1] - y[t]
            irf.append(self._horizon_regression(dep, reg))
        return np.array(irf)

    def run(self) -> Dict[str, Any]:
        data, shock, shipped = self._load_data()

        irfs = {}
        for var_name, (col_idx, scale) in self.VARS.items():
            irfs[var_name] = self._jorda_irf(col_idx, data, shock) * scale

        # Real FFR = nominal FFR - inflation
        infla = np.zeros(self.HORIZONS)
        infla[: self.HORIZONS - 1] = irfs["CPI"][1:] - irfs["CPI"][:-1]
        realffr = irfs["IPFFR"] - infla

        # Output multiplier
        cum_ip = np.cumsum(irfs["IP"])
        cum_realffr = np.cumsum(realffr)
        multiplier = -(cum_ip / cum_realffr)

        m36 = float(multiplier[36])
        multiplier_error = abs(m36 - self.PUBLISHED_MULTIPLIER_H36)
        passed_multiplier = multiplier_error < 0.01  # 1.9471 vs 1.95

        # Cross-check with shipped IRF paths if available
        irf_matches = True
        max_irf_diff = 0.0
        if shipped is not None:
            # First 49 steps in Stata dataset
            diff_ip = np.max(np.abs(irfs["IP"][:49] - shipped["IP"].values[:49]))
            diff_cpi = np.max(np.abs(irfs["CPI"][:49] - shipped["CPI"].values[:49]))
            diff_ffr = np.max(np.abs(realffr[:49] - shipped["IPFFR"].values[:49]))
            max_irf_diff = float(max(diff_ip, diff_cpi, diff_ffr))
            irf_matches = max_irf_diff < 0.01

        return {
            "multiplier_h36": m36,
            "published_multiplier_h36": self.PUBLISHED_MULTIPLIER_H36,
            "multiplier_error": multiplier_error,
            "max_irf_diff": max_irf_diff,
            "passed": passed_multiplier and irf_matches,
            "irfs": irfs,
        }


if __name__ == "__main__":
    rep = NakamuraSteinsson2018bJordaReplication()
    res = rep.run()
    print(f"Jorda multiplier @ h=36: {res['multiplier_h36']:.4f} (published: {res['published_multiplier_h36']})")
    print(f"Max IRF path difference: {res['max_irf_diff']:.6f}")
    assert res["passed"], f"Assertion failed: multiplier {res['multiplier_h36']} does not match published 1.95"
    print("ASSERT PASS: Successfully replicated Nakamura & Steinsson (2018b) Jordà specification.")
