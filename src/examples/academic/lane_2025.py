"""Replication of Lane (2025), "Manufacturing Revolutions: Industrial Policy
and Industrialization in South Korea" (QJE).

Replicates the paper's HCI research design: industries targeted by the 1973
Heavy and Chemical Industry (HCI) drive (the `hci` column, a fixed
per-product indicator) are compared with non-targeted industries before vs
after 1973, on the authors' policy panel (tariff and quantitative
restriction measures) and the policy-loans robust OLS. Data follows the
authors' replication files as shipped in data/examples/academic/lane_2025/
(tariffs.parquet.gzip, policy_loans.parquet.gzip), estimated with the
library's DiDModel (Callaway-Sant'Anna, single cohort, never-treated
controls) and RobustOLSModel.

Approximation: 1973 itself is not an observed period in the tariff panel
(years 1968-1982 in two-year steps), so the cohort is dated to the first
observed post-drive year (1974) with 1968 as the pre-drive baseline; with a
single pre-drive period the DiD estimator yields no pre-trend ATTs, so the
pre-trend test is unavailable rather than reported.
"""

from pathlib import Path

import pandas as pd

from stats_transformer.models.regression.did import DiDModel
from stats_transformer.models.regression.robust_ols import RobustOLSModel

from examples._print import format_metrics

HCI_DRIVE_YEAR = 1973
# First observed post-drive year in the tariff panel: `hci` is a fixed
# per-product indicator, `post` is year >= HCI_DRIVE_YEAR, and 1973 itself
# is not in the panel (1968, 74, 76, 78, 80, 82).
HCI_COHORT = 74


class Lane2025Replication:

    def __init__(self, data_path=None, tariffs_path=None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path(__file__).resolve().parents[3] / "data/examples/academic/lane_2025/policy_loans.parquet.gzip"
        if tariffs_path:
            self.tariffs_path = Path(tariffs_path)
        else:
            self.tariffs_path = Path(__file__).resolve().parents[3] / "data/examples/academic/lane_2025/tariffs.parquet.gzip"
        self.model = None
        self.did_models = {}

    def _load_data(self):
        df = pd.read_parquet(self.data_path)
        return df.reset_index().dropna(subset=["tot_change", "eq_change", "hci"])

    def _load_tariff_panel(self):
        df = pd.read_parquet(self.tariffs_path).reset_index()
        df = df[["id", "productcode", "year", "tariff", "qr", "hci"]].dropna(subset=["tariff", "qr"])
        # id == 0 is an unassigned/placeholder id in the source extract with
        # multiple conflicting tariff rows per year; drop it rather than
        # silently averaging incompatible observations.
        df = df[df["id"] != 0].drop_duplicates(subset=["id", "year"])
        # Paper's treatment: targeted by the 1973 HCI drive (hci == 1) vs
        # never-targeted controls (hci == 0), post vs pre the drive.
        df["cohort"] = df["hci"].map({1: HCI_COHORT, 0: 0}).astype(int)
        return df

    def _run_did(self, tariff_df, outcome_column):
        model = DiDModel(entity_column="id", time_column="year", cohort_column="cohort", outcome_column=outcome_column, control_group="never_treated")
        self.did_models[outcome_column] = model
        metrics = model.fit(tariff_df)
        print(f"Lane (2025) HCI DiD ({outcome_column}, targeted 1973 vs never-treated, post vs pre):")
        format_metrics(metrics)
        # Only one pre-drive period (1968) is observed, so the varying base
        # period yields no pre-trend placebo ATT(g,t): the model's
        # pre-trend test is empty here, not passed or failed.
        return metrics

    def run(self):
        df = self._load_data()
        self.model = RobustOLSModel(target="tot_change", independent_variables=["eq_change", "hci"], cov_type="HC1")
        metrics = self.model.fit(df)
        print("Lane (2025) Policy Loans Robust OLS:")
        format_metrics(metrics)

        tariff_df = self._load_tariff_panel()
        did_metrics = self._run_did(tariff_df, "tariff")
        qr_did_metrics = self._run_did(tariff_df, "qr")

        return {"metrics": metrics, "did_metrics": did_metrics, "qr_did_metrics": qr_did_metrics}


if __name__ == "__main__":
    Lane2025Replication().run()