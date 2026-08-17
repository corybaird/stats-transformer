from pathlib import Path
import pandas as pd
from stats_transformer.models.regression.robust_ols import RobustOLSModel
from stats_transformer.models.regression.did import DiDModel

class Lane2025Replication:

    def __init__(self, data_path=None, tariffs_path=None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path("data/examples/academic/lane_2025/policy_loans.parquet.gzip")
        if tariffs_path:
            self.tariffs_path = Path(tariffs_path)
        else:
            self.tariffs_path = Path("data/examples/academic/lane_2025/tariffs.parquet.gzip")
        self.model = None
        self.did_model = None

    def _load_data(self):
        df = pd.read_parquet(self.data_path)
        return df.reset_index().dropna(subset=["tot_change", "eq_change", "hci"])

    def _load_tariff_panel_with_derived_cohort(self):
        # The tariffs panel has no first-treatment cohort column: `hci` is a
        # fixed per-product characteristic that never varies over time, so it
        # cannot serve as a staggered-adoption treatment indicator. This
        # derives a proxy cohort instead -- the first year each product's
        # tariff falls at least 20% below its own first-observed level, i.e.
        # a liberalization event -- which is NOT the paper's actual treatment
        # definition, only a plausible stand-in built from what this extract
        # contains. Results here are illustrative, not a verified replication.
        df = pd.read_parquet(self.tariffs_path).reset_index()
        df = df[["id", "productcode", "year", "tariff"]].dropna()
        # id == 0 is an unassigned/placeholder id in the source extract with
        # multiple conflicting tariff rows per year; drop it rather than
        # silently averaging incompatible observations.
        df = df[df["id"] != 0]
        df = df.drop_duplicates(subset=["id", "year"])
        first_tariff = df.sort_values("year").groupby("id")["tariff"].first()
        df = df.merge(first_tariff.rename("first_tariff"), on="id")
        df["liberalized"] = df["tariff"] < 0.8 * df["first_tariff"]
        cohort = df[df["liberalized"]].groupby("id")["year"].min()
        df["cohort"] = df["id"].map(cohort).fillna(0).astype(int)
        return df

    def run(self):
        df = self._load_data()
        self.model = RobustOLSModel(target="tot_change", independent_variables=["eq_change", "hci"], cov_type="HC1")
        metrics = self.model.fit(df)
        print("Lane (2025) Policy Loans Robust OLS Metrics:", metrics)

        tariff_df = self._load_tariff_panel_with_derived_cohort()
        self.did_model = DiDModel(entity_column="id", time_column="year", cohort_column="cohort", outcome_column="tariff", control_group="never_treated")
        did_metrics = self.did_model.fit(tariff_df)
        print("Lane (2025) Tariff Liberalization DiD Metrics (illustrative, derived cohort):", did_metrics)
        # The pre-trend test rejects by construction here: "treated" is
        # defined directly from the tariff drop, so treated units mechanically
        # trend differently before their own event by definition, not because
        # of a genuine parallel-trends violation. This is an artifact of the
        # derived-cohort proxy, not a property of Callaway-Sant'Anna itself --
        # see the synthetic-panel tests in tests/test_models/test_did_model.py
        # for the estimator's actual correctness guarantees.

        return {"metrics": metrics, "did_metrics": did_metrics}
