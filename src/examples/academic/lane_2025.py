from pathlib import Path
import pandas as pd
from stats_transformer.models.regression.robust_ols import RobustOLSModel

class Lane2025Replication:

    def __init__(self, data_path=None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path("data/examples/academic/lane_2025/policy_loans.parquet.gzip")
        self.model = None

    def _load_data(self):
        df = pd.read_parquet(self.data_path)
        return df.reset_index().dropna(subset=["tot_change", "eq_change", "hci"])

    def run(self):
        df = self._load_data()
        self.model = RobustOLSModel(target="tot_change", independent_variables=["eq_change", "hci"], cov_type="HC1")
        metrics = self.model.fit(df)
        print("Lane (2025) Policy Loans Robust OLS Metrics:", metrics)
        return {"metrics": metrics}
