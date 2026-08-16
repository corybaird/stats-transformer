from pathlib import Path
import pandas as pd
from stats_transformer.models.timeseries.reduced_form.var import VARModel
from stats_transformer.models.regression.robust_ols import RobustOLSModel

class AcostaBrennanJacobson2024Replication:

    def __init__(self, data_path=None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path("data/examples/academic/acosta_brennan_jacobson_2024/sofr_surprises.parquet.gzip")
        self.var_model = None
        self.ols_model = None

    def _load_data(self):
        df = pd.read_parquet(self.data_path)
        return df.reset_index()

    def run(self):
        df = self._load_data()
        self.var_model = VARModel(target_variables=["gss_target", "gss_path", "ns"], maxlags=2)
        var_metrics = self.var_model.fit(df)
        self.ols_model = RobustOLSModel(target="ns", independent_variables=["gss_target", "gss_path"], cov_type="HC1")
        ols_metrics = self.ols_model.fit(df)
        print("Acosta, Brennan, & Jacobson (2024) VAR Metrics:", var_metrics)
        print("Acosta, Brennan, & Jacobson (2024) OLS Metrics:", ols_metrics)
        return {"var_metrics": var_metrics, "ols_metrics": ols_metrics}
