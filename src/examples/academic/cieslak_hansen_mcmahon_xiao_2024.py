from pathlib import Path
import pandas as pd
from stats_transformer.models.regression.regression import RegressionModel
from stats_transformer.models.timeseries.reduced_form.var import VARModel

class CieslakHansenMcMahonXiao2024Replication:

    def __init__(self, data_path=None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path("data/examples/academic/cieslak_hansen_mcmahon_xiao_2024/pmu_data.parquet.gzip")
        self.ols_model = None
        self.var_model = None

    def _load_data(self):
        df = pd.read_parquet(self.data_path)
        return df.reset_index()

    def run(self):
        df = self._load_data()
        self.ols_model = RegressionModel(target="unct_iw_staff_1", independent_variables=["unct_eg_staff_1", "unct_mkt_staff_1"])
        ols_metrics = self.ols_model.fit(df)
        self.var_model = VARModel(target_variables=["unct_iw_staff_1", "unct_eg_staff_1"], maxlags=1)
        var_metrics = self.var_model.fit(df)
        print("Cieslak et al. (2024) OLS Metrics:", ols_metrics)
        print("Cieslak et al. (2024) VAR Metrics:", var_metrics)
        return {"ols_metrics": ols_metrics, "var_metrics": var_metrics}
