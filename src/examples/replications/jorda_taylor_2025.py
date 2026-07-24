import pathlib
import pandas as pd
from stats_transformer.models.timeseries.local_projections_iv import LocalProjectionsIVModel

class JordaTaylor2025Replication:

    def __init__(self, data_path=None):
        if data_path:
            self.data_path = pathlib.Path(data_path)
        else:
            self.data_path = pathlib.Path("references/matlab_benchmarks/Replic/JT2025/JT2025_Data.xlsx")
        self.model = None

    def load_data(self):
        df = pd.read_excel(self.data_path)
        df = df.iloc[1:].reset_index(drop=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def run(self):
        df = self.load_data()
        num_cols = [c for c in df.columns if c.lower() not in ['unnamed: 0', 'date', 'year', 'quarter'] and pd.api.types.is_numeric_dtype(df[c])]
        target_var = num_cols[0]
        shock_var = num_cols[1] if len(num_cols) > 1 else num_cols[0]
        inst_var = num_cols[2] if len(num_cols) > 2 else shock_var
        self.model = LocalProjectionsIVModel(target_variable=target_var, shock_variable=shock_var, instrument_variable=inst_var, horizons=10)
        metrics = self.model.fit(df)
        return {"metrics": metrics, "irf_coefficients": self.model.irf_coefficients}
