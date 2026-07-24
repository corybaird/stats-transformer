import pathlib
import pandas as pd
from stats_transformer.models.timeseries.proxy_svar import ProxySVARModel

class GertlerKaradi2015Replication:

    def __init__(self, data_path=None):
        if data_path:
            self.data_path = pathlib.Path(data_path)
        else:
            self.data_path = pathlib.Path("data/examples/matlab_examples/GK2015_Data.xlsx")
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
        vars_to_use = num_cols[:3]
        inst_var = num_cols[-1] if len(num_cols) > 3 else vars_to_use[0]
        self.model = ProxySVARModel(target_variables=vars_to_use, instrument_variable=inst_var, maxlags=12)
        metrics = self.model.fit(df)
        return {"metrics": metrics, "summary": self.model.get_summary(), "impact": self.model.impact_column}
