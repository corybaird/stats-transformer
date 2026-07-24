import pathlib
import pandas as pd
from stats_transformer.models.timeseries.var import VARModel

class StockWatson2001Replication:

    def __init__(self, data_path=None):
        if data_path:
            self.data_path = pathlib.Path(data_path)
        else:
            self.data_path = pathlib.Path("data/examples/matlab_examples/SW2001_Data.xlsx")
        self.model = None

    def load_data(self):
        df = pd.read_excel(self.data_path)
        df = df.iloc[1:].reset_index(drop=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def run(self):
        df = self.load_data()
        vars_to_use = [col for col in df.columns if col.lower() not in ['unnamed: 0', 'date', 'year', 'quarter'] and pd.api.types.is_numeric_dtype(df[col])][:3]
        self.model = VARModel(target_variables=vars_to_use, maxlags=4)
        metrics = self.model.fit(df)
        return {"metrics": metrics, "summary": self.model.get_summary()}
