import pathlib
import pandas as pd
from stats_transformer.models.timeseries.blanchard_quah import BlanchardQuahModel

class BlanchardQuah1989Replication:

    def __init__(self, data_path=None):
        if data_path:
            self.data_path = pathlib.Path(data_path)
        else:
            self.data_path = pathlib.Path("references/matlab_benchmarks/Replic/BQ1989/BQ1989_Data.xlsx")
        self.model = None

    def load_data(self):
        df = pd.read_excel(self.data_path)
        df = df.iloc[1:].reset_index(drop=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def run(self):
        df = self.load_data()
        vars_to_use = [col for col in df.columns if col.lower() not in ['unnamed: 0', 'date', 'year', 'quarter'] and pd.api.types.is_numeric_dtype(df[col])][:2]
        self.model = BlanchardQuahModel(target_variables=vars_to_use, maxlags=8)
        metrics = self.model.fit(df)
        return {"metrics": metrics, "summary": self.model.get_summary(), "B_0": self.model.B_0}
