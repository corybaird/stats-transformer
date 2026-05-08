import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from stats_transformer.models.discrete.logit import LogitModel
from src.examples.base import BaseExample

class SpectorLogitExample(BaseExample):
    def __init__(self):
        self.data_dir = "data/examples/discrete"
        self.data_path = f"{self.data_dir}/spector.csv"
        os.makedirs(self.data_dir, exist_ok=True)

    def _fetch_and_prepare_data(self):
        print("Fetching Spector and Mazzeo (1980) Dataset...")
        df_raw = sm.datasets.spector.load_pandas().data
        df_raw.to_csv(self.data_path, index=False)
        print(f"Data saved to {self.data_path}")
        return df_raw

    def _compute_baseline(self, df):
        print("\n" + "="*50)
        print("2. ORIGINAL STATSMODELS LOGIT")
        print("="*50)
        
        y = df["GRADE"]
        X = sm.add_constant(df[["GPA", "TUCE", "PSI"]])
        
        model = sm.Logit(y, X).fit(disp=False)
        print(model.summary())
        return model

    def _compute_stats_transformer(self, df):
        print("\n" + "="*50)
        print("3. STATS-TRANSFORMER LOGIT MODEL")
        print("="*50)
        
        st_model = LogitModel(
            target="GRADE",
            independent_variables=["GPA", "TUCE", "PSI"]
        )
        st_model.fit(df)
        print(st_model.get_summary())
        return st_model

    def _compare_and_report(self, orig_model, st_model):
        print("\n" + "="*50)
        print("4. FINAL COMPARISON")
        print("="*50)
        
        orig_params = orig_model.params.sort_index().values
        st_params = st_model.model.params.sort_index().values
        
        orig_bse = orig_model.bse.sort_index().values
        st_bse = st_model.model.bse.sort_index().values
        
        param_diff = np.max(np.abs(orig_params - st_params))
        bse_diff = np.max(np.abs(orig_bse - st_bse))
        
        print(f"Max difference in Coefficients: {param_diff:.6e}")
        print(f"Max difference in Standard Errors: {bse_diff:.6e}")
        
        if param_diff < 1e-10 and bse_diff < 1e-10:
            print("\nCONCLUSION: Stats-Transformer perfectly matches native statsmodels Logit.")
        else:
            print("\nCONCLUSION: Significant deviations detected.")

if __name__ == "__main__":
    example = SpectorLogitExample()
    example.run()
