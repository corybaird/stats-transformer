import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from src.stats_transformer.models.timeseries.var import VARModel

class AgentSanityMacroVAR:
    def __init__(self):
        self.data_dir = "data/examples/timeseries"
        self.data_path = f"{self.data_dir}/macrodata.csv"
        os.makedirs(self.data_dir, exist_ok=True)

    def _fetch_and_prepare_data(self):
        print("Fetching Statsmodels Macrodata for VAR...")
        df_raw = sm.datasets.macrodata.load_pandas().data
        
        # Following statsmodels VAR example, we use the differences of log data for realgdp, realcons, realinv
        # We'll just use raw realgdp, realcons, realinv for simplicity, or difference them
        data = df_raw[['realgdp', 'realcons', 'realinv']].copy()
        data = 100 * np.log(data).diff().dropna()
        data['year'] = df_raw['year'][1:].values
        data['quarter'] = df_raw['quarter'][1:].values
        
        data.to_csv(self.data_path, index=False)
        print(f"Data saved to {self.data_path}")
        return data

    def _compute_statsmodels_var(self, df):
        print("\n" + "="*50)
        print("2. ORIGINAL STATSMODELS VAR")
        print("="*50)
        
        y = df[['realgdp', 'realcons', 'realinv']]
        var_spec = VAR(y)
        model = var_spec.fit(maxlags=2)
        print(model.summary())
        return model

    def _compute_stats_transformer_var(self, df):
        print("\n" + "="*50)
        print("3. STATS-TRANSFORMER VAR MODEL")
        print("="*50)
        
        st_model = VARModel(
            target_variables=['realgdp', 'realcons', 'realinv'],
            maxlags=2
        )
        st_model.fit(df)
        print(st_model.get_summary())
        return st_model

    def _compare_and_report(self, orig_model, st_model):
        print("\n" + "="*50)
        print("4. FINAL COMPARISON")
        print("="*50)
        
        orig_params = orig_model.params.values
        st_params = st_model.model.params.values
        
        orig_bse = orig_model.stderr.values
        st_bse = st_model.model.stderr.values
        
        param_diff = np.max(np.abs(orig_params - st_params))
        bse_diff = np.max(np.abs(orig_bse - st_bse))
        
        print(f"Max difference in Coefficients: {param_diff:.6e}")
        print(f"Max difference in Standard Errors: {bse_diff:.6e}")
        
        if param_diff < 1e-10 and bse_diff < 1e-10:
            print("\nCONCLUSION: Stats-Transformer perfectly matches native statsmodels VAR.")
        else:
            print("\nCONCLUSION: Significant deviations detected.")

    def run(self):
        df_clean = self._fetch_and_prepare_data()
        orig_model = self._compute_statsmodels_var(df_clean)
        st_model = self._compute_stats_transformer_var(df_clean)
        self._compare_and_report(orig_model, st_model)

if __name__ == "__main__":
    sanity = AgentSanityMacroVAR()
    sanity.run()
