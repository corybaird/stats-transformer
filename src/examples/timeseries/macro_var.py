import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from stats_transformer.data import load_example
from stats_transformer.models.timeseries.reduced_form.var import VARModel
from src.examples.base import BaseExample

class MacroVARExample(BaseExample):
    def _fetch_and_prepare_data(self):
        print("Loading statsmodels macrodata (100 * log-diff of realgdp, realcons, realinv)...")
        return load_example("macrodata")

    def _compute_baseline(self, df):
        print("\n" + "="*50)
        print("2. ORIGINAL STATSMODELS VAR")
        print("="*50)
        
        y = df[['realgdp', 'realcons', 'realinv']]
        var_spec = VAR(y)
        model = var_spec.fit(maxlags=2)
        print(model.summary())
        return model

    def _compute_stats_transformer(self, df):
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

if __name__ == "__main__":
    example = MacroVARExample()
    example.run()
