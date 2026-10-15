import numpy as np
import statsmodels.api as sm
from stats_transformer.data import load_example
from stats_transformer.models.regression.robust_ols import RobustOLSModel
from src.examples.base import BaseExample

class MincerWageExample(BaseExample):
    def _fetch_and_prepare_data(self):
        print("Loading Mincer Wage Data (1976 CPS from Wooldridge)...")
        df_raw = load_example("mincer_wage")
        
        print("\n" + "="*50)
        print("1. PREPARED DATA (Last 5 Rows)")
        print("="*50)
        print(df_raw[["lwage", "educ", "exper", "expersq", "tenure"]].tail())
        
        return df_raw

    def _compute_baseline(self, df):
        print("\n" + "="*50)
        print("2. ORIGINAL STATSMODELS OLS (HC3 Robust SE)")
        print("="*50)
        
        y = df["lwage"]
        X = sm.add_constant(df[["educ", "exper", "expersq", "tenure"]])
        
        model = sm.OLS(y, X).fit(cov_type="HC3")
        
        print(model.summary())
        return model

    def _compute_stats_transformer(self, df):
        print("\n" + "="*50)
        print("3. STATS-TRANSFORMER ROBUST OLS MODEL")
        print("="*50)
        
        st_model = RobustOLSModel(
            target="lwage",
            independent_variables=["educ", "exper", "expersq", "tenure"],
            cov_type="HC3"
        )
        st_model.fit(df)
        
        print(st_model.get_summary())
        return st_model

    def _compare_and_report(self, sm_model, st_model):
        print("\n" + "="*50)
        print("4. FINAL COMPARISON")
        print("="*50)
        
        sm_params = sm_model.params.values
        st_params = st_model.model.params.values
        
        sm_bse = sm_model.bse.values
        st_bse = st_model.model.bse.values
        
        param_diff = np.max(np.abs(sm_params - st_params))
        bse_diff = np.max(np.abs(sm_bse - st_bse))
        
        print(f"Max difference in Coefficients: {param_diff:.6e}")
        print(f"Max difference in Standard Errors: {bse_diff:.6e}")
        
        if param_diff < 1e-10 and bse_diff < 1e-10:
            print("\nCONCLUSION: Stats-Transformer perfectly matches native statsmodels robust OLS for the Mincer wage equation.")
        else:
            print("\nCONCLUSION: Significant deviations detected.")

if __name__ == "__main__":
    example = MincerWageExample()
    example.run()
