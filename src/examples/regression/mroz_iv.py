import numpy as np
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
from stats_transformer.data import load_example
from stats_transformer.models.regression.iv import IV2SLSModel
from src.examples.base import BaseExample

class MrozIVExample(BaseExample):
    def _fetch_and_prepare_data(self):
        print("Loading Mroz (1987) Female Labor Supply Data...")
        # Tracked file is pre-cleaned: labor-force women, expersq added, rows with
        # missing motheduc/fatheduc dropped.
        return load_example("mroz")

    def _compute_baseline(self, df):
        print("\n" + "="*50)
        print("2. ORIGINAL LINEARMODELS IV/2SLS")
        print("="*50)
        
        y = df["lwage"]
        X_exog = sm.add_constant(df[["exper", "expersq"]])
        X_endog = df[["educ"]]
        Z = df[["motheduc", "fatheduc"]]
        
        model = IV2SLS(dependent=y, exog=X_exog, endog=X_endog, instruments=Z).fit(cov_type="robust")
        print(model.summary)
        return model

    def _compute_stats_transformer(self, df):
        print("\n" + "="*50)
        print("3. STATS-TRANSFORMER IV/2SLS MODEL")
        print("="*50)
        
        st_model = IV2SLSModel(
            target="lwage",
            independent_variables=["exper", "expersq"],
            endogenous=["educ"],
            instruments=["motheduc", "fatheduc"],
            cov_type="robust"
        )
        st_model.fit(df)
        print(st_model.get_summary())
        return st_model

    def _compare_and_report(self, orig_model, st_model):
        print("\n" + "="*50)
        print("4. FINAL COMPARISON")
        print("="*50)
        
        orig_params = orig_model.params.sort_index()
        st_params = st_model.model.params.sort_index()
        
        orig_bse = orig_model.std_errors.sort_index()
        st_bse = st_model.model.std_errors.sort_index()
        
        param_diff = np.max(np.abs(orig_params.values - st_params.values))
        bse_diff = np.max(np.abs(orig_bse.values - st_bse.values))
        
        print(f"Max difference in Coefficients: {param_diff:.6e}")
        print(f"Max difference in Standard Errors: {bse_diff:.6e}")
        
        if param_diff < 1e-10 and bse_diff < 1e-10:
            print("\nCONCLUSION: Stats-Transformer perfectly matches native linearmodels IV2SLS.")
        else:
            print("\nCONCLUSION: Significant deviations detected.")

if __name__ == "__main__":
    example = MrozIVExample()
    example.run()
