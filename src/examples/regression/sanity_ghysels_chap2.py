import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
from src.stats_transformer.models.regression.diagnostics import RegressionDiagnostics

class GhyselsChap2Sanity:
    def __init__(self):
        self.data_path = Path("data/raw/examples/Ghysels /Ch_2/simulated_datac2.csv")

    def run(self):
        sim_data_full = pd.read_csv(self.data_path)
        sim_data_est = sim_data_full.iloc[101:301]
        
        ols_fit = smf.ols(formula='y ~ x', data=sim_data_est).fit()
        print("OLS Summary:")
        print(ols_fit.summary())

        diagnostics = RegressionDiagnostics(ols_fit)
        
        bp = diagnostics.breusch_pagan_test()
        print(f"Breusch-Pagan: Stat={bp['statistic']:.4f}, p={bp['p_value']:.4f}")
        
        jb = diagnostics.jarque_bera_test()
        print(f"Jarque-Bera: Stat={jb['statistic']:.4f}, p={jb['p_value']:.4f}")
        
        white = diagnostics.white_test()
        print(f"White: Stat={white['statistic']:.4f}, p={white['p_value']:.4f}")
        
        dw = diagnostics.durbin_watson_test()
        print(f"Durbin-Watson: Stat={dw['statistic']:.4f}")
        
        split_point = int(0.5 * len(sim_data_est))
        chow = diagnostics.chow_test(sim_data_est, 'y ~ x', split_point)
        print(f"Chow Test (50% split): Stat={chow['statistic']:.4f}, p={chow['p_value']:.4f}")

if __name__ == "__main__":
    runner = GhyselsChap2Sanity()
    runner.run()
