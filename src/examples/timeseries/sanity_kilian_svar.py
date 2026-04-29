import statsmodels.api as sm
import pandas as pd
import numpy as np
from src.stats_transformer.models.timeseries.svar import SVARModel

class SVARSanityCheck:
    def __init__(self):
        self.data = sm.datasets.macrodata.load_pandas().data
        self.target_vars = ['realgdp', 'realcons', 'realinv']
        for col in self.target_vars:
            self.data[col] = np.log(self.data[col])
            
    def run(self):
        print("Running SVAR Model Sanity Check...")
        A = np.asarray([[1, 0, 0],
                        ['E', 1, 0],
                        ['E', 'E', 1]], dtype='object')
        B = np.asarray([['E', 0, 0],
                        [0, 'E', 0],
                        [0, 0, 'E']], dtype='object')
        svar = SVARModel(target_variables=self.target_vars, maxlags=2, svar_type='AB', A=A, B=B)
        metrics = svar.fit(self.data)
        print("Model fitted successfully.")
        print("Metrics:", metrics)
        print("\nSummary:")
        print(svar.get_summary())

if __name__ == "__main__":
    SVARSanityCheck().run()
