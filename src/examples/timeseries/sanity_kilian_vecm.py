import statsmodels.api as sm
import pandas as pd
import numpy as np
from stats_transformer.models.timeseries.vecm import VECMModel

class VECMSanityCheck:
    def __init__(self):
        self.data = sm.datasets.macrodata.load_pandas().data
        self.target_vars = ['realgdp', 'realcons', 'realinv']
        # Take logs for these variables typically done in macro
        for col in self.target_vars:
            self.data[col] = np.log(self.data[col])
            
    def run(self):
        print("Running VECM Model Sanity Check...")
        vecm = VECMModel(target_variables=self.target_vars, k_ar_diff=2, deterministic='co')
        metrics = vecm.fit(self.data)
        print("Model fitted successfully.")
        print("Metrics:", metrics)
        print("\nSummary:")
        print(vecm.get_summary())

if __name__ == "__main__":
    VECMSanityCheck().run()
