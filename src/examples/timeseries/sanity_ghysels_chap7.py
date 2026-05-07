import pandas as pd
from pathlib import Path
from stats_transformer.models.timeseries.vecm import VECMModel

class GhyselsChap7VECMSanity:
    def __init__(self):
        self.data_path = Path("data/raw/examples/Ghysels /Ch_7/simulated_cointegration.csv")

    def run(self):
        sim_data = pd.read_csv(self.data_path)
        
        # VECM modeling with cointegration tests from Ch 7
        model = VECMModel(target_variables=['x', 'y'], k_ar_diff=1, deterministic='co')
        model.fit(sim_data)
        
        print("VECM Model Summary:")
        print(model.get_summary())

if __name__ == "__main__":
    runner = GhyselsChap7VECMSanity()
    runner.run()
