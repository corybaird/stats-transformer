import pandas as pd
from pathlib import Path
from stats_transformer.models.timeseries.var import VARModel

class GhyselsChap6VARExample:
    def __init__(self):
        self.data_path = Path("data/examples/timeseries/ghysels_ch6/var_simulated_ch6_sec9.csv")

    def run(self):
        sim_data = pd.read_csv(self.data_path)
        
        # VAR requires multiple target variables, in Ch_6 data has x and y
        model = VARModel(target_variables=['x', 'y'], maxlags=2)
        model.fit(sim_data)
        
        print("VAR Model Summary:")
        print(model.get_summary())

if __name__ == "__main__":
    runner = GhyselsChap6VARExample()
    runner.run()
