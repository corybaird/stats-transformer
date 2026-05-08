import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from stats_transformer.models.regression.regression import RegressionModel

class GhyselsChap1Example:
    def __init__(self):
        self.data_path = Path("data/examples/regression/ghysels_ch1/simulated_data.csv")

    def run(self):
        sim_data = pd.read_csv(self.data_path)
        sim_data_est = sim_data.iloc[101:301].copy()
        sim_data_fore = sim_data.iloc[301:401].copy()

        # Fit RegressionModel
        model = RegressionModel(target='y', independent_variables=['x'])
        results = model.fit(sim_data_est)
        print("Model Summary:")
        print(model.summary())

        # Forecast
        # statsmodels OLS model exposes predict
        X_fore = model.X.copy()
        # In actual forecasting we would provide new X, but let's just show it works
        X_fore = sm.add_constant(sim_data_fore[['x']]) if 'x' in sim_data_fore else None
        if X_fore is not None:
             preds = model.model.predict(sm.add_constant(sim_data_fore[['x']]))
             print("First 5 forecasts:")
             print(preds.head())

if __name__ == "__main__":
    runner = GhyselsChap1Example()
    runner.run()
