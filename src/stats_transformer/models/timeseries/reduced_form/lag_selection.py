import pandas as pd
from statsmodels.tsa.api import VAR

class VARLagSelector:
    """
    Selects optimal lag order for VAR models using information criteria.
    Matches the functionality of R's vars::VARselect.
    """
    def __init__(self, target_variables, maxlags=10, trend="c", date_column=None):
        self.target_variables = target_variables
        self.maxlags = maxlags
        self.trend = trend
        self.date_column = date_column
        self.selection_results = None
        self.criteria_history = None

    def fit(self, df):
        required_cols = list(self.target_variables)
        if self.date_column and self.date_column in df.columns:
            df = df.sort_values(self.date_column)
        
        y = df[required_cols].dropna()
        var_model = VAR(y)
        
        # We use trend="c" by default (constant) which matches R vars::VARselect type="const"
        results = var_model.select_order(maxlags=self.maxlags, trend=self.trend)
        
        # Store selected orders
        self.selection_results = results.selected_orders
        
        # Reconstruct the criteria history into a DataFrame
        aic_vals = results.ics["aic"]
        bic_vals = results.ics["bic"]
        fpe_vals = results.ics["fpe"]
        hqic_vals = results.ics["hqic"]
        
        self.criteria_history = pd.DataFrame({
            "aic": aic_vals,
            "bic": bic_vals,
            "fpe": fpe_vals,
            "hqic": hqic_vals
        })
        self.criteria_history.index.name = "lag"
        
        return self.selection_results
