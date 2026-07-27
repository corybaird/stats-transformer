import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from stats_transformer.models.timeseries.identification.sign_zero import SignZeroSVARModel

class SVARBootstrap:
    """
    Implements a frequentist residual bootstrap for Set-Identified SVAR models.
    """
    
    def __init__(self, svar_model, n_bootstrap=100, seed=42):
        if not isinstance(svar_model, SignZeroSVARModel):
            raise ValueError("svar_model must be an instance of SignZeroSVARModel")
            
        self.svar_model = svar_model
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.bootstrap_results = []
        
    def run(self):
        """
        Runs the residual bootstrap procedure.
        1. Fit the reduced form VAR
        2. Generate bootstrap samples of residuals
        3. Construct bootstrap data series
        4. Re-fit VAR on bootstrap data
        5. Run the restriction sampler to find accepted draws
        6. Store the representative draw (median target) for the bootstrap sample
        """
        np.random.seed(self.seed)
        
        var_res = self.svar_model.var_result
        if var_res is None:
            raise ValueError("The SVAR model must be fitted before running bootstrap.")
            
        y = self.svar_model.y.values
        T, K = y.shape
        p = var_res.k_ar
        
        # Residuals
        resid = var_res.resid.values if isinstance(var_res.resid, pd.DataFrame) else var_res.resid
        T_res = resid.shape[0]
        
        # Coefficients
        coefs = var_res.coefs # (p, K, K)
        intercept = var_res.intercept if hasattr(var_res, 'intercept') else np.zeros(K)
        
        self.bootstrap_results = []
        
        for b in range(self.n_bootstrap):
            # Resample residuals with replacement
            idx = np.random.randint(0, T_res, size=T_res)
            u_star = resid[idx]
            
            # Reconstruct series y_star
            y_star = np.zeros_like(y)
            # Use original data for initial conditions
            y_star[:p] = y[:p]
            
            for t in range(p, T):
                y_star[t] = intercept.copy()
                for i in range(p):
                    y_star[t] += coefs[i] @ y_star[t - i - 1]
                y_star[t] += u_star[t - p]
                
            # Create a new DataFrame for the bootstrap sample
            df_star = pd.DataFrame(y_star, columns=self.svar_model.target_variables)
            if self.svar_model.date_column:
                df_star[self.svar_model.date_column] = self.svar_model.df_clean[self.svar_model.date_column].values
                
            # Create a new SVAR model instance to evaluate restrictions
            b_model = SignZeroSVARModel(
                target_variables=self.svar_model.target_variables,
                config_path=self.svar_model.config_path,
                date_column=self.svar_model.date_column,
                maxlags=self.svar_model.maxlags,
                max_draws=self.svar_model.max_draws,
                required_accepts=self.svar_model.required_accepts
            )
            
            # Fit and evaluate
            b_model.df_clean = df_star
            try:
                b_model.build_model()
                if b_model.accepted_irfs:
                    rep_draw = b_model.get_representative_draw()
                    self.bootstrap_results.append(rep_draw)
            except Exception as e:
                # If a bootstrap sample fails to find a valid draw, we skip it
                pass
                
        return self.bootstrap_results

    def get_confidence_intervals(self, alpha=0.05):
        """
        Computes pointwise confidence intervals for the IRFs across bootstrap samples.
        """
        if not self.bootstrap_results:
            raise ValueError("Bootstrap has not been run or no accepted draws were found.")
            
        # Extract IRFs from representative draws
        irfs = np.array([res["irf"] for res in self.bootstrap_results])
        
        lower = np.percentile(irfs, 100 * (alpha / 2), axis=0)
        upper = np.percentile(irfs, 100 * (1 - alpha / 2), axis=0)
        
        return lower, upper
