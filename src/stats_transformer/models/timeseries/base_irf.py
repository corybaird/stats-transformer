import logging
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class BaseIRFEstimator(ABC):
    """
    Abstract base class for Impulse Response Function estimators.
    Provides a standard interface for generating IRFs from models like
    Local Projections (LP-OLS) or Vector Autoregression (VAR).
    """

    def __init__(self, max_horizon=8, ci_level=0.95):
        self.max_horizon = max_horizon
        self.ci_level = ci_level
        self.results = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def estimate(self, panel, shock_var, response_vars, controls=None):
        """
        Estimate the impulse response function.
        
        Args:
            panel (pd.DataFrame): Panel data.
            shock_var (str): The variable representing the shock.
            response_vars (list): Variables for which to compute the response.
            controls (list, optional): Control variables.
            
        Returns:
            dict: {response_var: pd.DataFrame(horizon, coef, ci_lower, ci_upper, pval)}
        """
        pass

    def run(self, panel, shock_var, response_vars, controls=None):
        """
        Run the estimation and store results.
        """
        self.logger.info(f"Running IRF estimation for {len(response_vars)} response variables up to horizon {self.max_horizon}")
        self.results = self.estimate(panel, shock_var, response_vars, controls)
        return self.results

    def plot(self, response_var, ax=None, title=None, ylabel=None):
        """
        Generic plot for an IRF.
        """
        if not self.results or response_var not in self.results:
            self.logger.warning(f"No results found for {response_var}")
            return None

        df_irf = self.results[response_var]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        ax.plot(df_irf['horizon'], df_irf['coef'], color='blue', linewidth=2, label='IRF')
        ax.fill_between(
            df_irf['horizon'], 
            df_irf['ci_lower'], 
            df_irf['ci_upper'], 
            color='blue', alpha=0.2, label=f'{int(self.ci_level*100)}% CI'
        )
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        ax.set_title(title or f"Response of {response_var} to Shock")
        ax.set_xlabel("Horizon")
        ax.set_ylabel(ylabel or "Response")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

        return fig, ax
