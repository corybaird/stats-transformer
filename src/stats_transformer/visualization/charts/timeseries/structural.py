import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class RestrictionHeatmap:
    """
    Visualizes the restriction mapping (variables vs shocks).
    """
    def __init__(self, restrictions, variables, shocks):
        self.restrictions = restrictions
        self.variables = variables
        self.shocks = shocks
        
    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
            
        # Create mapping matrix
        # 1 for +, -1 for -, 0 for 0, nan for unrestricted
        mapping = np.full((len(self.variables), len(self.shocks)), np.nan)
        
        var_idx = {v: i for i, v in enumerate(self.variables)}
        shock_idx = {s: i for i, s in enumerate(self.shocks)}
        
        for r in self.restrictions:
            if r.get("horizon", 0) == 0: # Only plot impact matrix restrictions for now
                v = r["response"]
                s = r["shock"]
                if v in var_idx and s in shock_idx:
                    if r["type"] == "sign":
                        mapping[var_idx[v], shock_idx[s]] = 1 if r["value"] == "+" else -1
                    elif r["type"] == "zero":
                        mapping[var_idx[v], shock_idx[s]] = 0
                        
        cmap = sns.color_palette("vlag", as_cmap=True)
        sns.heatmap(mapping, annot=True, cmap=cmap, cbar=False, ax=ax, 
                    xticklabels=self.shocks, yticklabels=self.variables, 
                    mask=np.isnan(mapping), center=0)
        
        ax.set_title("Impact Matrix Restrictions (h=0)")
        return ax

class SwathePlot:
    """
    Visualizes the distribution of accepted IRFs (pointwise quantiles)
    overlaid with the representative draw.
    """
    def __init__(self, bootstrap_results, representative_draw, variables, shocks):
        self.bootstrap_results = bootstrap_results
        self.representative_draw = representative_draw
        self.variables = variables
        self.shocks = shocks
        
    def plot(self, axes=None):
        K = len(self.variables)
        S = len(self.shocks)
        
        if axes is None:
            fig, axes = plt.subplots(K, S, figsize=(4 * S, 3 * K), sharex=True)
            if K == 1 and S == 1:
                axes = np.array([[axes]])
            elif K == 1:
                axes = axes[np.newaxis, :]
            elif S == 1:
                axes = axes[:, np.newaxis]
                
        # Extract IRFs
        irfs = np.array([res["irf"] for res in self.bootstrap_results]) # (B, H, K, S)
        # We need (B, H, K_vars, K_shocks)
        # The irf shape is (H, K, K) where K is number of variables.
        # Assuming number of shocks == number of variables for this plot structure
        
        lower_16 = np.percentile(irfs, 16, axis=0)
        upper_84 = np.percentile(irfs, 84, axis=0)
        lower_5 = np.percentile(irfs, 5, axis=0)
        upper_95 = np.percentile(irfs, 95, axis=0)
        
        rep_irf = self.representative_draw["irf"]
        
        H = rep_irf.shape[0]
        horizons = np.arange(H)
        
        for i, var in enumerate(self.variables):
            for j, shock in enumerate(self.shocks):
                ax = axes[i, j]
                
                ax.fill_between(horizons, lower_5[:, i, j], upper_95[:, i, j], color='blue', alpha=0.1, label='90% CI')
                ax.fill_between(horizons, lower_16[:, i, j], upper_84[:, i, j], color='blue', alpha=0.2, label='68% CI')
                ax.plot(horizons, rep_irf[:, i, j], color='black', linewidth=2, label='Median Target')
                ax.axhline(0, color='red', linestyle='--', linewidth=1)
                
                if i == 0:
                    ax.set_title(f"Shock: {shock}")
                if j == 0:
                    ax.set_ylabel(f"Response: {var}")
                    
        return axes
