import logging
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Note: For advanced panel unit root tests like IPS/LLC or Pesaran CD,
# additional packages (like linearmodels or custom implementations) may be required.

class PanelDiagnostics:
    """
    Standardized pre-estimation checks for panel data models.
    Includes missing coverage, Variance Inflation Factors (VIF),
    and stubs for panel unit root / cross-sectional dependence tests.
    """
    
    def __init__(self, entity_col='country', time_col='date'):
        self.entity_col = entity_col
        self.time_col = time_col
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def missing_coverage(self, panel, cols, threshold=0.3):
        """
        Reports per-entity data coverage for specified columns.
        
        Args:
            panel (pd.DataFrame): The panel dataset.
            cols (list): Columns to check.
            threshold (float): Flag entities where missing % > threshold.
            
        Returns:
            pd.DataFrame: A summary of missing coverage by entity.
        """
        df = panel.copy()
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
            
        self.logger.info(f"Checking missing coverage for {len(cols)} columns across entities.")
        
        # Calculate percent missing by entity
        grouped = df.groupby(self.entity_col)
        
        rows = []
        for entity, group in grouped:
            n_total = len(group)
            missing_counts = group[cols].isna().sum()
            missing_pcts = missing_counts / n_total
            
            flagged = missing_pcts[missing_pcts > threshold].index.tolist()
            
            rows.append({
                'entity': entity,
                'n_obs': n_total,
                'avg_missing_pct': missing_pcts.mean(),
                'max_missing_pct': missing_pcts.max(),
                'flagged_cols': ", ".join(flagged) if flagged else "None"
            })
            
        summary = pd.DataFrame(rows).set_index('entity')
        return summary
        
    def vif_table(self, panel, cols):
        """
        Calculates Variance Inflation Factors for a set of regressors to check multicollinearity.
        
        Args:
            panel (pd.DataFrame): The dataset.
            cols (list): The independent variables to check.
            
        Returns:
            pd.DataFrame: A table of VIF values.
        """
        self.logger.info(f"Calculating VIF for {len(cols)} variables.")
        X = panel[cols].dropna()
        
        if X.empty:
            self.logger.warning("Empty dataframe after dropping NaNs for VIF calculation.")
            return pd.DataFrame()
            
        # Add a constant if not present, as VIF needs it
        if 'const' not in X.columns:
            import statsmodels.api as sm
            X = sm.add_constant(X)
            
        vif_data = []
        for i in range(X.shape[1]):
            try:
                vif = variance_inflation_factor(X.values, i)
            except Exception as e:
                vif = np.nan
            vif_data.append({
                'Variable': X.columns[i],
                'VIF': vif
            })
            
        df_vif = pd.DataFrame(vif_data)
        # Drop constant from output usually
        df_vif = df_vif[df_vif['Variable'] != 'const']
        return df_vif

    def unit_root_summary(self, panel, cols):
        """
        Placeholder for panel unit root tests (e.g., Im-Pesaran-Shin).
        """
        self.logger.info("Panel unit root summary requested (Placeholder).")
        # In a real implementation, this would use linearmodels or statsmodels
        # to loop through columns and run IPS tests.
        return pd.DataFrame(columns=['col', 'statistic', 'pval', 'verdict'])

    def cross_section_dependence(self, panel, cols):
        """
        Placeholder for Pesaran CD test for cross-sectional dependence.
        """
        self.logger.info("Cross-sectional dependence test requested (Placeholder).")
        return pd.DataFrame(columns=['col', 'CD_stat', 'pval'])

    def run(self, panel, cols):
        """
        Runs a suite of diagnostic checks and returns a dictionary of results.
        """
        self.logger.info("Running full suite of panel diagnostics.")
        results = {
            'missing_coverage': self.missing_coverage(panel, cols),
            'vif': self.vif_table(panel, cols),
            'unit_roots': self.unit_root_summary(panel, cols),
            'cross_section_dependence': self.cross_section_dependence(panel, cols)
        }
        return results
