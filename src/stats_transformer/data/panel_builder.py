import logging
import pandas as pd

class PanelDataBuilder:
    """
    Utility class to standardize the assembly of panel data, 
    especially handling mixed-frequency joins (e.g., merging monthly 
    macroeconomic data onto a daily panel) and broadcasting global 
    series (e.g., VIX, oil prices) onto country panels.
    """
    
    def __init__(self, entity_col='country', time_col='date'):
        self.entity_col = entity_col
        self.time_col = time_col
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def join(self, panel, df, name, on_freq='M', cols=None, index_cols=None):
        """
        Merge a lower-frequency (or matched-frequency) dataframe onto a panel.
        
        Args:
            panel (pd.DataFrame): The base panel, indexed by [entity_col, time_col].
            df (pd.DataFrame): The dataset to merge.
            name (str): A descriptive name for the dataset being joined (for logging).
            on_freq (str): The frequency to align on before joining. 
                           E.g., 'M' means both the panel and df will have their dates 
                           converted to monthly periods to perform the match.
            cols (list): Specific columns from `df` to keep. If None, keeps all.
            index_cols (list): Columns in `df` that represent the entity and time. 
                               Defaults to [self.entity_col, self.time_col].
                               
        Returns:
            pd.DataFrame: The merged panel.
        """
        if df is None or df.empty:
            self.logger.warning(f"Dataset '{name}' is empty or None. Returning original panel.")
            return panel
            
        index_cols = index_cols or [self.entity_col, self.time_col]
        
        # Prepare the incoming dataframe
        df_clean = df.copy()
        
        # Un-index if necessary to work with columns
        if isinstance(df_clean.index, pd.MultiIndex):
            df_clean = df_clean.reset_index()
        elif df_clean.index.name in index_cols:
            df_clean = df_clean.reset_index()
            
        # Ensure index columns are present
        missing_cols = [c for c in index_cols if c not in df_clean.columns]
        if missing_cols:
            self.logger.warning(f"Cannot join '{name}': missing index columns {missing_cols}")
            return panel
            
        # Keep only required columns
        if cols:
            keep_cols = index_cols + [c for c in cols if c in df_clean.columns and c not in index_cols]
            df_clean = df_clean[keep_cols]
            
        # Drop overlapping columns that already exist in the panel (except the join keys)
        panel_cols = panel.columns.tolist() if not isinstance(panel.index, pd.MultiIndex) else panel.columns.tolist() + list(panel.index.names)
        overlap = [c for c in df_clean.columns if c in panel_cols and c not in index_cols]
        if overlap:
            df_clean = df_clean.drop(columns=overlap)
            
        if len(df_clean.columns) <= len(index_cols):
            self.logger.info(f"No new columns to join for '{name}'. Returning original panel.")
            return panel
            
        # Align frequencies
        panel_reset = panel.copy()
        is_multiindex_panel = isinstance(panel_reset.index, pd.MultiIndex)
        if is_multiindex_panel:
            panel_reset = panel_reset.reset_index()
            
        if self.time_col not in panel_reset.columns:
            self.logger.warning(f"Panel is missing time column '{self.time_col}'.")
            return panel
            
        # Create a temporary period column for merging
        temp_period_col = f'_period_merge_{on_freq}'
        
        # Convert panel dates to period
        panel_reset[self.time_col] = pd.to_datetime(panel_reset[self.time_col])
        panel_reset[temp_period_col] = panel_reset[self.time_col].dt.to_period(on_freq)
        
        # Convert incoming df dates to period
        # Assume the second element in index_cols is the time column
        df_time_col = index_cols[1]
        df_entity_col = index_cols[0]
        
        df_clean[df_time_col] = pd.to_datetime(df_clean[df_time_col])
        df_clean[temp_period_col] = df_clean[df_time_col].dt.to_period(on_freq)
        
        # Deduplicate incoming df based on entity and period (keep last)
        df_clean = df_clean.drop_duplicates(subset=[df_entity_col, temp_period_col], keep='last')
        
        # Prepare for merge
        df_clean = df_clean.rename(columns={df_entity_col: self.entity_col})
        df_clean = df_clean.drop(columns=[df_time_col])
        
        # Perform the merge
        merged = panel_reset.merge(
            df_clean,
            on=[self.entity_col, temp_period_col],
            how='left'
        )
        
        # Cleanup
        merged = merged.drop(columns=[temp_period_col])
        
        if is_multiindex_panel:
            merged = merged.set_index([self.entity_col, self.time_col])
            
        # Log successful merge metrics
        added_cols = [c for c in df_clean.columns if c not in [self.entity_col, temp_period_col]]
        not_null_counts = [f"{c}={merged[c].notna().sum()}" for c in added_cols if c in merged.columns]
        self.logger.info(f"Joined '{name}': {merged.shape}, non-null: {', '.join(not_null_counts)}")
        
        return merged

    def join_global(self, panel, df, name, time_col=None, cols=None):
        """
        Broadcast a global time series (e.g., VIX, oil) onto a panel where 
        the incoming dataset has no entity column, just dates.
        
        Args:
            panel (pd.DataFrame): The base panel.
            df (pd.DataFrame): The global dataset to merge.
            name (str): A descriptive name for logging.
            time_col (str): The column in `df` representing time. If None, defaults to `self.time_col`.
            cols (list): Specific columns from `df` to keep.
            
        Returns:
            pd.DataFrame: The merged panel.
        """
        if df is None or df.empty:
            self.logger.warning(f"Global dataset '{name}' is empty or None. Returning original panel.")
            return panel
            
        time_col = time_col or self.time_col
        df_clean = df.copy()
        
        if df_clean.index.name == time_col:
            df_clean = df_clean.reset_index()
            
        if time_col not in df_clean.columns:
            self.logger.warning(f"Cannot join global '{name}': missing time column '{time_col}'")
            return panel
            
        if cols:
            keep_cols = [time_col] + [c for c in cols if c in df_clean.columns and c != time_col]
            df_clean = df_clean[keep_cols]
            
        panel_cols = panel.columns.tolist() if not isinstance(panel.index, pd.MultiIndex) else panel.columns.tolist() + list(panel.index.names)
        overlap = [c for c in df_clean.columns if c in panel_cols and c != time_col]
        if overlap:
            df_clean = df_clean.drop(columns=overlap)
            
        if len(df_clean.columns) <= 1:
            self.logger.info(f"No new columns to join for global '{name}'. Returning original panel.")
            return panel
            
        df_clean[time_col] = pd.to_datetime(df_clean[time_col])
        df_clean = df_clean.drop_duplicates(subset=[time_col], keep='last')
        
        panel_reset = panel.copy()
        is_multiindex_panel = isinstance(panel_reset.index, pd.MultiIndex)
        if is_multiindex_panel:
            panel_reset = panel_reset.reset_index()
            
        panel_reset[self.time_col] = pd.to_datetime(panel_reset[self.time_col])
        
        # If the incoming time_col is named differently than the panel's time_col, align them
        if time_col != self.time_col:
            df_clean = df_clean.rename(columns={time_col: self.time_col})
            
        # Left merge based on exact date match
        # Note: If forward-filling (asof join) is desired instead of exact match, 
        # that would require a separate method or flag (pd.merge_asof).
        merged = panel_reset.merge(
            df_clean,
            on=self.time_col,
            how='left'
        )
        
        if is_multiindex_panel:
            merged = merged.set_index([self.entity_col, self.time_col])
            
        added_cols = [c for c in df_clean.columns if c != self.time_col]
        not_null_counts = [f"{c}={merged[c].notna().sum()}" for c in added_cols if c in merged.columns]
        self.logger.info(f"Joined global '{name}': {merged.shape}, non-null: {', '.join(not_null_counts)}")
        
        return merged

    def build(self, base_panel, join_specs):
        """
        Fluent-style builder to chain multiple joins.
        
        Args:
            base_panel (pd.DataFrame): The initial panel.
            join_specs (list of dicts): E.g.
                [
                    {"type": "entity", "df": cpi_df, "name": "CPI", "on_freq": "M", "cols": ["infl"]},
                    {"type": "global", "df": vix_df, "name": "VIX", "cols": ["vix_close"]}
                ]
        Returns:
            pd.DataFrame: Fully built panel.
        """
        panel = base_panel.copy()
        
        for spec in join_specs:
            j_type = spec.get("type", "entity")
            if j_type == "entity":
                panel = self.join(
                    panel,
                    spec.get("df"),
                    spec.get("name"),
                    on_freq=spec.get("on_freq", "M"),
                    cols=spec.get("cols"),
                    index_cols=spec.get("index_cols")
                )
            elif j_type == "global":
                panel = self.join_global(
                    panel,
                    spec.get("df"),
                    spec.get("name"),
                    time_col=spec.get("time_col"),
                    cols=spec.get("cols")
                )
        return panel
