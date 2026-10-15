import logging
import pandas as pd
import numpy as np

from stats_transformer.reporting.formatters.significance import format_number
from stats_transformer.reporting.tables.table_generator import TableGenerator

class DescriptiveStatsTable:
    """
    Standardized summary statistics generator for "Table 1" format in academic papers.
    Calculates N, Mean, Std, Min, Max, and Percentiles, organized into labeled sections.
    Integrates with TableGenerator to export to LaTeX or Excel.
    """

    def __init__(self, table_generator):
        self.table_gen = table_generator
        self.logger = logging.getLogger(self.__class__.__name__)

    def _compute_stats(self, series):
        """Helper to compute statistics for a single series."""
        s = series.dropna()
        if s.empty:
            return pd.Series({
                'N': 0, 'Mean': np.nan, 'Std Dev': np.nan, 
                'Min': np.nan, 'p25': np.nan, 'p50': np.nan, 
                'p75': np.nan, 'Max': np.nan
            })
            
        return pd.Series({
            'N': len(s),
            'Mean': s.mean(),
            'Std Dev': s.std(),
            'Min': s.min(),
            'p25': s.quantile(0.25),
            'p50': s.quantile(0.50),
            'p75': s.quantile(0.75),
            'Max': s.max()
        })

    def build(self, sections):
        """
        Builds a descriptive statistics DataFrame organized by sections.
        
        Args:
            sections (list of dicts): Format should be:
                [
                    {
                        'label': 'Dependent Variables',
                        'vars': [
                            ('original_col_name', 'Display Label', df_reference)
                        ]
                    },
                    ...
                ]
                
        Returns:
            pd.DataFrame: A formatted dataframe with section headers as empty rows.
        """
        self.logger.info(f"Building descriptive stats table with {len(sections)} sections.")
        
        rows = []
        for section in sections:
            # Add section header
            section_label = section.get('label', '')
            rows.append({
                'Variable': f"\\textit{{{section_label}}}",
                'N': '', 'Mean': '', 'Std Dev': '', 
                'Min': '', 'p25': '', 'p50': '', 'p75': '', 'Max': ''
            })
            
            for var_tuple in section.get('vars', []):
                if len(var_tuple) == 3:
                    col_name, display_label, df = var_tuple
                else:
                    self.logger.warning(f"Invalid variable tuple format: {var_tuple}. Expected (col_name, display_label, df).")
                    continue
                    
                if col_name not in df.columns:
                    self.logger.warning(f"Column '{col_name}' not found in dataframe for '{display_label}'. Skipping.")
                    continue
                    
                stats = self._compute_stats(df[col_name])
                
                # Format the numbers nicely
                row_data = {
                    'Variable': display_label,
                    'N': f"{int(stats['N']):,}",
                    'Mean': f"{stats['Mean']:.3f}" if pd.notna(stats['Mean']) else "",
                    'Std Dev': f"{stats['Std Dev']:.3f}" if pd.notna(stats['Std Dev']) else "",
                    'Min': f"{stats['Min']:.3f}" if pd.notna(stats['Min']) else "",
                    'p25': f"{stats['p25']:.3f}" if pd.notna(stats['p25']) else "",
                    'p50': f"{stats['p50']:.3f}" if pd.notna(stats['p50']) else "",
                    'p75': f"{stats['p75']:.3f}" if pd.notna(stats['p75']) else "",
                    'Max': f"{stats['Max']:.3f}" if pd.notna(stats['Max']) else ""
                }
                rows.append(row_data)
                
        final_df = pd.DataFrame(rows)
        return final_df

    @staticmethod
    def from_dataframe(df, variables=None):
        """
        Numeric N/Mean/SD/Min/Max table for each numeric variable of `df`.
        
        Args:
            df (pd.DataFrame): Source data (only numeric dtype columns contribute).
            variables (list, optional): Column names to include; defaults to all numeric columns.
            
        Returns:
            pd.DataFrame: One row per variable, columns [Variable, N, Mean, SD, Min, Max],
                unformatted so callers can compare directly against pandas describe().
        """
        cols = ["Variable", "N", "Mean", "SD", "Min", "Max"]
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)
        selected = list(variables) if variables else list(df.select_dtypes(include="number").columns)
        rows = []
        for col in selected:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            s = df[col].dropna()
            rows.append({
                "Variable": col,
                "N": len(s),
                "Mean": s.mean() if not s.empty else np.nan,
                "SD": s.std() if not s.empty else np.nan,
                "Min": s.min() if not s.empty else np.nan,
                "Max": s.max() if not s.empty else np.nan,
            })
        return pd.DataFrame(rows, columns=cols)

    def export_dataframe(self, df, filename, variables=None):
        """
        Builds the from_dataframe() table and exports it via TableGenerator (LaTeX).
        Returns the written path, or None when there is nothing to export.
        """
        stats = self.from_dataframe(df, variables)
        if stats.empty:
            self.logger.warning("Generated table is empty. Nothing to export.")
            return None
        formatted = stats.copy()
        formatted["N"] = formatted["N"].apply(lambda n: f"{int(n):,}")
        for col in ["Mean", "SD", "Min", "Max"]:
            formatted[col] = formatted[col].apply(format_number)
        return self.table_gen.export_latex(formatted, filename)

    def export(self, sections, filename, **kwargs):
        """
        Builds the table and exports it using the provided TableGenerator.
        """
        df_stats = self.build(sections)
        
        if df_stats.empty:
            self.logger.warning("Generated table is empty. Nothing to export.")
            return None
            
        return self.table_gen.export_all(df_stats, filename, **kwargs)
