import logging
import pandas as pd
import numpy as np

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

    def export(self, sections, filename, **kwargs):
        """
        Builds the table and exports it using the provided TableGenerator.
        """
        df_stats = self.build(sections)
        
        if df_stats.empty:
            self.logger.warning("Generated table is empty. Nothing to export.")
            return None
            
        return self.table_gen.export_all(df_stats, filename, **kwargs)
