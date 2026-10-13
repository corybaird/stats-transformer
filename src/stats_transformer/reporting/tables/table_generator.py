import os
from pathlib import Path
import pandas as pd
import logging

class TableGenerator:
    """
    A utility class for generating, formatting, and exporting tables 
    (Excel and LaTeX) for academic and econometric reporting.
    """
    def __init__(self, output_dir="reports/tables"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

    def clean_dataframe_for_export(self, df):
        """Clean dataframe types and escape characters for LaTeX."""
        tbl = df.copy()
        
        # Handle Floats -> Ints if they represent counts/whole numbers
        for col in tbl.select_dtypes(include=['float']).columns:
            # Only convert to int if they actually are whole numbers
            if tbl[col].dropna().apply(lambda x: x.is_integer()).all():
                tbl[col] = tbl[col].fillna(0).astype(int)
        
        # Escape specific characters for LaTeX
        for col in tbl.select_dtypes(include=['object']).columns:
            tbl[col] = tbl[col].astype(str).str.replace('&', r'\&', regex=False)
            tbl[col] = tbl[col].str.replace('%', r'\%', regex=False)
            tbl[col] = tbl[col].str.replace('_', r'\_', regex=False)
            
        return tbl

    def export_latex(self, df, filename, col_format=None, align_first_n_left=1):
        """
        Export a DataFrame to a styled LaTeX table with multi-line headers.
        
        Args:
            df (pd.DataFrame): The dataframe to export.
            filename (str): Name of the file (e.g., 'table1').
            col_format (str): Custom column format (e.g., 'llcccccc'). If None, generated automatically.
            align_first_n_left (int): If col_format is None, how many of the first columns to align left 'l'.
        """
        tbl = self.clean_dataframe_for_export(df)
        
        # Generate Base Body (No Header)
        latex = tbl.to_latex(index=False, escape=False, header=False)
        
        # Clean up default pandas rules to match target style
        latex = latex.replace(r'\toprule', '').replace(r'\midrule', '').replace(r'\bottomrule', r'\hline\hline')
        
        # Generate the exact Header format
        if not col_format:
            col_format = 'l' * align_first_n_left + 'c' * (len(tbl.columns) - align_first_n_left)
        
        header_parts = []
        for col in tbl.columns:
            # Convert newlines in column name to LaTeX line breaks
            col_str = str(col).replace('\n', r'\\')
            
            # If the column has a line break, wrap it in a nested tabular
            if r'\\' in col_str:
                wrapped = r'\multicolumn{1}{c}{\begin{tabular}[c]{@{}c@{}}' + col_str + r'\end{tabular}}'
                header_parts.append(wrapped)
            else:
                header_parts.append(col_str)
        
        # Join headers with alignment tabs and add the row end command
        header_line = ' & '.join(header_parts) + r'\\'
        
        # Inject into the LaTeX string
        lines = latex.split('\n')
        
        try:
            begin_idx = next(i for i, line in enumerate(lines) if r'\begin{tabular}' in line)
            
            # Replace the default column format with the custom one
            lines[begin_idx] = r'\begin{tabular}{' + col_format + '}'
            
            # Insert the specific border/header sequence
            lines.insert(begin_idx + 1, r'\hline\hline')
            lines.insert(begin_idx + 2, header_line)
            lines.insert(begin_idx + 3, r'\hline')
            
        except StopIteration:
            pass # Tabular start not found

        latex = '\n'.join(lines)
        
        if not str(filename).endswith('.tex'):
            filename = f"{filename}.tex"
            
        out_path = self.output_dir / filename
        with open(out_path, 'w') as f:
            f.write(latex)
            
        self.logger.info(f"LaTeX table saved to {out_path}")
        return out_path

    def export_excel(self, df, filename):
        """Export a DataFrame to Excel."""
        if not str(filename).endswith('.xlsx'):
            filename = f"{filename}.xlsx"
            
        out_path = self.output_dir / filename
        df.to_excel(out_path, index=False)
        self.logger.info(f"Excel table saved to {out_path}")
        return out_path

    def export_all(self, df, filename, **kwargs):
        """Export a DataFrame to both LaTeX and Excel."""
        tex_path = self.export_latex(df, filename, **kwargs)
        xlsx_path = self.export_excel(df, filename)
        return {"tex": tex_path, "xlsx": xlsx_path}
