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
        
        # Escape specific characters for LaTeX (is_string_dtype covers object
        # on pandas 2 and the new str dtype on pandas 3)
        for col in [c for c in tbl.columns if pd.api.types.is_string_dtype(tbl[c])]:
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

        # Rendered by hand instead of DataFrame.to_latex: pandas routes to_latex
        # through Styler, which drags in jinja2. Keeps booktabs rules
        # (toprule/midrule/bottomrule) — compile with \usepackage{booktabs}.
        body = [
            " & ".join("" if pd.isna(v) else str(v) for v in row) + r" \\"
            for row in tbl.itertuples(index=False, name=None)
        ]

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

        lines = [r'\begin{tabular}{' + col_format + '}', r'\toprule', header_line, r'\midrule']
        lines.extend(body)
        lines.extend([r'\bottomrule', r'\end{tabular}'])

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
