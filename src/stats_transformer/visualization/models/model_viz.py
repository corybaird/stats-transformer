import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stats_transformer.visualization.base import BaseVisualizer
from references.dictionaries.COLUMN_LABELS import get_readable_label

class ModelVisualizer(BaseVisualizer):

    def __init__(self, params_path=None, output_dir="reports/visualizations", file_format="png", dpi=300, style="default"):
        super().__init__(params_path, output_dir, file_format, dpi, style)
    
    def load_model_from_json(self, json_path):
        try:
            with open(json_path, "r") as f:
                model_data = json.load(f)
            self.logger.info(f"Loaded model data from {json_path}")
            return model_data
        except Exception as e:
            self.logger.error(f"Error loading model data: {str(e)}")
            raise
    
    def create_coefficient_plot_from_json(self, model_summary, subdir="regression", display_only=False):
        self.logger.info("Creating coefficient plot from JSON")
        if 'coefficients' in model_summary:
            coefficients = model_summary['coefficients']
        elif 'params' in model_summary:
            coefficients = {}
            for var, value in model_summary['params'].items():
                if type(value) in [int, float]:
                    coefficients[var] = {'value': value}
                elif type(value) == dict and 'value' in value:
                    coefficients[var] = value
        else:
            self.logger.warning("No coefficients found in model_summary")
            return None
        
        if not coefficients:
            self.logger.warning("Coefficients dictionary is empty")
            return None
        
        plot_data = []
        for var, coef_info in coefficients.items():
            value = coef_info.get('value', 0)
            std_err = coef_info.get('std_err', None)
            if var.lower() in ['const', 'intercept']:
                continue
            plot_data.append({'variable': var, 'coefficient': value, 'std_error': std_err if std_err else 0})
        
        if not plot_data:
            self.logger.warning("No non-intercept coefficients to plot")
            return None
        
        df = pd.DataFrame(plot_data)
        df['abs_coef'] = df['coefficient'].abs()
        df = df.sort_values('abs_coef', ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))
        y_pos = np.arange(len(df))
        bars = ax.barh(y_pos, df['coefficient'])
        
        if not df['std_error'].isnull().all():
            ax.errorbar(df['coefficient'], y_pos, xerr=df['std_error'], fmt='none', color='black', capsize=3)
        
        max_abs = max(df['coefficient'].abs())
        for i, (bar, row) in enumerate(zip(bars, df.itertuples())):
            width = bar.get_width()
            text_x = width + (0.02 * max_abs) if width >= 0 else width - (0.02 * max_abs)
            ha = 'left' if width >= 0 else 'right'
            coef_text = f"{row.coefficient:.3f}"
            if not pd.isna(row.std_error) and row.std_error > 0:
                coef_text += f" ± {row.std_error:.3f}"
            ax.text(text_x, bar.get_y() + bar.get_height()/2, coef_text, ha=ha, va='center', fontsize=9)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([get_readable_label(v) for v in df['variable']])
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlabel('Coefficient Value')
        ax.set_title('Regression Coefficients with Standard Errors', fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        return self.save_figure(fig, "coefficient_plot", subdir, display_only=display_only)
    
    def create_visualization(self, model_data, visualization_type="coefficient", **kwargs):
        if type(model_data) == str:
            model_data = self.load_model_from_json(model_data)
        if visualization_type == "coefficient":
            return self.create_coefficient_plot_from_json(model_data, **kwargs)
        return None

    def run(self):
        pass
