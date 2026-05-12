import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stats_transformer.visualization.base import BaseVisualizer
from stats_transformer.visualization.defaults.labels import get_readable_label
from stats_transformer.visualization.charts.bar import CoefficientBarChart

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
            p_val = coef_info.get('p_value', np.nan)
            if var.lower() in ['const', 'intercept']:
                continue
            plot_data.append({'variable': var, 'coefficient': value, 'std_error': std_err if std_err else 0, 'p_value': p_val})
        
        if not plot_data:
            self.logger.warning("No non-intercept coefficients to plot")
            return None
        
        df = pd.DataFrame(plot_data)
        df['abs_coef'] = df['coefficient'].abs()
        df = df.sort_values('abs_coef', ascending=True)
        
        chart = CoefficientBarChart(style_path=self.viz_params.get("style", "barchart"))
        fig, ax = chart.plot(
            labels=df['variable'].tolist(),
            coefficients=df['coefficient'].tolist(),
            std_errors=df['std_error'].tolist(),
            p_values=df['p_value'].tolist(),
            title='Regression Coefficients with Standard Errors',
            ylabel='Coefficient Value'
        )
        
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
