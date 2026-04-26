import os
import logging
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import sys

class BaseVisualizer(ABC):

    def __init__(self, params_path=None, output_dir="reports/visualizations", file_format="png", dpi=300, style="default"):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
        self.logger = logging.getLogger(__name__)
        self.params = {}
        
        if params_path:
            try:
                with open(params_path, "r") as f:
                    self.params = yaml.safe_load(f)
                self.logger.info(f"Loaded parameters from {params_path}")
            except Exception as e:
                self.logger.warning(f"Error loading params: {str(e)}")
        
        self.viz_params = self.params.get("visualization", {
            "output_dir": output_dir,
            "file_format": file_format,
            "dpi": dpi,
            "style": style
        })
            
        os.makedirs(self.viz_params.get("output_dir", output_dir), exist_ok=True)
        style_name = self.viz_params.get("style", style)
        try:
            plt.style.use(style_name)
        except Exception as e:
            plt.style.use("default")
            
        plt.rc('axes', grid=True)
        plt.rc('grid', linestyle='--', alpha=0.7)
        plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'Verdana'], 'font.size': 10})
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    
    @abstractmethod
    def create_visualization(self, *args, **kwargs):
        pass
    
    def run(self):
        pass

    def save_figure(self, fig, filename, subdir=None, display_only=False):
        if display_only:
            plt.tight_layout()
            plt.show(block=False)
            return None
            
        output_dir = self.viz_params.get("output_dir", "reports/visualizations")
        if subdir:
            output_dir = os.path.join(output_dir, subdir)
            os.makedirs(output_dir, exist_ok=True)
        
        formats = self.viz_params.get("formats", [self.viz_params.get("file_format", "png")])
        dpi = self.viz_params.get("dpi", 300)
        saved_paths = []
        for fmt in formats:
            filepath = os.path.join(output_dir, f"{filename}.{fmt}")
            try:
                fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches='tight')
                self.logger.info(f"Saved visualization to {filepath}")
                saved_paths.append(filepath)
            except Exception as e:
                self.logger.error(f"Error saving figure: {str(e)}")
        
        plt.close(fig)
        return saved_paths[0] if len(saved_paths) == 1 else saved_paths
