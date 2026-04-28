import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.stats_transformer.visualization.base import BaseVisualizer

class EDAVisualizer(BaseVisualizer):
    # Automates initial data inspection sequence (missingness, distributions)

    def __init__(self, params_path=None, output_dir="reports/visualizations", file_format="png", dpi=300, style="default"):
        super().__init__(params_path=params_path, output_dir=output_dir, file_format=file_format, dpi=dpi, style=style)

    def _get_numeric_cols(self, df):
        return df.select_dtypes(include=[np.number]).columns.tolist()

    def plot_missingness(self, df, subdir="eda", display_only=False):
        # Plots a missing value bar chart and heatmap
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        missing_pct = (df.isnull().sum() / len(df)) * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
        
        if not missing_pct.empty:
            sns.barplot(x=missing_pct.values, y=missing_pct.index, ax=axes[0], color='salmon')
            axes[0].set_title("Missing Values Percentage by Column")
            axes[0].set_xlabel("% Missing")
            
            sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=axes[1])
            axes[1].set_title("Missing Values Heatmap")
        else:
            axes[0].text(0.5, 0.5, "No missing values", ha='center', va='center')
            axes[1].text(0.5, 0.5, "No missing values", ha='center', va='center')
            
        fig.tight_layout()
        return self.save_figure(fig, "missingness_profile", subdir=subdir, display_only=display_only)

    def plot_distributions(self, df, subdir="eda", display_only=False):
        # Plots histograms and KDE estimates for numeric variables
        numeric_cols = self._get_numeric_cols(df)
        if not numeric_cols:
            self.logger.warning("No numeric columns found for distributions.")
            return []
            
        n_cols = 3
        n_rows = int(np.ceil(len(numeric_cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color='steelblue')
                axes[i].set_title(f"Distribution of {col}")
                axes[i].set_xlabel("")
                
        for j in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[j])
            
        fig.tight_layout()
        return self.save_figure(fig, "numeric_distributions", subdir=subdir, display_only=display_only)

    def create_visualization(self, data, viz_type="all", subdir="eda", display_only=False):
        # Dispatcher for EDA charts
        if isinstance(data, str):
            data = pd.read_csv(data) if data.endswith(".csv") else pd.read_parquet(data)
            
        results = {}
        if viz_type in ["all", "missingness"]:
            results["missingness"] = self.plot_missingness(data, subdir=subdir, display_only=display_only)
        if viz_type in ["all", "distributions"]:
            results["distributions"] = self.plot_distributions(data, subdir=subdir, display_only=display_only)
            
        return results

    def run(self, data_path=None, output_path=None):
        # Orchestrates the EDA pipeline phase
        if not data_path and self.params:
            data_config = self.params.get("data", {})
            data_path = data_config.get("featurization", {}).get("output_path", data_config.get("raw_data_file"))
            
        if not data_path:
            raise ValueError("No data_path provided for EDA")
            
        self.logger.info(f"Running EDA on {data_path}")
        df = pd.read_csv(data_path) if data_path.endswith(".csv") else pd.read_parquet(data_path)
        return self.create_visualization(df, viz_type="all")
