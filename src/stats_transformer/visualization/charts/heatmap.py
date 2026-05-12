import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from stats_transformer.visualization.formatters.style import apply_style
from stats_transformer.visualization.defaults.labels import get_readable_label

class CorrelationHeatmap:
    """
    Correlation matrix heatmap, optionally lower-triangle only.
    """
    def __init__(self, style_path="default"):
        self.style_path = style_path

    def plot(self, df, columns=None, method="pearson", labels=None, title=None, mask_upper=True, ax=None):
        if ax is None:
            apply_style(self.style_path)
            size = max(8, len(columns) if columns else len(df.columns))
            fig, ax = plt.subplots(figsize=(size, size * 0.8))
        else:
            fig = ax.figure

        if columns:
            df = df[columns]
            
        corr_matrix = df.corr(method=method)
        
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
        display_labels = [labels.get(c, get_readable_label(c)) if labels else get_readable_label(c) for c in corr_matrix.columns]

        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap="coolwarm", 
            center=0, 
            linewidths=0.5, 
            ax=ax, 
            fmt=".2f", 
            xticklabels=display_labels, 
            yticklabels=display_labels,
            mask=mask,
            cbar_kws={"shrink": .8}
        )
        
        if title is None:
            title = f"{method.capitalize()} Correlation Matrix"
            
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        fig.tight_layout()

        return fig, ax
