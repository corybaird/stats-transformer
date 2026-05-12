import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stats_transformer.visualization.formatters.style import apply_style
from stats_transformer.visualization.defaults.labels import get_readable_label

class TimeSeriesPlot:
    """
    Single or multi-line time series with optional CI bands.
    """
    def __init__(self, style_path="timeseries"):
        self.style_path = style_path

    def plot(self, df, x_col, y_cols, ci_lower_cols=None, ci_upper_cols=None, labels=None, title=None, ylabel=None, ax=None, colors=None):
        if ax is None:
            apply_style(self.style_path)
            fig, ax = plt.subplots(figsize=(14, 7))
        else:
            fig = ax.figure

        if colors is None:
            from stats_transformer.visualization.defaults.colors import get_color_palette
            colors = get_color_palette("default", len(y_cols))

        x = df[x_col]
        
        for i, y_col in enumerate(y_cols):
            y = df[y_col]
            label = labels.get(y_col, get_readable_label(y_col)) if labels else get_readable_label(y_col)
            color = colors[i % len(colors)]
            
            ax.plot(x, y, label=label, color=color, linewidth=2)
            
            if ci_lower_cols and ci_upper_cols and i < len(ci_lower_cols) and i < len(ci_upper_cols):
                lower_col = ci_lower_cols[i]
                upper_col = ci_upper_cols[i]
                if lower_col in df.columns and upper_col in df.columns:
                    ax.fill_between(x, df[lower_col], df[upper_col], color=color, alpha=0.2)

        if ylabel:
            ax.set_ylabel(ylabel, size=16)
        if title:
            ax.set_title(title, size=18)
            
        ax.legend(loc='best', fontsize=10)
        
        return fig, ax

class IRFPlot:
    """
    Impulse Response Function: line + CI fill_between, faceted by variable.
    """
    def __init__(self, style_path="timeseries"):
        self.style_path = style_path

    def plot(self, df, horizon_col, coef_col, ci_lower_col, ci_upper_col, group_col, labels=None, title=None, ylabel=None, colors=None):
        apply_style(self.style_path)
        
        groups = df[group_col].unique().tolist()
        n_vars = len(groups)
        
        ncols = 2
        nrows = (n_vars + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharex=True)
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[np.newaxis, :]
        elif ncols == 1:
            axes = axes[:, np.newaxis]
            
        axes = axes.flatten()

        if colors is None:
            from stats_transformer.visualization.defaults.colors import PALETTE_CATEGORICAL
            colors = PALETTE_CATEGORICAL

        for idx, group in enumerate(groups):
            ax = axes[idx]
            sub = df[df[group_col] == group].sort_values(horizon_col)
            
            horizons = sub[horizon_col].values
            coefs = sub[coef_col].values
            ci_lower = sub[ci_lower_col].values
            ci_upper = sub[ci_upper_col].values
            
            color = colors[idx % len(colors)]
            label = labels.get(group, get_readable_label(group)) if labels else get_readable_label(group)
            
            ax.plot(horizons, coefs, marker='o', linewidth=2, color=color, markersize=5)
            ax.fill_between(horizons, ci_lower, ci_upper, alpha=0.2, color=color)
            
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            
            if idx >= n_vars - ncols:
                ax.set_xlabel('Horizon')
                
            if idx % ncols == 0:
                ax.set_ylabel(ylabel if ylabel else 'IRF Coefficient')
                
            ax.set_title(label)
            ax.set_xticks(horizons)

        for idx in range(n_vars, len(axes)):
            axes[idx].set_visible(False)
            
        if title:
            fig.suptitle(title, size=18, y=1.02)
            
        fig.tight_layout()
        
        return fig, axes

class FacetedTimeSeries:
    """
    Small multiples of time series, one per entity.
    """
    def __init__(self, style_path="timeseries"):
        self.style_path = style_path

    def plot(self, df, date_col, value_cols, facet_col, labels=None, ncols=3, title=None, ylabel=None, colors=None):
        apply_style(self.style_path)
        
        facets = df[facet_col].unique().tolist()
        n_facets = len(facets)
        
        nrows = (n_facets + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True)
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[np.newaxis, :]
        elif ncols == 1:
            axes = axes[:, np.newaxis]
            
        axes = axes.flatten()

        if colors is None:
            from stats_transformer.visualization.defaults.colors import get_color_palette
            colors = get_color_palette("default", len(value_cols))

        for idx, facet in enumerate(facets):
            ax = axes[idx]
            sub = df[df[facet_col] == facet].sort_values(date_col)
            
            x = sub[date_col]
            for i, y_col in enumerate(value_cols):
                y = sub[y_col]
                label = labels.get(y_col, get_readable_label(y_col)) if labels else get_readable_label(y_col)
                color = colors[i % len(colors)]
                
                ax.plot(x, y, label=label, color=color, linewidth=2)
            
            ax.set_title(str(facet))
            
            if idx >= n_facets - ncols:
                ax.set_xlabel('Date')
                
            if idx % ncols == 0:
                ax.set_ylabel(ylabel if ylabel else 'Value')

        for idx in range(n_facets, len(axes)):
            axes[idx].set_visible(False)
            
        if title:
            fig.suptitle(title, size=18, y=1.02)
            
        if n_facets > 0:
            axes[0].legend(loc='best', fontsize=9)
            
        fig.tight_layout()
        
        return fig, axes
