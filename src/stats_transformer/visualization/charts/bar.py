import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stats_transformer.visualization.formatters.style import apply_style
from stats_transformer.visualization.formatters.significance import annotate_stars
from stats_transformer.visualization.defaults.labels import get_readable_label

class CoefficientBarChart:
    """
    Single group of coefficients with significance stars and CI error bars.
    """
    def __init__(self, style_path="barchart"):
        self.style_path = style_path

    def plot(self, labels, coefficients, std_errors, p_values, ylabel=None, footer=None, title=None, ax=None, color='#2196F3', width=0.5):
        if ax is None:
            apply_style(self.style_path)
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure

        x = np.arange(len(labels))
        
        valid = [not pd.isna(c) for c in coefficients]
        v_pos = x[valid]
        v_coefs = [coefficients[i] for i in range(len(coefficients)) if valid[i]]
        v_ses = [std_errors[i] for i in range(len(std_errors)) if valid[i]]
        v_pvals = [p_values[i] for i in range(len(p_values)) if valid[i]]

        ax.bar(
            v_pos, v_coefs, width, 
            yerr=v_ses, capsize=4, 
            color=color, edgecolor='black', linewidth=0.5, alpha=0.85
        )

        annotate_stars(ax, v_pos, v_coefs, v_pvals)

        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
        ax.set_xticks(x)
        ax.set_xticklabels([get_readable_label(l) for l in labels])
        
        if ylabel:
            ax.set_ylabel(ylabel, size=16)
        if title:
            ax.set_title(title, size=18)
            
        if footer:
            ax.text(0.99, 0.01, footer, transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color='grey')

        return fig, ax

class GroupedBarChart:
    """
    Multiple groups side-by-side (e.g., conventional vs information shocks).
    """
    def __init__(self, style_path="barchart"):
        self.style_path = style_path

    def plot(self, df, x_col, y_col, group_col, error_col=None, pval_col=None, labels=None, ylabel=None, title=None, footer=None, ax=None, colors=None):
        if ax is None:
            apply_style(self.style_path)
            fig, ax = plt.subplots(figsize=(14, 6))
        else:
            fig = ax.figure

        if colors is None:
            from stats_transformer.visualization.defaults.colors import PALETTE_CATEGORICAL
            colors = PALETTE_CATEGORICAL

        x_vals = df[x_col].unique().tolist()
        groups = df[group_col].unique().tolist()
        
        n_x = len(x_vals)
        n_groups = len(groups)
        
        x = np.arange(n_x)
        total_width = 0.8
        width = total_width / n_groups
        offsets = np.linspace(-(n_groups - 1) / 2 * width, (n_groups - 1) / 2 * width, n_groups)

        for i, group in enumerate(groups):
            sub = df[df[group_col] == group]
            coefs, ses, pvals = [], [], []
            for xv in x_vals:
                row = sub[sub[x_col] == xv]
                if row.empty:
                    coefs.append(np.nan)
                    ses.append(0)
                    pvals.append(np.nan)
                else:
                    coefs.append(row.iloc[0][y_col])
                    ses.append(row.iloc[0][error_col] if error_col else 0)
                    pvals.append(row.iloc[0][pval_col] if pval_col else np.nan)
            
            pos = x + offsets[i]
            color = colors[i % len(colors)]
            
            valid = [not pd.isna(c) for c in coefs]
            v_pos = pos[valid]
            v_coefs = [coefs[j] for j in range(len(coefs)) if valid[j]]
            v_ses = [ses[j] for j in range(len(ses)) if valid[j]]
            v_pvals = [pvals[j] for j in range(len(pvals)) if valid[j]]

            ax.bar(
                v_pos, v_coefs, width, 
                yerr=v_ses, capsize=3, 
                color=color, edgecolor='black', linewidth=0.5, alpha=0.85, 
                label=str(group)
            )

            if pval_col:
                annotate_stars(ax, v_pos, v_coefs, v_pvals)

        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
        ax.set_xticks(x)
        
        display_labels = [labels.get(xv, get_readable_label(xv)) if labels else get_readable_label(xv) for xv in x_vals]
        ax.set_xticklabels(display_labels)
        
        if ylabel:
            ax.set_ylabel(ylabel, size=16)
        if title:
            ax.set_title(title, size=18)
            
        ax.legend(loc='best', fontsize=10)
        
        if footer:
            ax.text(0.99, 0.01, footer, transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color='grey')

        return fig, ax

class StackedBarChart:
    """
    Stacked bars.
    """
    def __init__(self, style_path="barchart"):
        self.style_path = style_path

    def plot(self, df, x_col, y_cols, labels=None, ylabel=None, title=None, ax=None, colors=None):
        if ax is None:
            apply_style(self.style_path)
            fig, ax = plt.subplots(figsize=(14, 7))
        else:
            fig = ax.figure

        if colors is None:
            from stats_transformer.visualization.defaults.colors import get_color_palette
            colors = get_color_palette("default", len(y_cols))

        x = df[x_col]
        bottom = np.zeros(len(x))

        for i, y_col in enumerate(y_cols):
            y_vals = df[y_col]
            label = labels.get(y_col, get_readable_label(y_col)) if labels else get_readable_label(y_col)
            ax.bar(x, y_vals, bottom=bottom, color=colors[i % len(colors)], label=label, edgecolor='white', linewidth=0.5)
            bottom += y_vals

        if ylabel:
            ax.set_ylabel(ylabel, size=16)
        if title:
            ax.set_title(title, size=18)

        ax.legend(loc='best', fontsize=10)

        return fig, ax
