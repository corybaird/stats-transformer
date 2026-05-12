import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyfit
from stats_transformer.visualization.formatters.style import apply_style
from stats_transformer.visualization.defaults.labels import get_readable_label

class BinnedScatterPlot:
    """
    Raw scatter + binned means + error bars + OLS trend line.
    """
    def __init__(self, style_path="scatter"):
        self.style_path = style_path

    def plot(self, df, x_col, y_col, hue_col=None, n_bins=15, title=None, xlabel=None, ylabel=None, ax=None, colors=None):
        if ax is None:
            apply_style(self.style_path)
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        df = df.copy().dropna(subset=[x_col, y_col])
        if df.empty:
            return fig, ax

        # Raw scatter points
        if hue_col and hue_col in df.columns:
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, palette=colors, ax=ax, legend=False, s=25, alpha=0.2, linewidth=0)
        else:
            ax.scatter(df[x_col], df[y_col], s=25, alpha=0.2, color='#757575', linewidth=0)

        # Binning
        df['_bin'] = pd.qcut(df[x_col], q=n_bins, duplicates='drop')
        bin_means = df.groupby('_bin', observed=True).agg(
            x_mean=(x_col, 'mean'), 
            y_mean=(y_col, 'mean'), 
            y_se=(y_col, 'sem'), 
            n=(y_col, 'count')
        ).reset_index()
        bin_means = bin_means[bin_means['n'] >= 5]

        if not bin_means.empty:
            ax.scatter(bin_means['x_mean'], bin_means['y_mean'], s=bin_means['n'] * 0.4 + 40, color='#1565C0', zorder=5, edgecolors='white', linewidths=0.8)
            ax.errorbar(bin_means['x_mean'], bin_means['y_mean'], yerr=1.96 * bin_means['y_se'], fmt='none', color='#1565C0', alpha=0.5, zorder=4, capsize=3)

        # OLS trend line
        if len(df) > 1:
            coeffs = polyfit(df[x_col], df[y_col], 1)
            x_line = np.linspace(df[x_col].min(), df[x_col].max(), 100)
            y_line = coeffs[0] + coeffs[1] * x_line
            ax.plot(x_line, y_line, color='#C62828', linewidth=2, zorder=6, label=f'slope = {coeffs[1]:.3f}')

        ax.axhline(0, color='black', linewidth=0.8, alpha=0.4)

        ax.set_xlabel(xlabel if xlabel else get_readable_label(x_col), fontsize=11)
        ax.set_ylabel(ylabel if ylabel else get_readable_label(y_col), fontsize=11)
        
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
            
        ax.legend(fontsize=9, loc='upper right')

        return fig, ax

class ScatterWithRegression:
    """
    Simple scatter with regression line and correlation annotation.
    """
    def __init__(self, style_path="scatter"):
        self.style_path = style_path

    def plot(self, df, x_col, y_col, hue_col=None, title=None, xlabel=None, ylabel=None, ax=None, colors=None):
        if ax is None:
            apply_style(self.style_path)
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        df = df.copy().dropna(subset=[x_col, y_col])
        if df.empty:
            return fig, ax

        if hue_col and hue_col in df.columns:
            sns.scatterplot(x=x_col, y=y_col, data=df, hue=hue_col, palette=colors, alpha=0.7, ax=ax)
        else:
            sns.scatterplot(x=x_col, y=y_col, data=df, alpha=0.7, ax=ax, color='#1565C0')
            
        if len(df) > 1:
            corr = df[[x_col, y_col]].corr().iloc[0, 1]
            ax.text(0.05, 0.95, f"Corr: {corr:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            sns.regplot(x=x_col, y=y_col, data=df, scatter=False, ax=ax, line_kws={"color":"red", "alpha":0.7, "lw":2})

        ax.set_xlabel(xlabel if xlabel else get_readable_label(x_col), fontsize=11)
        ax.set_ylabel(ylabel if ylabel else get_readable_label(y_col), fontsize=11)
        
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
            
        ax.grid(True, linestyle='--', alpha=0.7)

        return fig, ax
