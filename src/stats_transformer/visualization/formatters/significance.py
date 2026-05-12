import pandas as pd
import numpy as np

def star(pval, thresholds=None):
    """
    Returns a significance star string based on the p-value.
    """
    if pd.isna(pval):
        return ""
        
    if thresholds is None:
        from stats_transformer.visualization.defaults.labels import SIGNIFICANCE_THRESHOLDS
        thresholds = SIGNIFICANCE_THRESHOLDS
        
    for thresh in sorted(thresholds.keys()):
        if pval < thresh:
            return thresholds[thresh]
    return ""

def format_pvalue(pval):
    """
    Formats a p-value for display.
    """
    if pd.isna(pval) or pval == "N/A":
        return "N/A"
    try:
        fv = float(pval)
        if fv < 0.001:
            return f"{fv:.3e}"
        return f"{fv:.4f}"
    except (TypeError, ValueError):
        return str(pval)

def annotate_stars(ax, x_positions, coefs, pvals, thresholds=None, offset_factor=0.1, min_offset=0.005, fontsize=11):
    """
    Annotates a matplotlib axes with significance stars.
    """
    for x, c, p in zip(x_positions, coefs, pvals):
        if pd.isna(c) or pd.isna(p):
            continue
        s = star(p, thresholds)
        if s:
            offset_y = max(abs(c) * offset_factor, min_offset)
            y_pos = c + offset_y if c >= 0 else c - offset_y
            va = 'bottom' if c >= 0 else 'top'
            ax.text(x, y_pos, s, ha='center', va=va, fontsize=fontsize, fontweight='bold')
