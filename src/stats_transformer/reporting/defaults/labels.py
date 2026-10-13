"""
Default label mappings, formatting templates, and thresholds for stats-transformer visualizations.
"""

SIGNIFICANCE_THRESHOLDS = {
    0.01: "***",
    0.05: "**",
    0.1: "*"
}

FOOTER_TEMPLATES = {
    "panel_fe": "Entity FE  |  Clustered SE",
    "panel_fe_time": "Entity + Time FE  |  Clustered SE",
}

# Standard sentiment variable labels
SENTIMENT_LABELS = {
    'correa_sentiment': 'Correa',
    'hubert_sentiment': 'Hubert',
    'lm_sentiment': 'Loughran-McDonald',
    'hiv_sentiment': 'Harvard IV-4',
    'mp_score': 'MP Score',
    'info_score': 'Info Score',
    'mp_score_lag1': 'MP Score (t-1)',
    'info_score_lag1': 'Info Score (t-1)',
    'mp_score_purged_lag1': 'Purged MP Score (t-1)',
    'info_score_purged_lag1': 'Purged Info Score (t-1)'
}

def get_readable_label(x):
    """Fallback readable label generator if not found in a dict."""
    return str(x).replace("_", " ").title()
