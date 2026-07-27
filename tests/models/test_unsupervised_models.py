import pytest
import pandas as pd
import numpy as np
from stats_transformer.models.unsupervised.unsupervised import PCAModel, KMeansModel

def test_pca_model():
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(size=n)
    x2 = x1 + np.random.normal(scale=0.1, size=n)
    x3 = np.random.normal(size=n)
    
    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": x3
    })
    
    model = PCAModel(
        features=["x1", "x2", "x3"],
        n_components=2
    )
    
    model.fit(df)
    metrics = model.get_model_metrics()
    
    assert "explained_variance_ratio" in metrics
    assert len(metrics["explained_variance_ratio"]) == 2
    assert "pca_1" in model.df_clean.columns
    assert "pca_2" in model.df_clean.columns

def test_kmeans_model():
    np.random.seed(42)
    n = 100
    # Create two clusters
    cluster1 = np.random.normal(loc=0, size=(50, 2))
    cluster2 = np.random.normal(loc=5, size=(50, 2))
    X = np.vstack([cluster1, cluster2])
    
    df = pd.DataFrame(X, columns=["x1", "x2"])
    
    model = KMeansModel(
        features=["x1", "x2"],
        n_clusters=2
    )
    
    model.fit(df)
    metrics = model.get_model_metrics()
    
    assert "inertia" in metrics
    assert "cluster" in model.df_clean.columns
    assert len(model.df_clean['cluster'].unique()) == 2
