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

def test_pca_with_mean_false_does_not_center():
    # Means dominate the matrix: non-centered PCA must differ from centered PCA
    df = pd.DataFrame({
        "c1": [100.0, 101.0, 103.0],
        "c2": [2.0, 5.0, 3.0],
    })
    centered = PCAModel(features=["c1", "c2"], n_components=2)
    centered.fit(df)
    uncentered = PCAModel(features=["c1", "c2"], n_components=2, with_mean=False)
    uncentered.fit(df)
    # Centered first PC reflects the (symmetric-ish) covariance of the deviations
    assert np.allclose(np.abs(centered.model.components_[0]), [0.7071, 0.7071], atol=0.05)
    # Non-centered first PC aligns with the mean direction: nearly all weight on c1
    assert uncentered.model.components_[0, 0] > 0.9
    assert abs(uncentered.model.components_[0, 1]) < 0.2
    assert uncentered.get_model_metrics()["explained_variance_ratio"][0] > 0.99
