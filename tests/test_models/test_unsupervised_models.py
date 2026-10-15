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

def test_rotate_target_path_recovers_two_factor_structure():
    # 2 latent factors: MP1 (target surprise) loads only on the Target factor,
    # ED/FF futures load on both (Path factor). GSS (2005) rotation must recover them.
    from stats_transformer.models.unsupervised.unsupervised import rotate_target_path
    np.random.seed(42)
    n = 600
    target_true = np.random.normal(size=n)
    path_true = np.random.normal(size=n)
    loadings_true = np.array([  # rows: mp1, ff4, ed2, ed3, ed4
        [1.0, 0.0],
        [0.3, 0.5],
        [0.5, 0.7],
        [0.8, 0.9],
        [1.0, 1.0],
    ])
    X = np.column_stack([target_true, path_true]) @ loadings_true.T + np.random.normal(scale=0.05, size=(n, 5))
    df = pd.DataFrame(X, columns=["mp1", "ff4", "ed2", "ed3", "ed4"])

    model = PCAModel(features=["mp1", "ff4", "ed2", "ed3", "ed4"], n_components=2, standardize=False)
    model.fit(df)
    model.rotate_target_path(target_index=0, path_index=4)
    F_rot = model.df_clean[["target", "path"]].values
    L_rot = model.rotated_loadings_.values

    # Zero path loading on the target surprise; unit-effect pivots (GSS normalisation)
    assert abs(L_rot[0, 1]) < 1e-10
    assert np.isclose(L_rot[0, 0], 1.0)
    assert np.isclose(L_rot[4, 1], 1.0)
    # Rotated factors remain uncorrelated
    assert abs(np.corrcoef(F_rot[:, 0], F_rot[:, 1])[0, 1]) < 1e-10
    # Recover the true latent factors
    corr_target = np.corrcoef(F_rot[:, 0], target_true)[0, 1]
    corr_path = np.corrcoef(F_rot[:, 1], path_true)[0, 1]
    assert corr_target > 0.99 and corr_path > 0.99
    # Fitted reconstruction unchanged: F L' identical before and after rotation
    F_pre = model.transformed_X
    L_pre = model.model.components_.T
    assert np.allclose(F_pre @ L_pre.T, F_rot @ L_rot.T, atol=1e-8)
    # df_clean carries the named columns without disturbing pca_ columns
    assert "target" in model.df_clean.columns and "path" in model.df_clean.columns

def test_rotate_target_path_pure_function_and_unit_std():
    from stats_transformer.models.unsupervised.unsupervised import rotate_target_path
    # Exactly uncorrelated unit-std columns (as PCA scores are, up to roundoff)
    Q, _ = np.linalg.qr(np.random.normal(size=(200, 2)))
    F = pd.DataFrame(Q * np.sqrt(199), columns=["f1", "f2"])
    L = np.array([[0.6, 0.8], [1.0, 2.0]])  # target_index=0: path loading 0.8
    F_rot, L_rot = rotate_target_path(F, L, target_index=0)
    # Zero-loading condition by construction
    assert abs(L_rot.values[0, 1]) < 1e-10
    assert np.isclose(np.hypot(L[0, 0], L[0, 1]), np.sqrt(0.6**2 + 0.8**2))
    # path_index=None: both factors rescaled to (approximately) unit std;
    # exact for zero-mean orthogonal factors like PCA scores
    assert np.allclose(F_rot.std(ddof=1), [1.0, 1.0], atol=0.02)
    # Orthogonal U + consistent rescaling: F L' preserved
    assert np.allclose(F.values @ L.T, F_rot.values @ L_rot.values.T, atol=1e-10)
    # Rotated factor columns stay uncorrelated (exactly so for truly orthogonal inputs)
    assert abs(np.corrcoef(F_rot.values[:, 0], F_rot.values[:, 1])[0, 1]) < 1e-2
