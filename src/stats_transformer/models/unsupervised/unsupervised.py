import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from stats_transformer.models.base import ModelBase

class UnsupervisedModel(ModelBase):
    # Base class for unsupervised models bypassing some statsmodels assumptions

    """Base class for unsupervised models."""
    def __init__(
        self,
        params_path: Optional[str] = None,
        features: Optional[List[str]] = None,
        n_components: int = 2,
        model_type: str = "pca",
        standardize: bool = True,
        with_mean: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(params_path=params_path, target=features[0] if features else "dummy", independent_variables=features, **kwargs)
        self.features = self.independent_variables
        self.n_components = n_components
        self.model_type = model_type
        self.standardize = standardize
        self.with_mean = with_mean
        self.scaler = StandardScaler(with_mean=self.with_mean, with_std=self.standardize)

    def _get_required_columns(self) -> List[str]:
        return self.features

    def load_data(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Load the feature columns and store the scaled design matrix."""
        self.logger.info("Loading data for UnsupervisedModel")
        self.df = pd.read_csv(data) if isinstance(data, str) and data.endswith(".csv") else (pd.read_parquet(data) if isinstance(data, str) else data.copy())
        required = self._get_required_columns()
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        self.df_clean = self.df.dropna(subset=required).copy()
        self.X = self.df_clean[required]
        self.X_scaled = self.scaler.fit_transform(self.X)
        return self.df_clean

    def get_summary(self) -> str:
        """Return a text summary of the fitted model."""
        return f"{self.model_type} Model with {self.n_components} components"

    def get_model_metrics(self) -> Dict[str, Any]:
        """Return key fit metrics as a dictionary."""
        return {}

    def get_model_metadata(self, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Assemble the metadata dictionary saved with the model."""
        return {
            "model_version": self.model_version,
            "creation_timestamp": datetime.now().isoformat(),
            "model_type": self.model_type,
            "metrics": self.get_model_metrics()
        }

class _UncenteredPCA(TruncatedSVD):
    # TruncatedSVD.explained_variance_ratio_ divides by the per-feature variance of the
    # input, which is only meaningful for centered data. Rescale to the PCA convention
    # sigma_i^2 / sum_j sigma_j^2 for uncentered matrices.
    def fit(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        """Fit the decomposition and store the rescaled variance measures."""
        return self.fit_transform(X, y)

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        """Fit, transform and store the rescaled variance measures."""
        Xtr = super().fit_transform(X, y)
        X = np.asarray(X, dtype=float)
        total_ss = float(np.sum(X**2))
        sv2 = np.asarray(self.singular_values_) ** 2
        self.explained_variance_ = sv2 / (X.shape[0] - 1)
        self.explained_variance_ratio_ = sv2 / total_ss
        return Xtr

def rotate_target_path(
    factors: Union[np.ndarray, pd.DataFrame],
    loadings: Union[np.ndarray, pd.DataFrame],
    target_index: int = 0,
    path_index: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rotate a two-factor PCA solution to target and path factors."""
    # Gürkaynak-Sack-Swanson (2005) Target/Path rotation of a 2-factor PCA solution.
    # Zero-loading condition on the rotated loading of the target surprise:
    #   (L U)[target_index, 1] = L[i,0]*sin(th) + L[i,1]*cos(th) = 0
    #   => th = atan2(-L[i,1], L[i,0]), which also gives (L U)[target_index,0] = |L[i,:]| > 0.
    # Factors are first rescaled to unit variance (scores /s_j, loadings *s_j, fit F L'
    # preserved) because F @ U keeps rotated columns uncorrelated only for equal-variance
    # columns. GSS unit-effect normalisation then rescales scores by the pivot loadings
    # and divides loadings by them, again preserving the fit F L'.
    F = np.asarray(factors, dtype=float)
    L = np.asarray(loadings, dtype=float)
    if F.ndim != 2 or F.shape[1] != 2 or L.shape[1] != 2:
        raise ValueError("rotate_target_path requires exactly 2 factor columns")
    if not 0 <= target_index < L.shape[0]:
        raise ValueError(f"target_index {target_index} out of range for {L.shape[0]} features")
    unit = F.std(axis=0, ddof=1)
    unit = np.where(unit > 0, unit, 1.0)
    F = F / unit
    L = L * unit
    theta = np.arctan2(-L[target_index, 1], L[target_index, 0])
    c, s = np.cos(theta), np.sin(theta)
    U = np.array([[c, s], [-s, c]])
    F_rot = F @ U
    L_rot = L @ U
    if path_index is None:
        # GSS fallback: unit-variance factors (already true after the pre-rescale)
        pivot = np.ones(2)
    else:
        if not 0 <= path_index < L.shape[0]:
            raise ValueError(f"path_index {path_index} out of range for {L.shape[0]} features")
        # GSS normalisation: Target has unit effect on the target surprise, Path on the path surprise
        pivot = np.array([L_rot[target_index, 0], L_rot[path_index, 1]])
        if np.any(np.abs(pivot) < 1e-12):
            raise ValueError("Degenerate loadings at the normalisation pivot")
    F_rot = F_rot * pivot
    L_rot = L_rot / pivot
    F_df = pd.DataFrame(F_rot, columns=["target", "path"], index=factors.index if isinstance(factors, pd.DataFrame) else None)
    L_df = pd.DataFrame(L_rot, columns=["target", "path"], index=loadings.index if isinstance(loadings, pd.DataFrame) else None)
    return F_df, L_df

class PCAModel(UnsupervisedModel):
    # Principal Component Analysis

    """Principal component analysis."""
    def __init__(
        self,
        params_path: Optional[str] = None,
        features: Optional[List[str]] = None,
        n_components: int = 2,
        standardize: bool = True,
        with_mean: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(
            params_path=params_path,
            features=features,
            n_components=n_components,
            model_type="PCA",
            standardize=standardize,
            with_mean=with_mean,
            **kwargs
        )

    def rotate_target_path(self, target_index: int = 0, path_index: Optional[int] = None) -> pd.DataFrame:
        """Rotate the first two factors and return the target and path scores."""
        # Rotates the first two PCA factors GSS-style so the second ("path") factor has zero
        # loading on the surprise at target_index, then normalises (see module-level function).
        if self.model is None or self.transformed_X is None or self.df_clean is None:
            raise ValueError("PCAModel must be fit (build_model) before rotate_target_path")
        if self.n_components < 2:
            raise ValueError("rotate_target_path requires n_components >= 2")
        F_df, L_df = rotate_target_path(
            factors=pd.DataFrame(self.transformed_X[:, :2], index=self.df_clean.index),
            loadings=pd.DataFrame(self.model.components_.T[:, :2], index=self.features),
            target_index=target_index,
            path_index=path_index
        )
        self.df_clean[["target", "path"]] = F_df
        self.rotated_loadings_ = L_df
        return F_df

    def build_model(self) -> Any:
        """Fit PCA on the scaled features and store the component scores."""
        assert self.df_clean is not None  # set by load_data, called via fit() before build_model()
        if self.with_mean:
            self.model = PCA(n_components=self.n_components)
        else:
            # sklearn PCA always centers, so with_mean=False uses SVD on the uncentered matrix
            self.model = _UncenteredPCA(n_components=self.n_components, random_state=0)
        self.transformed_X = self.model.fit_transform(self.X_scaled)
        for i in range(self.n_components):
            self.df_clean[f'pca_{i+1}'] = self.transformed_X[:, i]
        return self.model

    def get_model_metrics(self) -> Dict[str, Any]:
        """Return key fit metrics as a dictionary."""
        if not self.model:
            return {}
        return {
            "explained_variance_ratio": self.model.explained_variance_ratio_.tolist(),
            "cumulative_variance": float(np.sum(self.model.explained_variance_ratio_))
        }

    def get_summary(self) -> str:
        """Return a text summary of the fitted model."""
        if not self.model:
            return "PCA Model not trained"
        var_ratios = self.model.explained_variance_ratio_
        res = ["PCA Explained Variance Ratio per component:"]
        for i, var in enumerate(var_ratios):
            res.append(f"  Component {i+1}: {var:.4f}")
        res.append(f"Total variance explained: {np.sum(var_ratios):.4f}")
        return "\n".join(res)

    def run(self, data_path: Optional[str] = None, output_path: Optional[str] = None) -> pd.DataFrame:
        """Fit the model and return the features with component scores."""
        self.fit(data_path)
        assert self.df_clean is not None  # set by fit() -> load_data()
        if output_path:
            self.df_clean.to_csv(output_path, index=False)
        return self.df_clean

class KMeansModel(UnsupervisedModel):
    # K-Means Clustering

    """K-means clustering model."""
    def __init__(self, params_path: Optional[str] = None, features: Optional[List[str]] = None, n_clusters: int = 3, **kwargs: Any) -> None:
        super().__init__(params_path=params_path, features=features, n_components=n_clusters, model_type="KMeans", **kwargs)
        self.n_clusters = n_clusters

    def build_model(self) -> KMeans:
        """Cluster the scaled features and store the cluster labels."""
        assert self.df_clean is not None  # set by load_data, called via fit() before build_model()
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.cluster_labels = self.model.fit_predict(self.X_scaled)
        self.df_clean['cluster'] = self.cluster_labels
        return self.model

    def get_model_metrics(self) -> Dict[str, Any]:
        """Return key fit metrics as a dictionary."""
        if not self.model:
            return {}
        return {
            "inertia": float(self.model.inertia_),
            "n_iter": int(self.model.n_iter_),
            "cluster_centers": self.model.cluster_centers_.tolist()
        }

    def get_summary(self) -> str:
        """Return a text summary of the fitted model."""
        if not self.model:
            return "KMeans Model not trained"
        return f"KMeans Clustering (k={self.n_clusters}) - Inertia: {self.model.inertia_:.2f}"

    def run(self, data_path: Optional[str] = None, output_path: Optional[str] = None) -> pd.DataFrame:
        """Fit the model and return the features with cluster labels."""
        self.fit(data_path)
        assert self.df_clean is not None  # set by fit() -> load_data()
        if output_path:
            self.df_clean.to_csv(output_path, index=False)
        return self.df_clean
