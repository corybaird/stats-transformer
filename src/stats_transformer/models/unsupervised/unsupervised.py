import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.stats_transformer.models.base import ModelBase

class UnsupervisedModel(ModelBase):
    # Base class for unsupervised models bypassing some statsmodels assumptions

    def __init__(self, params_path=None, features=None, n_components=2, model_type="pca", **kwargs):
        super().__init__(params_path=params_path, target=features[0] if features else "dummy", independent_variables=features, **kwargs)
        self.features = self.independent_variables
        self.n_components = n_components
        self.model_type = model_type
        self.scaler = StandardScaler()

    def _get_required_columns(self):
        return self.features

    def load_data(self, data):
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

    def get_summary(self):
        return f"{self.model_type} Model with {self.n_components} components"

    def get_model_metrics(self):
        return {}

    def get_model_metadata(self, metrics=None):
        return {
            "model_version": self.model_version,
            "creation_timestamp": datetime.now().isoformat(),
            "model_type": self.model_type,
            "metrics": self.get_model_metrics()
        }

class PCAModel(UnsupervisedModel):
    # Principal Component Analysis

    def __init__(self, params_path=None, features=None, n_components=2, **kwargs):
        super().__init__(params_path=params_path, features=features, n_components=n_components, model_type="PCA", **kwargs)

    def build_model(self):
        self.model = PCA(n_components=self.n_components)
        self.transformed_X = self.model.fit_transform(self.X_scaled)
        for i in range(self.n_components):
            self.df_clean[f'pca_{i+1}'] = self.transformed_X[:, i]
        return self.model

    def get_model_metrics(self):
        if not self.model:
            return {}
        return {
            "explained_variance_ratio": self.model.explained_variance_ratio_.tolist(),
            "cumulative_variance": float(np.sum(self.model.explained_variance_ratio_))
        }

    def get_summary(self):
        if not self.model:
            return "PCA Model not trained"
        var_ratios = self.model.explained_variance_ratio_
        res = ["PCA Explained Variance Ratio per component:"]
        for i, var in enumerate(var_ratios):
            res.append(f"  Component {i+1}: {var:.4f}")
        res.append(f"Total variance explained: {np.sum(var_ratios):.4f}")
        return "\n".join(res)

    def run(self, data_path=None, output_path=None):
        self.fit(data_path)
        if output_path:
            self.df_clean.to_csv(output_path, index=False)
        return self.df_clean

class KMeansModel(UnsupervisedModel):
    # K-Means Clustering

    def __init__(self, params_path=None, features=None, n_clusters=3, **kwargs):
        super().__init__(params_path=params_path, features=features, n_components=n_clusters, model_type="KMeans", **kwargs)
        self.n_clusters = n_clusters

    def build_model(self):
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.cluster_labels = self.model.fit_predict(self.X_scaled)
        self.df_clean['cluster'] = self.cluster_labels
        return self.model

    def get_model_metrics(self):
        if not self.model:
            return {}
        return {
            "inertia": float(self.model.inertia_),
            "n_iter": int(self.model.n_iter_),
            "cluster_centers": self.model.cluster_centers_.tolist()
        }

    def get_summary(self):
        if not self.model:
            return "KMeans Model not trained"
        return f"KMeans Clustering (k={self.n_clusters}) - Inertia: {self.model.inertia_:.2f}"

    def run(self, data_path=None, output_path=None):
        self.fit(data_path)
        if output_path:
            self.df_clean.to_csv(output_path, index=False)
        return self.df_clean
