import os
import numpy as np
import pandas as pd
from src.stats_transformer.models.unsupervised.unsupervised import PCAModel

class NakamuraSteinssonPCASanity:
    def __init__(self):
        self.n_samples = 150
        self.features = ['mp1', 'ff4', 'ed2', 'ed3', 'ed4']

    def generate_synthetic_data(self):
        np.random.seed(42)
        # Latent factor: the unobserved "policy news shock"
        true_shock = np.random.normal(0, 1, self.n_samples)
        
        # Loadings typically increase for longer horizons (forward guidance)
        loadings = np.array([0.2, 0.5, 0.8, 0.9, 0.95])
        
        data = {}
        for i, feature in enumerate(self.features):
            # Observed changes in futures = loading * shock + idiosyncratic noise
            data[feature] = true_shock * loadings[i] + np.random.normal(0, 0.1, self.n_samples)
            
        df = pd.DataFrame(data)
        return df

    def run(self):
        df = self.generate_synthetic_data()
        
        # Save for YAML example
        data_dir = "data/raw/examples/unsupervised"
        os.makedirs(data_dir, exist_ok=True)
        data_path = os.path.join(data_dir, "synthetic_nakamura.csv")
        df.to_csv(data_path, index=False)
        print(f"Synthetic data saved to {data_path}")
        
        # Instantiate the PCA model to extract 1 component
        model = PCAModel(features=self.features, n_components=1)
        
        # In PCAModel, load_data sets up X and X_scaled
        transformed_df = model.load_data(df)
        
        # Build the model to perform the PCA decomposition
        model.build_model()
        
        print("--- Nakamura & Steinsson (2018) PCA Shock Extraction Sanity Check ---")
        print("Simulated high-frequency changes in 5 interest rate futures around FOMC announcements.")
        print("Extracting the first principal component as the 'Policy News Shock'.\n")
        
        print(model.get_summary())
        
        metrics = model.get_model_metrics()
        explained_variance = metrics.get('explained_variance_ratio', [0])[0]
        print(f"\nVariance explained by the first principal component (Policy News Shock): {explained_variance:.4f}")
        
        print("\nFirst 5 extracted Policy News Shocks (pca_1):")
        print(transformed_df[['pca_1']].head())

if __name__ == "__main__":
    runner = NakamuraSteinssonPCASanity()
    runner.run()
