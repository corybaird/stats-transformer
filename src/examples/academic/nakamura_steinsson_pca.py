import numpy as np
from stats_transformer.data import load_example
from stats_transformer.models.unsupervised.unsupervised import PCAModel

class NakamuraSteinssonPCAExample:
    def __init__(self):
        self.features = ['mp1', 'ff4', 'ed2', 'ed3', 'ed4']

    def run(self):
        df = load_example('synthetic_nakamura')
        
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
    runner = NakamuraSteinssonPCAExample()
    runner.run()
