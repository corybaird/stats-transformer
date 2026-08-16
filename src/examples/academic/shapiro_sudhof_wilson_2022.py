from pathlib import Path
import pandas as pd
from stats_transformer.featurization.feature_engineering import FeatureEngineer
from stats_transformer.models.regression.robust_ols import RobustOLSModel

class ShapiroSudhofWilson2022Replication:

    def __init__(self, data_path=None):
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path("data/examples/academic/shapiro_sudhof_wilson_2022/news_sentiment.parquet.gzip")
        self.model = None

    def _load_data(self):
        df = pd.read_parquet(self.data_path)
        return df.reset_index()

    def run(self):
        df = self._load_data()
        engineer = FeatureEngineer(transformations=["lag1", "lag2"], entity_column="country", date_column="date", period="daily", data_columns=["news_sentiment"], verbose=False)
        df_transformed = engineer.fit_transform(df)
        self.model = RobustOLSModel(target="news_sentiment", independent_variables=["news_sentiment_lag1", "news_sentiment_lag2"], cov_type="HC1")
        metrics = self.model.fit(df_transformed)
        print("Shapiro, Sudhof, & Wilson (2022) News Sentiment Metrics:", metrics)
        return {"metrics": metrics}
