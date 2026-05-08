from abc import ABC, abstractmethod
import traceback

class BaseExample(ABC):
    """
    Base class for all stats-transformer examples to ensure a standardized execution pipeline.
    """

    @abstractmethod
    def _fetch_and_prepare_data(self):
        """Fetch and prepare data, saving it if necessary."""
        pass

    @abstractmethod
    def _compute_baseline(self, df):
        """Compute the baseline model (e.g., statsmodels or linearmodels)."""
        pass

    @abstractmethod
    def _compute_stats_transformer(self, df):
        """Compute the equivalent stats-transformer model."""
        pass

    @abstractmethod
    def _compare_and_report(self, baseline_model, st_model):
        """Compare the baseline and stats-transformer models and report differences."""
        pass

    def run(self):
        """Standardized execution pipeline."""
        try:
            df_clean = self._fetch_and_prepare_data()
            baseline_model = self._compute_baseline(df_clean)
            st_model = self._compute_stats_transformer(df_clean)
            self._compare_and_report(baseline_model, st_model)
        except Exception as e:
            print(f"Error executing {self.__class__.__name__}: {e}")
            traceback.print_exc()
