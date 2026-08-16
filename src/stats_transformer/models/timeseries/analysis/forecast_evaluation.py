import numpy as np
import pandas as pd

class RollingOriginEvaluator:
    """
    Evaluates forecast accuracy using metrics such as RMSE and MAE.
    Supports rolling-origin evaluations.

    Distinct from timeseries.utilities.ForecastEvaluator, which is
    instance-based and scores two columns of an existing DataFrame. This one
    is static and re-fits a model across expanding windows.
    """
    @staticmethod
    def calculate_rmse(actual, predicted):
        """
        Calculates Root Mean Squared Error.
        """
        return np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2, axis=0))

    @staticmethod
    def calculate_mae(actual, predicted):
        """
        Calculates Mean Absolute Error.
        """
        return np.mean(np.abs(np.asarray(actual) - np.asarray(predicted)), axis=0)

    @classmethod
    def evaluate_rolling_origin(cls, model_class, df, initial_train_size, steps, metric="rmse", **model_kwargs):
        """
        Performs rolling-origin forecast evaluation.
        
        model_class: The model class to instantiate (e.g., VARModel). Must have `fit()` and `model` attribute.
        df: The full dataset (pandas DataFrame).
        initial_train_size: Number of observations to use for the first training window.
        steps: Forecast horizon for each roll.
        metric: 'rmse' or 'mae'.
        model_kwargs: Additional arguments to pass to the model constructor.
        
        Returns:
            A numpy array of shape (neqs,) containing the average metric across all rolls.
        """
        total_obs = len(df)
        if initial_train_size + steps > total_obs:
            raise ValueError("Dataset too small for the specified initial_train_size and steps.")

        errors = []
        for i in range(initial_train_size, total_obs - steps + 1):
            train_df = df.iloc[:i].copy()
            test_df = df.iloc[i : i + steps].copy()
            
            # Instantiate and fit
            model = model_class(**model_kwargs)
            model.fit(train_df)
            
            # Forecast
            from stats_transformer.models.timeseries.reduced_form.forecasting import VARForecaster
            forecaster = VARForecaster(model.model)
            
            # The history for the forecaster is the end of the training data
            y_hist = train_df[model.target_variables].values
            
            point_fc, _, _ = forecaster.forecast(y_hist, steps=steps)
            actual = test_df[model.target_variables].values
            
            if metric == "rmse":
                err = cls.calculate_rmse(actual, point_fc)
            elif metric == "mae":
                err = cls.calculate_mae(actual, point_fc)
            else:
                raise ValueError(f"Unknown metric {metric}")
                
            errors.append(err)
            
        # Average the errors across all rolls
        return np.mean(errors, axis=0)
