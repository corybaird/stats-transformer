import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from stats_transformer.models.timeseries.reduced_form.forecasting import VARForecaster

def test_var_forecaster_matches_statsmodels():
    np.random.seed(42)
    data = np.random.randn(100, 2)
    
    # Train statsmodels VAR
    sm_model = VAR(data).fit(2)
    
    # Use our forecaster
    forecaster = VARForecaster(sm_model)
    y_hist = data[-2:]
    
    point_fc, lower_fc, upper_fc = forecaster.forecast(y_hist, steps=5, alpha=0.05)
    
    # Get statsmodels forecast
    sm_point, sm_lower, sm_upper = sm_model.forecast_interval(y_hist, steps=5, alpha=0.05)
    
    np.testing.assert_allclose(point_fc, sm_point, rtol=1e-5)
    np.testing.assert_allclose(lower_fc, sm_lower, rtol=1e-5)
    np.testing.assert_allclose(upper_fc, sm_upper, rtol=1e-5)

def test_forecast_evaluator_metrics():
    from stats_transformer.models.timeseries.analysis.forecast_evaluation import ForecastEvaluator
    
    # Hand calculated data
    actual = np.array([[1.0, 2.0], [3.0, 4.0]])
    predicted = np.array([[1.5, 1.5], [2.5, 4.5]])
    
    # errors:
    # row 0: [-0.5, 0.5]
    # row 1: [0.5, -0.5]
    # MAE:
    # col 0: (| -0.5 | + | 0.5 |) / 2 = 0.5
    # col 1: (| 0.5 | + | -0.5 |) / 2 = 0.5
    mae = ForecastEvaluator.calculate_mae(actual, predicted)
    np.testing.assert_allclose(mae, [0.5, 0.5])
    
    # RMSE:
    # col 0: sqrt(((-0.5)^2 + 0.5^2)/2) = sqrt((0.25+0.25)/2) = sqrt(0.25) = 0.5
    # col 1: sqrt(((0.5)^2 + (-0.5)^2)/2) = 0.5
    rmse = ForecastEvaluator.calculate_rmse(actual, predicted)
    np.testing.assert_allclose(rmse, [0.5, 0.5])

def test_forecast_evaluator_rolling_origin():
    from stats_transformer.models.timeseries.analysis.forecast_evaluation import ForecastEvaluator
    from stats_transformer.models.timeseries.reduced_form.var import VARModel
    
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(30, 2), columns=["y1", "y2"])
    
    rmse = ForecastEvaluator.evaluate_rolling_origin(
        model_class=VARModel,
        df=data,
        initial_train_size=25,
        steps=2,
        metric="rmse",
        target_variables=["y1", "y2"],
        maxlags=1
    )
    
    assert len(rmse) == 2
    assert np.all(rmse >= 0)

