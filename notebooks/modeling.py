import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from holidays import US
from typing import Dict
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting import MeanAbsoluteError
from sktime.forecasting.model_selection import ForecastingGridSearchCV


def get_forecasts(
    models, horizon, X_test: pd.DataFrame = None, ensemble: bool = True
) -> Dict:
    """
    Produce a dictionary with keys the model names and values as the model predictions
    """
    n = len(models)
    forecasts = {name: model.predict(horizon, X=X_test) for name, model in models.items()}
    if ensemble:
        # mean of all the models
        forecasts["Ensemble"] = sum(forecasts.values()) / n

    return forecasts


def tune_forecasters(forecasters, cross_validation, y, X, scoring=MeanAbsoluteError()):
    """
    Tune different forecasters specified in the forecasters dictionary,
    applying a cross validation strategy.
    """

    def tune(forecaster, param_grid):
        gscv = ForecastingGridSearchCV(
            forecaster=forecaster,
            param_grid=param_grid,
            cv=cross_validation,
            scoring=scoring,
        )
        gscv.fit(y, X)
        return gscv.best_forecaster_

    return {k: tune(m["forecaster"], m["grid"]) for k, m in forecasters.items()}


def evaluate_forecasters(forecasters, y, X, cv, scoring):
    """
    Evaluate different forecasters using a cross validation strategy
    """

    def evaluate_forecaster(name, forecaster, y, X, cv, scoring):
        return evaluate(
            forecaster=forecaster,
            y=y,
            X=X,
            cv=cv,
            scoring=scoring,
        ).assign(forecaster_name=name)

    models = forecasters.items()
    evaluation_dfs = [evaluate_forecaster(k, m, y, X, cv, scoring) for k, m in models]
    evaluation_result_df = pd.concat(evaluation_dfs)
    return evaluation_result_df


def get_metrics_results(
    metrics: Dict, forecasts: Dict, actual: pd.Series
) -> pd.DataFrame:
    """
    Get different metrics using the predictions and the actuals and return them as a dataframe
    """
    return pd.DataFrame(
        {
            metric_name: [metric(actual, y_pred) for y_pred in forecasts.values()]
            for metric_name, metric in metrics.items()
        },
        index=forecasts.keys(),
    )


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the dataset with some time dependent features like holidays and so on.
    Can be further extended.
    """
    result_df = df.copy().assign(
        year=lambda df: df["date"].dt.year,
        month=lambda df: df["date"].dt.month,
        week=lambda df: df["date"].dt.isocalendar().week,
        day=lambda df: df["date"].dt.day,
        weekday=lambda df: df["date"].dt.weekday + 1,
        is_weekend=lambda df: df["date"].dt.weekday >= 5,
        is_holiday=lambda df: df["date"].apply(lambda x: x in US()),
    )
    return result_df


class PositiveRidge(Ridge):
    def predict(self, X_test: pd.DataFrame) -> np.array:
        prediction = super().predict(X_test)
        return np.where(prediction < 0, 0, prediction)
