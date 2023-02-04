from urllib.parse import urlparse
import pandas as pd
import warnings
import constants
from hyperopt.pyll import scope
import numpy as np
import pandas as pd
import mlflow
import click
from warnings import simplefilter
from hyperopt import fmin, tpe, hp, Trials
from sktime.performance_metrics.forecasting import MeanAbsoluteError

# time series modeling and analysis
from sktime.forecasting.compose import make_reduction

# modeling
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    ExpandingWindowSplitter,
)
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError
from sktime.forecasting.base import ForecastingHorizon
from lightgbm import LGBMRegressor

from sktime.forecasting.model_evaluation import evaluate

# sktime models
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    temporal_train_test_split,
)


@click.command(
    help="Trains an Keras model on wine-quality dataset."
    "The input is expected in csv format."
    "The model and its metrics are logged with mlflow."
)
@click.option("--test_size", type=click.INT, default=28, help="Number of testing days")
@click.option("--n_folds", type=click.INT, default=1)
@click.option("--window_length", type=click.INT, default=50)
@click.option("--num_boost_round", type=click.INT, default=2)
@click.option("--num_leaves", type=click.INT, default=10)
@click.option("--reg_lambda", type=click.INT, default=1)
@click.option("--max_depth", type=click.INT, default=20)
@click.argument(
    "input_x_train_path",
    type=click.Path(exists=False),
    default="data/model_input/x_train.csv",
)
@click.argument(
    "input_y_train_path",
    type=click.Path(exists=False),
    default="data/model_input/y_train.csv",
)
def run(
    input_x_train_path: str,
    input_y_train_path: str,
    test_size: int,
    n_folds: int,
    window_length: int,
    **params,
):
    warnings.filterwarnings("ignore")

    with mlflow.start_run():
        X_train = pd.read_csv(input_x_train_path)
        y_train = pd.read_csv(input_y_train_path)

        # get start and end dates for setting the index with more information
        start_date = min(y_train[constants.DATE_COLUMN_NAME])
        last_date = max(y_train[constants.DATE_COLUMN_NAME])
        index_date_range = pd.date_range(start_date, last_date, freq="D")

        X_train = X_train.set_index(index_date_range).drop(
            columns=constants.DATE_COLUMN_NAME
        )
        y_train = y_train.set_index(index_date_range)[constants.TARGET_COLUMN_NAME]
        model = make_reduction(
            estimator=LGBMRegressor(**params),
            strategy="recursive",
            window_length=window_length,
        )

        if test_size > 0:
            evaluation_df = evaluate_lgbm_model(
                X_train=X_train,
                y_train=y_train,
                model=model,
                test_size=test_size,
                n_folds=n_folds,
            )
            metrics = evaluation_df["test_MeanSquaredError"].tolist()
            mean_score = evaluation_df["test_MeanSquaredError"].mean()
            fold_scores = {f"fold_{i}_mse": score for i, score in enumerate(metrics)}

            mlflow.log_metric("mean_rmse", mean_score)
            mlflow.log_metrics(fold_scores)
            mlflow.log_params(params)
        else:
            # fit on the entire training set instead of the evaluation one
            model.fit(X=X_train, y=y_train)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            # TODO: use mlflow sktime custom flavor instead
            mlflow.lightgbm.log_model(
                model.estimator,
                "lgbm",
                registered_model_name="LightGBMCashModel",
                metadata={"strategy": "recursive", "window_length": window_length},
            )
        else:
            mlflow.lightgbm.log_model(
                model.estimator,
                "lgbm",
                metadata={"strategy": "recursive", "window_length": window_length},
            )
        # trials = Trials()
        # n_iter = 100
        # random_state = 42

        # parameters = {k: v for k, v in params.items() if k != "window_length"}

        # best = fmin(
        #     fn=optimize,
        #     space=space,
        #     algo=tpe.suggest,
        #     max_evals=100,
        #     verbose=1,
        # trials=trials,
        # stratified=False,
        # rstate=np.random.RandomState(random_state),
        # )


def evaluate_lgbm_model(model, X_train, y_train, test_size, n_folds):
    forecast_horizon = list(range(1, test_size))
    initial_train_window = X_train.shape[0] - n_folds * test_size
    cross_validation = ExpandingWindowSplitter(
        initial_window=initial_train_window,
        step_length=test_size,
        fh=forecast_horizon,
    )

    evaluation_df = evaluate(
        forecaster=model,
        y=y_train,
        X=X_train,
        cv=cross_validation,
        scoring=MeanSquaredError(square_root=True),
    )
    return evaluation_df


if __name__ == "__main__":
    run()
