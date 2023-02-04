"""
Example of hyperparameter search in MLflow using Hyperopt.

The run method will instantiate and run Hyperopt optimizer. Each parameter configuration is
evaluated in a new MLflow run invoking main entry point with selected parameters.

The runs are evaluated based on validation set loss. Test set score is calculated to verify the
results.

This example currently does not support parallel execution.
"""

import click
import numpy as np

from hyperopt import fmin, hp, tpe, rand
from hyperopt.pyll import scope
import mlflow.projects
from mlflow.tracking import MlflowClient

_inf = np.finfo(np.float64).max


@click.command(
    help="Perform hyperparameter search with Hyperopt library. Optimize dl_train target."
)
@click.option(
    "--max-runs", type=click.INT, default=10, help="Maximum number of runs to evaluate."
)
@click.option(
    "--algo", type=click.STRING, default="tpe.suggest", help="Optimizer algorithm."
)
@click.option(
    "--metric", type=click.STRING, default="rmse", help="Metric to optimize on."
)
@click.option("--test_size", type=click.INT, default=28, help="Number of testing days")
@click.option("--n_folds", type=click.INT, default=3)
# @click.option("--seed", type=click.INT, default=97531, help="Seed for the random generator")
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
def train(
    input_x_train_path: str,
    input_y_train_path: str,
    max_runs,
    algo,
    metric,
    test_size: int = 28,
    n_folds: int = 3,
):
    """
    Run hyperparameter optimization.
    """
    # create random file to store run ids of the training tasks
    tracking_client = MlflowClient()

    def new_eval(experiment_id):
        def eval(params):
            import mlflow.tracking

            window_length, num_boost_round, max_depth, num_leaves, reg_lambda = params
            with mlflow.start_run(nested=True) as child_run:
                p = mlflow.projects.run(
                    uri=".",
                    entry_point="train",
                    run_id=child_run.info.run_id,
                    parameters={
                        "input_x_train_path": input_x_train_path,
                        "input_y_train_path": input_y_train_path,
                        "test_size": test_size,
                        "n_folds": n_folds,
                        # hyperparameters of the model
                        "window_length": window_length,
                        "num_boost_round": num_boost_round,
                        "num_leaves": num_leaves,
                        "reg_lambda": reg_lambda,
                        "max_depth": max_depth,
                    },
                    experiment_id=experiment_id,
                    synchronous=False,  # Allow the run to fail if a model is not properly created
                )
                succeeded = p.wait()
                mlflow.log_params(
                    {
                        "window_length": window_length,
                        "num_boost_round": num_boost_round,
                        "num_leaves": num_leaves,
                        "reg_lambda": reg_lambda,
                        "max_depth": max_depth,
                    }
                )

            if succeeded:
                training_run = tracking_client.get_run(p.run_id)
                metrics = training_run.data.metrics
                # cap the loss at the loss of the null model
                mean_metric = metrics[f"mean_{metric}"]
            else:
                tracking_client.set_terminated(p.run_id, "FAILED")
                mean_metric = -1

            # log metrics of the model
            mlflow.log_metric(f"mean_{metric}", mean_metric)
            print(mean_metric)
            return mean_metric

        return eval

    space = {
        "window_length": hp.randint("window_length", 300),
        "num_boost_round": scope.int(hp.quniform("num_boost_round", 2, 5, 1)),
        "num_leaves": scope.int(hp.quniform("num_leaves", 2, 20, 1)),
        "max_depth": hp.randint("max_depth", 20),
        "reg_lambda": hp.randint("reg_lambda", 60),
    }

    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id

        best = fmin(
            fn=new_eval(experiment_id),
            space=space,
            algo=tpe.suggest if algo == "tpe.suggest" else rand.suggest,
            max_evals=max_runs,
        )
        mlflow.set_tag("best params", str(best))
        # find the best run, log its metrics as the final metrics of this run.
        client = MlflowClient()
        runs = client.search_runs(
            [experiment_id],
            "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id),
        )
        best_mean_rmse = _inf
        best_run = None
        for r in runs:
            if r.data.metrics["mean_rmse"] < best_mean_rmse:
                best_run = r.data.metrics["mean_rmse"]
        mlflow.set_tag("best_run", best_run.info.run_id)
        mlflow.log_metric("mean_rmse", best_mean_rmse)


if __name__ == "__main__":
    train()
