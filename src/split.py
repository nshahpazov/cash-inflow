"""Split the dataset into train and evaluation test sets"""
import click
import pandas as pd
import constants
from sktime.forecasting.model_selection import temporal_train_test_split
import mlflow


@click.command()
@click.option(
    "--test_size",
    "test_size",
    required=True,
    default=28,
    help="test size",
)
@click.argument(
    "input_path",
    type=click.Path(exists=True),
    default="data/processed/cash_data.csv",
)
@click.argument(
    "output_x_train_path",
    type=click.Path(exists=False),
    default="data/model_input/x_train.csv",
)
@click.argument(
    "output_y_train_path",
    type=click.Path(exists=False),
    default="data/model_input/y_train.csv",
)
@click.argument(
    "output_x_test_path",
    type=click.Path(exists=False),
    default="data/model_input/x_test.csv",
)
@click.argument(
    "output_y_test_path",
    type=click.Path(exists=False),
    default="data/model_input/y_test.csv",
)
def split(
    input_path: str,
    output_x_train_path: str,
    output_y_train_path: str,
    output_x_test_path: str,
    output_y_test_path: str,
    test_size: int,
):
    """Split into train and test by a given test set size"""
    df = pd.read_csv(input_path)
    y = df[[constants.DATE_COLUMN_NAME, constants.TARGET_COLUMN_NAME]]
    X = df.drop(columns=[constants.TARGET_COLUMN_NAME])

    y_train, y_test, X_train, X_test = temporal_train_test_split(
        y=y, X=X, test_size=test_size
    )

    # write the splitted datasets
    X_train.to_csv(output_x_train_path, index=False)
    y_train.to_csv(output_y_train_path, index=False)
    X_test.to_csv(output_x_test_path, index=False)
    y_test.to_csv(output_y_test_path, index=False)


if __name__ == "__main__":
    split()
    print("Splitted baby")
