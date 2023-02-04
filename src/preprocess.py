import pandas as pd
import click
from holidays import US


@click.command(
    help="Preprocesses the dataset in some way" "The input is expected in a xlsx format."
)
@click.argument("input_path")
@click.argument("output_path")
def preprocess(input_path: str, output_path: str):
    cash_df = pd.read_excel(input_path)

    # transform the column names to a more computer format
    cash_df.columns = [x.replace(" ", "_") for x in cash_df.columns.str.lower()]

    # filter only usd
    usd_mask = cash_df["currency"] == "USD"
    cash_df = cash_df[usd_mask][["date", "cash_inflow"]]

    cash_df = cash_df.assign(date=pd.to_datetime(cash_df["date"])).sort_values(by="date")

    # add different time dependent features; it can be extended
    cash_df = add_time_features(cash_df)

    cash_df.to_csv(output_path)
    return cash_df


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


if __name__ == "__main__":
    preprocess()
