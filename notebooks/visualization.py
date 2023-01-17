"""
Module with helper visualization functions to be used in our analysis
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pandas import DataFrame
import seaborn as sns
from warnings import simplefilter
from sktime.utils.plotting import plot_series
from matplotlib.ticker import MaxNLocator

import numpy as np
from pandas import DataFrame


def plot_best_n_models(metrics_results, forecasts, y_test, metric: str, n: int, model="all"):
    """Visualize best n models based on a selected metric"""

    top_n_model_names = metrics_results[metric].sort_values().index[:n]
    top_n_models = {k: forecasts[k] for k in forecasts.keys() & top_n_model_names}
    if model != "all":
        top_n_models = {}
        top_n_models[model] = forecasts[model]

    fig, ax = plot_series(
        y_test,
        *tuple(top_n_models.values()),
        labels=['y=Observed'] + list(top_n_models.keys())
    )

    fig.set_figwidth(22)
    fig.set_figheight(6)

    formatter = FuncFormatter(lambda x, pos: '%1.1fM' % (x * 1e-6))
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cash Inflow")
    ax.set_title(f"Observed Cash Inflow on Test set and top {n} models predictions")

    return ax

def millions(x, pos):
    """The two args are the value and tick position"""
    return '%1.1fM' % (x * 1e-6)


def plot_time_series(ts, title, ylabel, figsize=(22, 6)):
    top_n_model_names = metrics_results[metric].sort_values().index[:n]
    top_n_models = {k: forecasts[k] for k in forecasts.keys() & top_n_model_names}
    if model != "all":
        top_n_models = {}
        top_n_models[model] = forecasts[model]

    fig, ax = plot_series(
        y_test,
        *tuple(top_n_models.values()),
        labels=['y=Observed'] + list(top_n_models.keys())
    )

    fig.set_figwidth(22)
    fig.set_figheight(6)

    formatter = FuncFormatter(lambda x, pos: '%1.1fM' % (x * 1e-6))
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cash Inflow")
    ax.set_title(f"Observed Cash Inflow on Test set and top {n} models predictions")

    return ax
    """
    Plot a time series and set the formatter
    """
    ax = ts.plot(figsize=(22, 6), color="#339966")
    formatter = FuncFormatter(millions)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return ax


def plot_currency_series(data: DataFrame, column: str, currency: str):
    """Plot a time series of the dataset, filtered by a particular currency account"""
    selected_currency_mask = data["currency"] == currency
    filtered_currency_ts = data[selected_currency_mask].set_index("date")[column]
    ax = filtered_currency_ts.plot(figsize=(22, 6), color="#339966")

    formatter = FuncFormatter(millions)

    ax.yaxis.set_major_formatter(formatter)

    ax.set_title(f"{column} for {currency}")
    ax.set_ylabel(column)


def plot_windows(y, train_windows, test_windows, title=""):
    """Visualize training and test windows"""

    simplefilter("ignore", category=UserWarning)

    def get_y(length, split):
        # Create a constant vector based on the split for y-axis."""
        return np.ones(length) * split

    n_splits = len(train_windows)
    n_timepoints = len(y)
    len_test = len(test_windows[0])

    train_color, test_color = sns.color_palette("colorblind")[:2]

    fig, ax = plt.subplots(figsize=plt.figaspect(0.3))

    for i in range(n_splits):
        train = train_windows[i]
        test = test_windows[i]

        ax.plot(
            np.arange(n_timepoints),
            get_y(n_timepoints, i),
            marker="o",
            c="lightgray",
        )
        ax.plot(
            train,
            get_y(len(train), i),
            marker="o",
            c=train_color,
            label="Window",
        )
        ax.plot(
            test,
            get_y(len_test, i),
            marker="o",
            c=test_color,
            label="Forecasting horizon",
        )
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(
        title=title,
        ylabel="Window number",
        xlabel="Time",
        xticklabels=y.index,
    )
    # remove duplicate labels/handles
    handles, labels = [(leg[:2]) for leg in ax.get_legend_handles_labels()]
    ax.legend(handles, labels);
    return ax


def get_windows(y, cv):
    """Generate windows for cross-validation"""
    train_windows = []
    test_windows = []
    for i, (train, test) in enumerate(cv.split(y)):
        train_windows.append(train)
        test_windows.append(test)
    return train_windows, test_windows
