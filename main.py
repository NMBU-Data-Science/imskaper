# -*- coding: utf-8 -*-
#
# comparison_schemes.py
#

"""
Features selection and classifications
"""

__author__ = "Ahmed Albuni"
__email__ = "ahmed.albuni@gmail.com"


import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_comparison import model_comparison_experiment
from utils import features_selectors
from utils import classifiers

parser = argparse.ArgumentParser(
    description="Features selection and " "classifications (2 classes)"
)
parser.add_argument(
    "-file",
    type=str,
    help="Features CSV file name and "
    "path, response var y should be "
    "the last field",
)


def experiment(config):
    # The number cross-validation folds.
    CV = config["config"]["CV"]
    # The number of times each experiment is repeated with a different
    # random seed.
    SEED = config["config"]["SEED"]
    # The number of hyper-parameter configurations to try evaluate.
    MAX_EVALS = config["config"]["MAX_EVALS"]
    # Read from the CSV file that contains the features and the response.
    X_y = pd.read_csv(config["config"]["features_file"])
    columns_names = X_y.columns.tolist()
    X = X_y.iloc[:, : X_y.shape[1] - 1].values
    y = X_y.iloc[:, X_y.shape[1] - 1 :].values
    y = y.reshape(-1)

    # Define a series of models wrapped in a pipeline
    scalar = (StandardScaler.__name__, StandardScaler())
    f_list = features_selectors.get_features_selectors(config)
    c_list = classifiers.get_classifiers(config)

    df = DataFrame(dtype="float")

    # np.random.seed(seed=0)
    random_state = SEED

    # specify parameters and distributions to sample from
    for f_k, f_v in f_list.items():
        path_to_results = Path(
            config["config"]["output_dir"],
            "results_" + f_v[0][0] + "_" + str(time.strftime("%Y%m%d-%H%M%S")),
        ).with_suffix(".csv")

        models = dict()
        hparams = dict()
        for k, v in c_list.items():
            if f_k == 'No feature selection':
                models[k] = Pipeline([scalar, v[0]])
                hparams[k] = v[1]
            else:
                models[k] = Pipeline([scalar, f_v[0], v[0]])
                hparams[k] = merge_dict(v[1], f_v[1])

        df = model_comparison_experiment(
            models=models,
            hparams=hparams,
            path_final_results=path_to_results,
            random_state=random_state,
            score_func=roc_auc_score,
            max_evals=MAX_EVALS,
            selector=f_v[0][0],
            cv=CV,
            X=X,
            y=y,
            columns_names=columns_names,
            df=df,
        )

    print(df)
    sns.heatmap(df.transpose(), annot=True)
    plt.tight_layout()
    path_to_image = Path(
        config["config"]["output_dir"],
        "image_" + str(time.strftime("%Y%m%d-%H%M%S")),
    ).with_suffix(".jpg")
    plt.savefig(path_to_image)
    plt.show()


def merge_dict(dict1, dict2):
    merged = dict1.copy()
    merged.update(dict2)
    return merged


def balanced_roc_auc(y_true, y_pred):
    """Define a balanced ROC AUC optimization metric."""
    return roc_auc_score(y_true, y_pred, average="weighted")


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")

    args = parser.parse_args()

    with open(args.file) as config_file:
        config = json.load(config_file)

    experiment(config)
