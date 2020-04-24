# -*- coding: utf-8 -*-
#
# comparison_schemes.py
#

"""
Features selection and classifications
"""

__author__ = "Ahmed Albuni"
__email__ = "ahmed.albuni@gmail.com"


import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiment.model_comparison import model_comparison_experiment
from utils import features_selectors
from utils import classifiers


def experiment(config):
    # The number cross-validation folds.
    CV = config["config"]["CV"]
    # The number of times each experiment is repeated with a different
    # random seed.
    random_state = config["config"]["SEED"]
    # The number of hyper-parameter configurations to try evaluating.
    MAX_EVALS = config["config"]["MAX_EVALS"]
    # Read from the CSV file that contains the features and the response.
    X_y = pd.read_csv(config["config"]["features_file"])
    # Store column names to be used to get selected features.
    columns_names = X_y.columns.tolist()
    # the response y should be the last field in the dataset csv file.
    X = X_y.iloc[:, : X_y.shape[1] - 1].values
    y = X_y.iloc[:, X_y.shape[1] - 1 :].values
    y = y.reshape(-1)

    scalar = (StandardScaler.__name__, StandardScaler())
    # Get lists of feature selectors and classifier to be used in the pipeline.
    f_list = features_selectors.get_features_selectors(config)
    c_list = classifiers.get_classifiers(config)

    # df to store the results for the final graph of the corss-validation.
    scores_df = DataFrame(dtype="float")

    # Loop over the feature selectors.
    for f_k, f_v in f_list.items():
        path_to_results = Path(
            config["config"]["output_dir"],
            "results_" + f_v[0][0] + "_" + str(time.strftime("%Y%m%d-%H%M%S")),
        ).with_suffix(".csv")

        models = dict()
        hparams = dict()
        # Loop over the classifications algorithms.
        for k, v in c_list.items():
            if f_k == "No feature selection":
                models[k] = Pipeline([scalar, v[0]])
                hparams[k] = v[1]
            else:
                # We should not scale the data before VarianceThreshold
                if f_v[0][0] == "VarianceThreshold":
                    models[k] = Pipeline([f_v[0], scalar, v[0]])
                else:
                    models[k] = Pipeline([scalar, f_v[0], v[0]])
                hparams[k] = merge_dict(v[1], f_v[1])

        scores_df = model_comparison_experiment(
            models=models,
            hparams=hparams,
            path_final_results=path_to_results,
            random_state=random_state,
            score_func="roc_auc",
            max_evals=MAX_EVALS,
            selector=f_v[0][0],
            cv=CV,
            X=X,
            y=y,
            columns_names=columns_names,
            df=scores_df,
        )

    plot_heat_map(scores_df, config)


def merge_dict(dict1, dict2):
    merged = dict1.copy()
    merged.update(dict2)
    return merged


def plot_heat_map(scores_df, config):
    # Plot a heat-map of the scores obtained from the various feature
    # selectors and classifiers.
    sns.heatmap(scores_df.transpose() * 100, annot=True, fmt=".1f")
    plt.xlabel("Classification Algorithms")
    plt.ylabel("Feature Selection Algorithms")
    plt.title("AUC", x=1.1, y=1.1)
    plt.tight_layout()
    path_to_image = Path(
        config["config"]["output_dir"],
        "image_" + str(time.strftime("%Y%m%d-%H%M%S")),
    ).with_suffix(".jpg")
    plt.savefig(path_to_image, dpi=200)
    plt.show()
