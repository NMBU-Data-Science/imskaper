# -*- coding: utf-8 -*-
#
# experiment.py
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
import numpy
from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiment.model_comparison import model_comparison_experiment
from utils import features_selectors
from utils import classifiers


def experiment(config, verbose=1):
    # The number cross-validation folds.
    CV = config["config"]["CV"]
    # The number of times each experiment is repeated with a different
    # random seed.
    random_state = config["config"]["SEED"]
    # setting the random seed globally to reproduce the results, setting
    # this value in the RandomizedSearchCV will did not provide that,
    # number of jobs should be 1 to get the exact results each time
    numpy.random.seed(random_state)
    # The number of hyper-parameter configurations to try evaluating.
    MAX_EVALS = config["config"]["MAX_EVALS"]
    # N_Jobs for parallelisation
    n_jobs = config["config"]["N_JOBS"]
    # Read from the CSV file that contains the features and the response.
    X, y, columns_names = read_Xy_data(config["config"]["features_file"])

    # Get lists of feature selectors and classifier to be used in the pipeline.
    feature_list = features_selectors.get_features_selectors(config)
    classifier_list = classifiers.get_classifiers(config)

    # df to store the results for the final graph of the cross-validation.
    scores_df = DataFrame(dtype="float")

    # Loop over the feature selectors.
    for feature_selector_k, feature_selector_v in feature_list.items():
        path_to_results = Path(
            config["config"]["output_dir"],
            "results_"
            + feature_selector_v[0][0]
            + "_"
            + str(time.strftime("%Y%m%d-%H%M%S")),
        ).with_suffix(".csv")

        # Loop over the classifiers and prepare the pipelines
        models, hparams = get_models(
            feature_selector_k, feature_selector_v, classifier_list
        )

        scores_df = model_comparison_experiment(
            models=models,
            hparams=hparams,
            path_final_results=path_to_results,
            random_state=random_state,
            score_func="roc_auc",
            max_evals=MAX_EVALS,
            selector=feature_selector_v[0][0],
            cv=CV,
            X=X,
            y=y,
            columns_names=columns_names,
            df=scores_df,
            verbose=verbose,
            n_jobs=n_jobs,
        )
    plot_heat_map(scores_df, config, verbose)
    return scores_df


def merge_dict(dict1, dict2):
    merged = dict1.copy()
    merged.update(dict2)
    return merged


def plot_heat_map(scores_df, config, verbose):
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
    if verbose > 0:
        plt.show()


def read_Xy_data(file):
    X_y = pd.read_csv(file)
    # Store column names to be used to get selected features.
    columns_names = X_y.columns.tolist()
    columns_names.pop()
    # the response y should be the last field in the dataset csv file.
    X = X_y.iloc[:, : X_y.shape[1] - 1].values
    y = X_y.iloc[:, X_y.shape[1] - 1 :].values
    y = y.reshape(-1)
    return X, y, columns_names


def get_models(feature_selector_k, feature_selector_v, classifiers_list):
    # Loop over the classifications algorithms.
    scalar = (StandardScaler.__name__, StandardScaler())
    models = dict()
    hparams = dict()
    for classifier_k, classifier_v in classifiers_list.items():
        if feature_selector_k == "No feature selection":
            models[classifier_k] = Pipeline([scalar, classifier_v[0]])
            hparams[classifier_k] = classifier_v[1]
        else:
            # We should not scale the data before VarianceThreshold
            if feature_selector_v[0][0] == "VarianceThreshold":
                models[classifier_k] = Pipeline(
                    [feature_selector_v[0], scalar, classifier_v[0]]
                )
            else:
                models[classifier_k] = Pipeline(
                    [scalar, feature_selector_v[0], classifier_v[0]]
                )
            hparams[classifier_k] = merge_dict(
                classifier_v[1], feature_selector_v[1]
            )
    return models, hparams
