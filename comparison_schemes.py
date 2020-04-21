# -*- coding: utf-8 -*-
#
# comparison_schemes.py
#

"""
Schemes for model comparison experiments.
Parts of the code taken from Severin Langberg:
https://github.com/gsel9/biorad
"""

__author__ = "Ahmed Albuni"
__email__ = "ahmed.albuni@gmail.com"


import os
from collections import OrderedDict
from datetime import datetime
from typing import Callable

import numpy as np
from pandas import DataFrame
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from utils import ioutil


def nested_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    columns_names: list,
    experiment_id: str,
    model: str,
    hparams: dict,
    score_func: Callable,
    df: DataFrame,
    selector: str,
    cv: int = 10,
    output_dir=None,
    max_evals: int = 100,
    verbose: int = 1,
    random_state=None,
    path_tmp_results: str = None,
):
    """
    Nested cross-validtion model comparison.

    Args:
        X: Feature matrix (n samples x n features).
        y: Ground truth vector (n samples).
        experiment_id: A unique name for the experiment.
        workflow: A scikit-learn pipeline or model and the associated
            hyperparameter space.
        score_func: Optimisation objective.
        cv (int): The number of cross-validation folds.
        random_states: A list of seed values for pseudo-random number
            generator.
        output_dir: Directory to store the output.
        path_tmp_results: Reference to preliminary experimental results.

    Returns:
        (dict):

    """

    # Run a new cross-validation experiment.
    # Name of file with preliminary results.
    if path_tmp_results is None:
        path_case_file = ""
    else:
        path_case_file = os.path.join(
            path_tmp_results, f"experiment_{random_state}_{experiment_id}"
        )
    # Theoutput written to file.
    output = {"random_state": random_state, "model_name": experiment_id}

    # Time the execution.
    if verbose > 0:
        start_time = datetime.now()
        print(f"Running experiment {random_state} with {experiment_id}")

    # Set random state for the model.
    model.random_state = random_state

    # Record model training and validation performance.

    selected_features = ""
    # Find optimal hyper-parameters and run inner K-folds.
    optimizer = RandomizedSearchCV(
        estimator=model,
        param_distributions=hparams,
        n_iter=max_evals,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        return_train_score=True,
    )
    optimizer.fit(X, y)
    # Include the optimal hyper-parameters in the output.
    output.update(**optimizer.best_params_)
    best_model = optimizer.best_estimator_

    if selector != "No_feature_selection":
        features = best_model.named_steps[selector]
    else:
        features = None
    if selector == "SelectKBest":
        f = get_selected_features_kbest(features.get_support(), columns_names)
        selected_features += ', '.join(f)
    elif selector == "ReliefF":
        # selected_features += str(features.feature_importances_)
        f = get_selected_features_reflieff(features.top_features_,
                                           columns_names,
                                           features.n_features_to_select)
        selected_features += ', '.join(f)
    if selector == "fisher_score":
        f = get_selected_features_kbest(features.get_support(), columns_names)
        selected_features += ', '.join(f)
    if selector == "mutual_info_classif":
        f = get_selected_features_kbest(features.get_support(), columns_names)
        selected_features += ', '.join(f)
    elif selector == "MultiSURF":
        # selected_features += str(features.top_features_)
        f = get_selected_features_reflieff(features.top_features_,
                                           columns_names,
                                           features.n_features_to_select)
        selected_features += ', '.join(f)

    elif selector == "SelectFromModel":
        pass

    # print(selected_features)
    # Record training and validation performance of the selected model.
    test_scores = optimizer.best_score_

    train_scores = optimizer.cv_results_.get("mean_train_score")[
        optimizer.best_index_
    ]

    # The model performance included in the output.
    output.update(
        OrderedDict(
            [
                ("test_score", "{:.5f}".format(test_scores)),
                ("train_score", "{:.5f}".format(train_scores)),
                (
                    "test_score_std",
                    "{:.5f}".format(
                        optimizer.cv_results_.get("std_test_score")[
                            optimizer.best_index_
                        ]
                    ),
                ),
                (
                    "train_score_std",
                    "{:.5f}".format(
                        optimizer.cv_results_.get("std_train_score")[
                            optimizer.best_index_
                        ]
                    ),
                ),
                ("selected features ", selected_features.replace("\n", ""),),
            ]
        )
    )
    df.at[experiment_id, selector] = test_scores
    if path_tmp_results is not None:
        ioutil.write_prelim_results(path_case_file, output)

        if verbose > 0:
            duration = datetime.now() - start_time
            days = duration.days
            hours, rem = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(rem, 60)

            print(f"Experiment {random_state} completed in {duration}")
            output["exp_duration"] = "{} days {:02d}:{:02d}:{:02d}".format(
                days, hours, minutes, seconds
            )
    return output, df


def get_selected_features_kbest(selector_array, features_list):

    selected_features = []
    for i, val in enumerate(selector_array):
        if val:
            selected_features.append(features_list[i])
    return selected_features


def get_selected_features_reflieff(slector_array, features_list, num):

    selected_features = []
    for i, val in enumerate(slector_array):
        if i < num:
            selected_features.append(features_list[val])

    return selected_features

