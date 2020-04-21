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
        selected_features += str(features.get_support())

    elif selector == "ReliefF":
        selected_features += str(features.top_features_)
        selected_features += str(features.n_features_to_select)
        selected_features += str(features.feature_importances_)

    elif selector == "VarianceThreshold":
        selected_features += str(features._get_param_names())
        selected_features += str(features._get_support_mask())

    elif selector == "MultiSURF":
        selected_features += str(features.n_features_to_select)
        selected_features += str(features.feature_importances_)
        selected_features += str(features.top_features_)

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


def nested_cross_validation_old(
    X: np.ndarray,
    y: np.ndarray,
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
        output_dir: Directory to store SMAC output.
        path_tmp_results: Reference to preliminary experimental results.

    Returns:
        (dict):

    """

    # Name of file with preliminary results.
    if path_tmp_results is None:
        path_case_file = ""
    else:
        path_case_file = os.path.join(
            path_tmp_results, f"experiment_{random_state}_{experiment_id}"
        )

    # Check if prelimnary results aleady exists. If so, load results and
    # proceed to next experiment.
    if os.path.isfile(path_case_file):
        output = ioutil.read_prelim_result(path_case_file)
        print(f"Reloading results from: {path_case_file}")

    # Run a new cross-validation experiment.
    else:
        # Theoutput written to file.
        output = {"random_state": random_state, "model_name": experiment_id}

        # Time the execution.
        if verbose > 0:
            start_time = datetime.now()
            print(f"Running experiment {random_state} with {experiment_id}")

        # Set random state for the model.
        model.random_state = random_state

        # Record model training and validation performance.
        # test_scores, train_scores = [], []
        selected_features = ""
        # Run outer K-folds.
        # kfolds = StratifiedKFold(cv, shuffle=True, random_state=random_state)
        # for (train_idx, test_idx) in kfolds.split(X, y):

        #    X_train, X_test = X[train_idx], X[test_idx]
        #    y_train, y_test = y[train_idx], y[test_idx]
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_state, stratify=y
        )
        # Find optimal hyper-parameters and run inner K-folds.
        optimizer = RandomizedSearchCV(
            estimator=model,
            param_distributions=hparams,
            n_iter=max_evals,
            scoring="roc_auc",
            cv=cv,
            random_state=random_state,
        )
        optimizer.fit(X_train, y_train)
        print(optimizer.cv_results_)
        # Include the optimal hyper-parameters in the output.
        output.update(**optimizer.best_params_)
        best_model = optimizer.best_estimator_
        # best_model.fit(X_train, y_train)
        if selector != "No_feature_selection":
            features = best_model.named_steps[selector]
        else:
            features = None
        if selector == "SelectKBest":
            selected_features += str(features.get_support())

        elif selector == "ReliefF":
            selected_features += str(features.top_features_)
            selected_features += str(features.n_features_to_select)
            selected_features += str(features.feature_importances_)

        elif selector == "VarianceThreshold":
            selected_features += str(features._get_param_names())
            selected_features += str(features._get_support_mask())

        elif selector == "MultiSURF":
            selected_features += str(features.n_features_to_select)
            selected_features += str(features.feature_importances_)
            selected_features += str(features.top_features_)

        elif selector == "SelectFromModel":
            pass

        # print(selected_features)
        # Record training and validation performance of the selected model.
        # test_scores = score_func(y_test, best_model.predict(X_test))
        test_scores = optimizer.score(X_test, y_test)

        train_scores = score_func(y_test, best_model.predict(X_test))

        # The model performance included in the output.
        output.update(
            OrderedDict(
                [
                    ("test_score", "{:.5f}".format(test_scores)),
                    ("train_score", "{:.5f}".format(train_scores)),
                    (
                        "test_score_variance",
                        "{:.5f}".format(np.var(test_scores)),
                    ),
                    (
                        "train_score_variance",
                        "{:.5f}".format(np.var(train_scores)),
                    ),
                    (
                        "selected features ",
                        selected_features.replace("\n", ""),
                    ),
                ]
            )
        )
        df.at[experiment_id, selector] = np.mean(test_scores)
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


def get_selected_features(selector_array):
    selected_features = []
    return selected_features
