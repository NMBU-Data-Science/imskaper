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
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm.sklearn import LGBMClassifier
from pandas import DataFrame
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
    RFECV,
    GenericUnivariateSelect,
    chi2,
)
from sklearn.linear_model import LassoCV, RidgeClassifier, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skrebate import MultiSURF, ReliefF

from model_comparison import model_comparison_experiment

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
    NUM_REPS = config["config"]["NUM_REPS"]
    # The number of hyper-parameter configurations to try evaluate.
    MAX_EVALS = config["config"]["MAX_EVALS"]
    # Read from the CSV file that contains the features and the response.
    X_y = pd.read_csv(config["config"]["features_file"])

    X = X_y.iloc[:, : X_y.shape[1] - 1].values
    y = X_y.iloc[:, X_y.shape[1] - 1:].values
    y = y.reshape(-1)
    # import sklearn.datasets as ds
    # X, y = ds.load_breast_cancer(return_X_y=True)

    # Define a series of models (example includes only one) wrapped in a
    # Pipeline.
    scalar = (StandardScaler.__name__, StandardScaler())
    ridge_classifier = (RidgeClassifier.__name__, RidgeClassifier())
    lgbm_classifier = (LGBMClassifier.__name__, LGBMClassifier())
    cvs_classifier = (SVC.__name__, SVC())
    dt_classifier = (DecisionTreeClassifier.__name__, DecisionTreeClassifier())
    lr_classifier = (LogisticRegression.__name__, LogisticRegression())
    ridge_param = dict()
    lgbm_param = dict()
    svc_param = dict()
    dt_param = dict()
    lr_param = dict()

    ridge_param["RidgeClassifier__alpha"] = sp_randint(
        config["config"]["classifications"]["Ridge"]["alpha_from"],
        config["config"]["classifications"]["Ridge"]["alpha_to"],
    )
    # ridge_param["RidgeClassifier__tol"] = (0.01, 0.001, 0.0001)
    lgbm_param["LGBMClassifier__max_depth"] = sp_randint(
        config["config"]["classifications"]["LGBM"]["max_depth_from"],
        config["config"]["classifications"]["LGBM"]["max_depth_to"],
    )
    # lgbm_param["LGBMClassifier__reg_alpha"] = (10, 1, 0.1, 0.01, 0.001,
    # 0.0001)
    # lgbm_param["LGBMClassifier__reg_lambda"] = (10, 1, 0.1, 0.01, 0.001,
    # 0.0001)
    lgbm_param["LGBMClassifier__min_child_samples"] = sp_randint(2, 5)
    # lgbm_param["LGBMClassifier__min_data_in_bin"] = sp_randint(5, 7)
    lgbm_param["LGBMClassifier__num_leaves"] = sp_randint(
        config["config"]["classifications"]["LGBM"]["num_leaves_from"],
        config["config"]["classifications"]["LGBM"]["num_leaves_to"],
    )
    # lgbm_param["LGBMClassifier__lambda_l1"] = sp_randint(1, 3)
    svc_param["SVC__C"] = sp_randint(
        config["config"]["classifications"]["CVS"]["C_from"],
        config["config"]["classifications"]["CVS"]["C_to"],
    )

    # dt_param["DecisionTreeClassifier__ccp_alpha"] = sp_randint(1, 2)
    lr_param["LogisticRegression__C"] = sp_randint(1, 2)
    selector = []
    selector_param = []

    selector.append((SelectKBest.__name__, SelectKBest(mutual_info_classif)))
    selector_param.append(
        {
            "SelectKBest__k": sp_randint(
                config["config"]["selectors"]["SelectKBest"]["K_from"],
                config["config"]["selectors"]["SelectKBest"]["K_to"],
            )
        }
    )

    selector.append((ReliefF.__name__, ReliefF()))
    selector_param.append(
        {
            "ReliefF__n_neighbors": sp_randint(
                config["config"]["selectors"]["ReliefF"]["n_neighbors_from"],
                config["config"]["selectors"]["ReliefF"]["n_neighbors_to"],
            ),
            "ReliefF__n_features_to_select": sp_randint(
                config["config"]["selectors"]["ReliefF"][
                    "n_features_to_select_from"
                ],
                config["config"]["selectors"]["ReliefF"][
                    "n_features_to_select_to"
                ],
            ),
        }
    )

    selector.append((VarianceThreshold.__name__, VarianceThreshold()))
    selector_param.append(
        {
            "VarianceThreshold__threshold": sp_uniform(
                config["config"]["selectors"]["VarianceThreshold"][
                    "threshold_from"
                ],
                config["config"]["selectors"]["VarianceThreshold"][
                    "threshold_to"
                ],
            )
        }
    )

    # very slow
    selector.append((MultiSURF.__name__, MultiSURF()))
    selector_param.append(
        {
            "MultiSURF__n_features_to_select": sp_randint(
                config["config"]["selectors"]["MultiSURF"][
                    "n_features_to_select_from"
                ],
                config["config"]["selectors"]["MultiSURF"][
                    "n_features_to_select_to"
                ],
            )
        }
    )

    selector.append((SelectFromModel.__name__, SelectFromModel(LassoCV())))
    selector_param.append({})

    # very slow for large number of features
    from sklearn.svm import SVR
    selector.append((RFECV.__name__, RFECV(SVR(kernel="linear"))))
    selector_param.append({})

    selector.append((mutual_info_classif.__name__,
                     GenericUnivariateSelect(mutual_info_classif, 'k_best')))
    selector_param.append(
        {
            "mutual_info_classif__param": sp_randint(15, 35)
        }
    )
    from skfeature.function.similarity_based.fisher_score import fisher_score
    selector.append((fisher_score.__name__,
                     GenericUnivariateSelect(fisher_score, 'k_best')))
    selector_param.append(
        {
            "fisher_score__param": sp_randint(10, 20)
        }
    )

    df = DataFrame(dtype="float")

    # X, y = make_classification(n_samples=50, n_features=4, n_classes=2)
    np.random.seed(seed=0)
    random_states = np.random.choice(1000, size=NUM_REPS)

    # specify parameters and distributions to sample from
    for i in (0, 1, 2, 6, 7):
        path_to_results = Path(
            config["config"]["output_dir"],
            "results_" + selector[i][0] + str(time.strftime("%Y%m%d-%H%M%S")),
        ).with_suffix(".csv")
        models = {
            "ridge": Pipeline([scalar, selector[i], ridge_classifier]),
            "lgbm": Pipeline([scalar, selector[i], lgbm_classifier]),
            "svc": Pipeline([scalar, selector[i], cvs_classifier]),
            "dt": Pipeline([scalar, selector[i], dt_classifier]),
            "lr": Pipeline([scalar, selector[i], lr_classifier]),
        }
        hparams = {
            "ridge": merge_dict(ridge_param, selector_param[i]),
            "lgbm": merge_dict(lgbm_param, selector_param[i]),
            "svc": merge_dict(svc_param, selector_param[i]),
            "dt": merge_dict(dt_param, selector_param[i]),
            "lr": merge_dict(lr_param, selector_param[i]),
        }

        df = model_comparison_experiment(
            models=models,
            hparams=hparams,
            path_final_results=path_to_results,
            random_states=random_states,
            score_func=roc_auc_score,
            max_evals=MAX_EVALS,
            selector=selector[i][0],
            cv=CV,
            X=X,
            y=y,
            df=df,
        )

    path_to_results = Path(
        config["config"]["output_dir"],
        "results_" + "No_feature_selection" + str(time.strftime(
            "%Y%m%d-%H%M%S")),
    ).with_suffix(".csv")
    models = {
        "ridge": Pipeline([scalar, ridge_classifier]),
        "lgbm": Pipeline([scalar, lgbm_classifier]),
        "svc": Pipeline([scalar, cvs_classifier]),
        "dt": Pipeline([scalar, dt_classifier]),
        "lr": Pipeline([scalar, lr_classifier]),
    }
    hparams = {
        "ridge": ridge_param,
        "lgbm": lgbm_param,
        "svc": svc_param,
        "dt": dt_param,
        "lr": lr_param,
    }

    df = model_comparison_experiment(
        models=models,
        hparams=hparams,
        path_final_results=path_to_results,
        random_states=random_states,
        score_func=roc_auc_score,
        max_evals=MAX_EVALS,
        selector="No_feature_selection",
        cv=CV,
        X=X,
        y=y,
        df=df,
    )

    print(df)
    sns.heatmap(df.transpose(), annot=True)
    plt.tight_layout()
    plt.savefig("x.jpg")
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
