# -*- coding: utf-8 -*-
#
# comparison_schemes.py
#

"""
Features selection and classifications
"""

__author__ = 'Ahmed Albuni'
__email__ = 'ahmed.albuni@gmail.com'


import numpy as np
from pandas import DataFrame
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from skrebate import ReliefF
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
import time
import logging
import warnings
from model_comparison import model_comparison_experiment
from sklearn.feature_selection import VarianceThreshold
from skrebate import MultiSURF
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='Features selection and '
                                             'classifications (2 classes)')
parser.add_argument("-file", type=str, help='Features CSV file name and '
                                            'path, response var y should be '
                                            'the last field')


def experiment(X, y):
    # The number cross-validation folds.
    CV = 5
    # The number of times each experiment is repeated with a different
    # random seed.
    NUM_REPS = 1
    # The number of hyper-parameter configurations to try evaluate.
    MAX_EVALS = 2
    # Name of the CSV file containing the experimental results.

    path_to_results = 'results_'+str(time.strftime("%Y%m%d-%H%M%S"))+'.csv'

    # Define a series of models (example includes only one) wrapped in a
    # Pipeline.

    scalar = (StandardScaler.__name__, StandardScaler())
    ridge_classifier = (RidgeClassifier.__name__, RidgeClassifier())
    lgbm_classifier = (LGBMClassifier.__name__, LGBMClassifier())
    cvs_classifier = (SVC.__name__, SVC())
    ridge_param = dict()
    lgbm_param = dict()
    svc_param = dict()
    ridge_param['RidgeClassifier__alpha'] = sp_randint(1, 11)
    lgbm_param['LGBMClassifier__max_depth'] = sp_randint(20, 40)
    lgbm_param['LGBMClassifier__lambda_l1'] = sp_randint(1, 3)
    svc_param['SVC__C'] = sp_randint(2, 4)

    i = 1  # 3 takes forever!!
    selector = []
    selector_param = []

    selector.append((SelectKBest.__name__, SelectKBest(mutual_info_classif)))
    selector_param.append({'SelectKBest__k': sp_randint(2, X.shape[1])})

    selector.append((ReliefF.__name__, ReliefF()))
    selector_param.append({'ReliefF__n_neighbors': sp_randint(2, 4),
                           'ReliefF__n_features_to_select': sp_randint(4, 7)})

    selector.append((VarianceThreshold.__name__, VarianceThreshold()))
    selector_param.append({'VarianceThreshold__threshold': sp_uniform(0, 0.9)})

    selector.append((MultiSURF.__name__, MultiSURF()))
    selector_param.append({'MultiSURF__n_features_to_select': sp_randint(
        3, 7)})

    selector.append((SelectFromModel.__name__, SelectFromModel(LassoCV())))
    selector_param.append({})

    df = DataFrame(dtype=float)

    # X, y = make_classification(n_samples=50, n_features=4, n_classes=2)
    np.random.seed(seed=0)
    random_states = np.random.choice(1000, size=NUM_REPS)

    # specify parameters and distributions to sample from
    for i in (0, 1, 2, 4):
        models = {
            'ridge': Pipeline([scalar,
                               selector[i],
                               ridge_classifier]),
            'lgbm': Pipeline([scalar,
                              selector[i],
                              lgbm_classifier]),
            'svc': Pipeline([scalar,
                             selector[i],
                             cvs_classifier])
        }
        hparams = {'ridge': merge_dict(ridge_param, selector_param[i]),
                   'lgbm': merge_dict(lgbm_param, selector_param[i]),
                   'svc': merge_dict(svc_param, selector_param[i])
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
            df=df
        )

    print(df)
    sns.heatmap(df, annot=True)
    plt.tight_layout()
    plt.savefig('x.jpg')
    plt.show()


def merge_dict(dict1, dict2):
    merged = dict1.copy()
    merged.update(dict2)
    return merged


def balanced_roc_auc(y_true, y_pred):
    """Define a balanced ROC AUC optimization metric."""
    return roc_auc_score(y_true, y_pred, average='weighted')


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")

    args = parser.parse_args()
    X_y = pd.read_csv(args.file)

    X = X_y.iloc[:, :X_y.shape[1]-1].values
    y = X_y.iloc[:, X_y.shape[1]-1:].values
    y = y.reshape(-1)
    #  X, y = ds.load_breast_cancer(return_X_y=True)
    experiment(X, y)
