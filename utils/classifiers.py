# -*- coding: utf-8 -*-
#
# comparison_schemes.py
#

"""
Features selection and classifications
"""

__author__ = "Ahmed Albuni"
__email__ = "ahmed.albuni@gmail.com"


from lightgbm.sklearn import LGBMClassifier
from scipy.stats import randint as sp_randint

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def get_classifiers(config):

    ridge_param = dict()
    lgbm_param = dict()
    svc_param = dict()
    dt_param = dict()
    lr_param = dict()
    et_param = dict()

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
    lgbm_param["LGBMClassifier__min_child_samples"] = sp_randint(
        config["config"]["classifications"]["LGBM"]["min_child_s_from"],
        config["config"]["classifications"]["LGBM"]["min_child_s_to"],
    )
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
    lr_param["LogisticRegression__C"] = sp_randint(
        config["config"]["classifications"]["LR"]["C_from"],
        config["config"]["classifications"]["LR"]["C_to"],
    )

    classifiers = dict()
    classifiers["ridge"] = (
        (RidgeClassifier.__name__, RidgeClassifier()),
        ridge_param,
    )
    classifiers["lgbm"] = (
        (LGBMClassifier.__name__, LGBMClassifier()),
        lgbm_param,
    )
    classifiers["svc"] = (SVC.__name__, SVC()), svc_param
    classifiers["dt"] = (
        (DecisionTreeClassifier.__name__, DecisionTreeClassifier()),
        dt_param,
    )
    classifiers["lr"] = (
        (LogisticRegression.__name__, LogisticRegression()),
        lr_param,
    )
    classifiers["et"] = (
        (ExtraTreesClassifier.__name__, ExtraTreesClassifier()),
        et_param,
    )
    return classifiers
