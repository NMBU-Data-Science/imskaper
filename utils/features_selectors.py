# -*- coding: utf-8 -*-
#
# comparison_schemes.py
#

"""
Features selection and classifications
"""

__author__ = "Ahmed Albuni"
__email__ = "ahmed.albuni@gmail.com"


from scipy.stats import uniform as sp_uniform
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    GenericUnivariateSelect,
    VarianceThreshold,
)
from skrebate import ReliefF
from scipy.stats import randint as sp_randint
from skfeature.function.similarity_based.fisher_score import fisher_score


def get_features_selectors(config):

    k_best_param = {
        "SelectKBest__k": sp_randint(
            config["config"]["selectors"]["SelectKBest"]["K_from"],
            config["config"]["selectors"]["SelectKBest"]["K_to"],
        )
    }
    relieff_param = {
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
    mutual_info_param = {
        "mutual_info_classif__param": sp_randint(
            config["config"]["selectors"]["mutual_info"]["param_from"],
            config["config"]["selectors"]["mutual_info"]["param_to"],
        )
    }
    fisher_param = {
        "fisher_score__param": sp_randint(
            config["config"]["selectors"]["fisher_score"]["param_from"],
            config["config"]["selectors"]["fisher_score"]["param_to"],
        )
    }
    var_t_param = {
        "VarianceThreshold__threshold": sp_uniform(
            config["config"]["selectors"]["VarianceThreshold"][
                "threshold_from"
            ],
            config["config"]["selectors"]["VarianceThreshold"]["threshold_to"],
        )
    }
    f_list = dict()
    f_list["select_k_best"] = (
        (SelectKBest.__name__, SelectKBest(mutual_info_classif)),
        k_best_param,
    )
    f_list["relief_f"] = (ReliefF.__name__, ReliefF()), relieff_param

    f_list["mutual_info"] = (
        (
            mutual_info_classif.__name__,
            GenericUnivariateSelect(mutual_info_classif, "k_best"),
        ),
        mutual_info_param,
    )
    f_list["fisher_score"] = (
        (
            fisher_score.__name__,
            GenericUnivariateSelect(fisher_score, "k_best"),
        ),
        fisher_param,
    )
    f_list["variance_threshold"] = (
        (VarianceThreshold.__name__, VarianceThreshold()),
        var_t_param,
    )

    f_list["No feature selection"] = ("No_feature_selection", None), None

    return f_list
