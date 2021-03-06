# -*- coding: utf-8 -*-
#
# model_comparison.py
#

"""
Work function for model comparison experiments.
"""

__author__ = "Severin Langberg, Ahmed Albuni"
__email__ = "langberg91@gmail.com, ahmed85@gmail.com"


from typing import Dict

import numpy as np
from pandas import DataFrame
from tqdm import tqdm

from .comparison_schemes import cross_validation
from ..utils import ioutil


def model_comparison_experiment(
    X: np.ndarray,
    y: np.ndarray,
    columns_names: list,
    models: Dict,
    hparams: Dict,
    score_func: str,
    cv: int,
    max_evals: int,
    df: DataFrame,
    selector: str,
    random_state: int = 0,
    path_final_results: str = None,
    verbose: int = 1,
    n_jobs: int = 1,
):
    """
    Compare model performances with optional feature selection.

    Args:
        X: Feature matrix (n samples x m features).
        y: Ground truth vector (n samples).
        models: Key-value pairs with model name and a scikit-learn Pipeline.
        columns_names: name of the features
        hparams: Optimisation objective.
        score_func: score function
        cv: number of cross validation splitting
        max_evals:
        df: dataframe to temporary store the results for plotting
        selector: feature selector
        random_state: random state value
        n_jobs: number of cpu units used for processing
        path_final_results: output directory
        verbose:

    """
    # Setup temporary directory to store preliminary results.
    path_tmp_results = ioutil.setup_tempdir("tmp_comparison", root=".")

    # Iterate through models and run experiments in parallel.
    results = []
    all_selected_features = list()
    for model_name, model in tqdm(models.items()):

        # Get hyper-parameters for this model.
        model_hparams = hparams[model_name]
        result, df, selected_features = cross_validation(
            X=X,
            y=y,
            columns_names=columns_names,
            model=model,
            experiment_id=model_name,
            hparams=model_hparams,
            cv=cv,
            score_func=score_func,
            max_evals=max_evals,
            random_state=random_state,
            path_tmp_results=path_tmp_results,
            df=df,
            selector=selector,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        results.append(result)
        all_selected_features += selected_features
    if verbose > 0:
        print(df)
    # Remove temporary directory.
    ioutil.teardown_tempdir(path_tmp_results)
    # Write final results to disk.
    ioutil.write_final_results(path_final_results, results)
    return df, all_selected_features
