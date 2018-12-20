# -*- coding: utf-8 -*-
#
# feature_selection.py
#

"""
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import utils

import numpy as np
import pandas as pd

from scipy import stats
from ReliefF import ReliefF
from sklearn import feature_selection
from mlxtend.feature_selection import SequentialFeatureSelector


def wilcoxon_selection(X_train, X_test, y_train, y_test, thresh=0.05):
    """Perform feature selection by the Wilcoxon Rank-Sum Test.

    Args:
        X_train (array-like): Training predictor set.
        X_test (array-like): Test predictor set.
        y_train (array-like): Training target set.
        y_test (array-like): Test target set.
        thresh (float):

    """
    # Z-score transformation.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    _, ncols = np.shape(X_train_std)

    indicators = wilcoxon_rank_sum_test(X_train_std, y_train, thresh=thresh)
    support = _check_support(indicators, X_train_std)

    return _check_feature_subset(X_train_std, X_test_std, support)


def wilcoxon_rank_sum_test(X, y, thresh=0.05):
    """

    Also known as Mann-Whitney U test or Mann-Whitney Wilcoxon test.

    H0: Sample distributions are equal.
    H1: Sample distributions are not equal.

    NOTE: Performs Bonferroni correction.

    Args:
        X ():
        y ():
        thresh (float):

    Returns:
        (numpy.ndarray): Support indicators.

    """
    _, ncols = np.shape(X)

    support = []
    for num in range(ncols):
        # Equiv. to two-sample wilcoxon test.
        _, pval = stats.mannwhitneyu(X[:, num], y)
        # If p-value > thresh => same distribution.
        if pval <= thresh / ncols:
            support.append(num)

    return np.zeros(support, dtype=int)


# TODO: Verify correct sorting.
def mutual_info(
        X_train, X_test, y_train, y_test,
        num_neighbors, num_features,
        random_state=None
    ):
    """A wrapper of scikit-learn mutual information feature selector.

    Args:
        X_train (array-like): Training predictor set.
        X_test (array-like): Test predictor set.
        y_train (array-like): Training target set.
        y_test (array-like): Test target set.
        num_neighbors (int): The number of neighbors to consider when assigning
            feature importance scores.
        random_state (int):

    Returns:
        (tuple):

    """
    # Z-score transformation.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    info = feature_selection.mutual_info_classif(
        X_train_std, y_train, n_neighbors=num_neighbors,
        random_state=random_state
    )
    # NOTE: Retain features contributing above threshold to model performance.
    support = _check_support(sorted(info)[:num_features], X_train_std)

    return _check_feature_subset(X_train_std, X_test_std, support)


def relieff(X_train, X_test, y_train, y_test, num_neighbors, num_features):
    """A wrapper for the ReliefF feature selection algorithm.

    Args:
        X_train (array-like): Training predictor set.
        X_test (array-like): Test predictor set.
        y_train (array-like): Training target set.
        y_test (array-like): Test target set.
        num_neighbors (int): The number of neighbors to consider when assigning
            feature importance scores.
        num_features (): The number of features to select.

    """
    # Z-score transformation.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    selector = ReliefF(n_neighbors=num_neighbors)
    selector.fit(X_train_std, y_train)

    support = _check_support(selector.top_features[:num_features], X_train_std)

    return _check_feature_subset(X_train_std, X_test_std, support)


def permutation_importance_selection(
        X_train, X_test, y_train, y_test,
        scoring=None,
        model=None,
        thresh=0,
        num_rounds=5,
        random_state=None
    ):
    """Perform feature selection by feature permutation importance.

    Args:
        X_train (array-like): Training predictor set.
        X_test (array-like): Test predictor set.
        y_train (array-like): Training target set.
        y_test (array-like): Test target set.
        scoring ():
        model ():
        thresh (float):
        num_rounds (int):
        random_state (int):

    """
    # Z-score transformation.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    model.fit(X_train_std, y_train)
    imp = feature_permutation_importancen(
        X_test_std, y_test,
        scoring=scoring,
        model=model,
        num_rounds=num_rounds,
        random_state=random_state
    )
    # Retain only features positively contributing to model performance.
    support = _check_support(np.where(imp > thresh), X_train_std)

    return _check_feature_subset(X_train_std, X_test_std, support)


def feature_permutation_importance(
        X, y,
        scoring=None,
        model=None,
        num_rounds=10, random_state=None
    ):
    """Assess feature importance by random feature permutations.

    Args:
        X (array-like):
        y (array-like):
        scoring ():
        model ():
        num_rounds (int):
        random_state (int):

    Returns:
        (array-like): Average feature permutation importances.

    """
    # Setup:
    _, num_features = np.shape(X)
    rgen = np.random.RandomState(random_state)

    # Baseline performance.
    baseline = score_func(y, model.predict(X))

    importance = np.zeros(num_features, dtype=float)
    for round_idx in range(num_rounds):
        for col_idx in range(num_features):
            # Store original feature permutation.
            temp = X[:, col_idx].copy()
            # Perform random feature permutation.
            rgen.shuffle(X[:, col_idx])
             # Permutation score.
            new_score = score_func(y, model.predict(X))
            X[:, col_idx] = temp
            # Likely feature is important if new score < baseline.
            importance[col_idx] += baseline - new_score

    return importance / num_rounds


def _check_support(support, X):
    # Auxillary function formatting selected feature subset.

    if not isinstance(support, np.ndarray):
        support = np.array(support, dtype=int)

    # NB: Default mechanism includes all features if none were selected.
    if len(support) < 1:
        support = np.arange(X.shape[1], dtype=int)
    else:
        if np.ndim(support) > 1:
            support = np.squeeze(support)
        if np.ndim(support) < 1:
            support = support[np.newaxis]
        if np.ndim(support) != 1:
            raise RuntimeError(
                'Invalid dimension {} to support.'.format(np.ndim(support))
            )
    return support


def _check_feature_subset(X_train, X_test, support):
    # Auxillary function formatting training and test subsets.

    # Support should be a non-empty vector (ensured in _check_support).
    _X_train, _X_test = X_train[:, support],  X_test[:, support]

    if np.ndim(_X_train) > 2:
        if np.ndim(np.squeeze(_X_train)) > 2:
            raise RuntimeError('X train ndim {}'.format(np.ndim(_X_train)))
        else:
            _X_train = np.squeeze(_X_train)

    if np.ndim(_X_test) > 2:
        if np.ndim(np.squeeze(_X_test)) > 2:
            raise RuntimeError('X test ndim {}'.format(np.ndim(_X_train)))
        else:
            _X_test = np.squeeze(_X_test)

    if np.ndim(_X_train) < 2:
        if np.ndim(_X_train.reshape(-1, 1)) == 2:
            _X_train = _X_train.reshape(-1, 1)
        else:
            raise RuntimeError('X train ndim {}'.format(np.ndim(_X_train)))

    if np.ndim(_X_test) < 2:
        if np.ndim(_X_test.reshape(-1, 1)) == 2:
            _X_test = _X_test.reshape(-1, 1)
        else:
            raise RuntimeError('X test ndim {}'.format(np.ndim(_X_test)))

    return (
        np.array(_X_train, dtype=float), np.array(_X_test, dtype=float),
        support
    )



# ERROR: Must handle arbitrary scoring function.
def forward_floating(
        X_train, X_test, y_train, y_test,
        scoring=None,
        model=None,
        k=3, cv=10
    ):
    """A wrapper of mlxtend Sequential Forward Floating Selection algorithm.

    Args:
        X_train (array-like):
        X_test (array-like):
        y_train (array-like):
        y_test (array-like):

    """

    # Z-score transformation.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    # NOTE: Nested calls not supported by multiprocessing => joblib converts
    # into sequential code (thus, default n_jobs=1).
    #n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()
    n_jobs = 1

    selector = SequentialFeatureSelector(
        model, k_features=k, forward=True, floating=True, scoring='roc_auc',
        cv=cv, n_jobs=n_jobs
    )
    selector.fit(X_train_std, y_train)

    support = _check_support(selector.k_feature_idx_, X_train_std)

    return _check_feature_subset(X_train_std, X_test_std, support)




if __name__ == '__main__':
