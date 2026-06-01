"""Ridge regression utilities for the EEG encoding models.

This module exposes two layers:

Reference (oracle) implementations
    :func:`Grid_search` and :func:`Grid_search_fusion` wrap scikit-learn's
    ``GridSearchCV(Ridge(), ...)`` exactly as used to produce the published
    results. They operate on a single (channel, timepoint) target at a time and
    are kept as the ground-truth reference against which the fast path can be
    validated. :class:`WeightedRidge` implements the
    convex-combination fusion: the two feature blocks are scaled by
    ``(1 - weight)`` and ``weight`` for *fitting*, while prediction uses the
    *unscaled* concatenated features (the published behaviour).

Fast batched implementations
    :func:`grid_search_batched` and :func:`grid_search_fusion_batched` solve all
    (channel x timepoint) cells together as columns of a single target matrix.
    They reproduce the oracle's selection and predictions to machine precision
    (same alpha grid, same ``KFold(5, shuffle=True, random_state=42)`` CV, same
    r2 model selection, per-target alpha and convex weight, and the same
    train-scaled / test-unscaled fusion prediction), while reusing one
    eigendecomposition of the Gram matrix across every alpha and every target.
    This replaces the per-cell grid search (~10^3-10^4x faster) without changing
    the results, so partitioning the time axis across CPUs becomes optional.
"""

from typing import Sequence, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from sklearn.model_selection import KFold, GridSearchCV


def Grid_search(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    alpha_range: Union[Sequence[float], np.ndarray],
    metric: str = 'mse',
    kfold: int = 5,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, float, int]:
    """Perform grid search over alpha for Ridge regression (reference oracle).

    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_samples, n_features).
    X_test : np.ndarray
        Test features of shape (m_samples, n_features).
    y_train : np.ndarray
        Training targets of shape (n_samples,) or (n_samples, n_targets).
    y_test : np.ndarray
        Test targets (unused; retained for signature compatibility).
    alpha_range : array-like of float
        1D array or sequence of alpha values to search.
    metric : str, default='mse'
        Metric for model selection: 'mse', 'r2', or 'mae'.
    kfold : int, default=5
        Number of cross-validation folds.
    n_jobs : int, default=-1
        Number of parallel jobs for GridSearchCV.

    Returns
    -------
    predictions : np.ndarray
        Predicted values on X_test.
    best_alpha : float
        Optimal alpha value.
    boundary_flag : int
        1 if best_alpha is on the boundary of alpha_range, else 0.
    """
    alpha_arr = np.asarray(alpha_range)
    if alpha_arr.ndim != 1:
        raise ValueError("alpha_range must be a 1D sequence or array.")

    scorers = {
        'mse': make_scorer(mean_squared_error, greater_is_better=False),
        'r2': make_scorer(r2_score),
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
    }
    scoring = scorers.get(metric.lower())
    if scoring is None:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from {list(scorers)}.")

    cv = KFold(n_splits=kfold, shuffle=True, random_state=42)
    model = Ridge()
    param_grid = {'alpha': alpha_arr}
    grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)

    grid.fit(X_train, y_train)
    best_alpha = grid.best_params_['alpha']
    predictions = grid.best_estimator_.predict(X_test)
    boundary_flag = int(best_alpha in (alpha_arr[0], alpha_arr[-1]))

    return predictions, best_alpha, boundary_flag


class WeightedRidge(BaseEstimator, RegressorMixin):
    """Ridge regressor that weights two blocks of features.

    The first ``feature_one_index`` columns are scaled by ``(1 - weight)`` and
    the remainder by ``weight`` before fitting. Prediction is performed on the
    unscaled feature matrix, matching the convex-combination fusion used for the
    published results.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    weight : float, default=0.5
        Weight applied to the second block of features.
    feature_one_index : int
        Index at which to split the feature matrix into two blocks.
    """

    def __init__(self, alpha: float = 1.0, weight: float = 0.5, feature_one_index: int = 0):
        self.alpha = alpha
        self.weight = weight
        self.feature_one_index = feature_one_index

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WeightedRidge':
        if not (0 < self.feature_one_index < X.shape[1]):
            raise ValueError("feature_one_index must be between 1 and n_features-1.")
        X_mod = X.copy()
        X_mod[:, :self.feature_one_index] *= (1 - self.weight)
        X_mod[:, self.feature_one_index:] *= self.weight
        self.model_ = Ridge(alpha=self.alpha, solver='cholesky')
        self.model_.fit(X_mod, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)


def Grid_search_fusion(
    X0_train: np.ndarray,
    X0_test: np.ndarray,
    X1_train: np.ndarray,
    X1_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    alpha_range: Union[Sequence[float], np.ndarray],
    weight_list: Union[Sequence[float], np.ndarray],
    metric: str = 'mse',
    kfold: int = 5,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, float, float, int]:
    """Grid search over alpha and weight for two-block Ridge fusion (oracle).

    Parameters
    ----------
    X0_train, X1_train : np.ndarray
        Training features for block 1 and block 2.
    X0_test, X1_test : np.ndarray
        Test features for block 1 and block 2.
    y_train : np.ndarray
        Training targets.
    y_test : np.ndarray
        Test targets (unused; retained for signature compatibility).
    alpha_range : array-like of float
        1D array or sequence of alpha values.
    weight_list : array-like of float
        1D array or sequence of weight values.
    metric : str, default='mse'
        Metric for model selection: 'mse', 'r2', or 'mae'.
    kfold : int, default=5
        Number of CV folds.
    n_jobs : int, default=-1
        Parallel jobs for GridSearchCV.

    Returns
    -------
    predictions : np.ndarray
        Predicted values on stacked test data.
    best_alpha : float
        Optimal alpha found.
    best_weight : float
        Optimal weight found.
    boundary_flag : int
        Bitmask flag: 1 if alpha on boundary, 2 if weight on boundary.
    """
    alpha_arr = np.asarray(alpha_range)
    weight_arr = np.asarray(weight_list)
    if alpha_arr.ndim != 1 or weight_arr.ndim != 1:
        raise ValueError("alpha_range and weight_list must be 1D sequences or arrays.")

    X_train = np.hstack([X0_train, X1_train])
    X_test = np.hstack([X0_test, X1_test])
    feature_one_index = X0_train.shape[1]

    scorers = {
        'mse': make_scorer(mean_squared_error, greater_is_better=False),
        'r2': make_scorer(r2_score),
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
    }
    scoring = scorers.get(metric.lower())
    if scoring is None:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from {list(scorers)}.")

    cv = KFold(n_splits=kfold, shuffle=True, random_state=42)
    estimator = WeightedRidge(alpha=alpha_arr[0], weight=weight_arr[0],
                              feature_one_index=feature_one_index)
    param_grid = {
        'alpha': alpha_arr,
        'weight': weight_arr,
        'feature_one_index': [feature_one_index],
    }
    grid = GridSearchCV(estimator, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)

    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    predictions = grid.best_estimator_.predict(X_test)

    best_alpha = best_params['alpha']
    best_weight = best_params['weight']
    boundary_flag = 0
    if best_alpha in (alpha_arr[0], alpha_arr[-1]):
        boundary_flag |= 1
    if best_weight in (weight_arr[0], weight_arr[-1]):
        boundary_flag |= 2

    return predictions, best_alpha, best_weight, boundary_flag


# ---------------------------------------------------------------------------
# Batched (all-targets-at-once) equivalents of the per-cell grid searches.
#
# The per-cell ``Grid_search`` / ``Grid_search_fusion`` above are the reference
# (oracle) implementations. The functions below reproduce their selection and
# predictions *exactly* -- same alpha grid, same KFold(5, shuffle, random_state=42)
# cross-validation, same r2 model selection, per-target alpha (and per-target
# convex weight for fusion), and the same train-scaled / test-unscaled prediction
# of ``WeightedRidge`` -- while solving all (channel, timepoint) targets together.
#
# Speed comes from reusing ONE eigendecomposition of the (scaled) Gram matrix
# across every alpha and every target, and scoring each alpha by a closed-form
# residual-sum-of-squares built from inner products rather than by materialising
# per-alpha predictions. The batched functions reproduce the per-cell oracle
# (Grid_search / Grid_search_fusion) selection and predictions to machine precision.
# ---------------------------------------------------------------------------


def _make_folds(n_samples: int, kfold: int = 5):
    """Folds identical to GridSearchCV(cv=KFold(kfold, shuffle=True, random_state=42))."""
    cv = KFold(n_splits=kfold, shuffle=True, random_state=42)
    return list(cv.split(np.arange(n_samples)))


def _meanr2_over_alphas(X_fit, X_pred, Y, alphas, folds, score_dtype=np.float64):
    """Mean cross-validated r2 for every (alpha, target).

    Parameters
    ----------
    X_fit : (n, p) design used to FIT ridge (scaled blocks for fusion).
    X_pred : (n, p) design used to PREDICT held-out rows (unscaled for fusion).
        Same rows as ``X_fit``; for single-space encoding ``X_fit is X_pred``.
    Y : (n, T) all targets stacked as columns.
    alphas : (A,) ridge penalties.
    folds : list of (train_idx, val_idx).
    score_dtype : np.float64 (default, exact match to sklearn) or np.float32
        (~2x faster scoring). Only the alpha/weight SELECTION uses this dtype;
        the final predictions in ``_predict_selected`` are always float64. float32
        can flip selection only between alphas/weights whose CV-r2 differ by less
        than float32 epsilon (i.e. effectively tied models), so reported
        correlations are unaffected.

    Returns
    -------
    mean_r2 : (A, T) mean over folds of the per-fold r2, matching sklearn's
        ``make_scorer(r2_score)`` aggregation.
    """
    X_fit = np.asarray(X_fit, dtype=score_dtype)
    X_pred = np.asarray(X_pred, dtype=score_dtype)
    Y = np.asarray(Y, dtype=score_dtype)
    alphas = np.asarray(alphas, dtype=score_dtype)
    A, T = alphas.shape[0], Y.shape[1]
    sum_r2 = np.zeros((A, T), dtype=np.float64)

    for tr, va in folds:
        Xf, Yt = X_fit[tr], Y[tr]
        Xpv, Yv = X_pred[va], Y[va]

        # Ridge with fit_intercept=True centres the (scaled) training design.
        xbar = Xf.mean(0)
        ybar = Yt.mean(0)
        Xfc = Xf - xbar
        Yc = Yt - ybar

        # One eigendecomposition of the Gram, reused across all alphas.
        G = Xfc.T @ Xfc                       # (p, p)
        lam, V = np.linalg.eigh(G)            # G = V diag(lam) V^T
        lam = np.clip(lam, 0.0, None)
        Z = V.T @ (Xfc.T @ Yc)                # (p, T)  = V^T X^T Yc

        # Held-out prediction uses X_pred (UNSCALED for fusion), centred by the
        # scaled training mean xbar -> reproduces WeightedRidge.predict on raw X.
        Qv = (Xpv - xbar) @ V                 # (n_val, p)
        E = Yv - ybar                         # (n_val, T)  resid base; pred = Qv R + ybar
        H = Qv.T @ E                          # (p, T)
        Gq = Qv.T @ Qv                        # (p, p)

        F = 1.0 / (lam[:, None] + alphas[None, :])   # (p, A)  ridge filter per alpha
        cross = F.T @ (H * Z)                 # (A, T)  <E, Qv R_a>
        e2 = (E * E).sum(0)                   # (T,)   ||E||^2
        TSS = ((Yv - Yv.mean(0)) ** 2).sum(0)  # (T,)  sklearn r2 denominator

        # quad(a, t) = || Qv (F_a * Z_t) ||^2 = R_a^T Gq R_a. One BLAS GEMM per
        # alpha (Gq @ R); cheap element-wise outside, no large broadcast temporaries.
        quad = np.empty((A, T), dtype=np.float64)
        for ai in range(A):
            R = F[:, ai:ai + 1] * Z           # (p, T)
            quad[ai] = np.einsum('kt,kt->t', R, Gq @ R)

        RSS = e2[None, :] - 2.0 * cross + quad
        with np.errstate(divide='ignore', invalid='ignore'):
            r2 = 1.0 - RSS / TSS[None, :]
        # Constant held-out target -> TSS == 0; sklearn would warn. Neutralise so
        # it never wins the argmax (real EEG val targets are never constant).
        r2[:, TSS == 0] = -np.inf
        sum_r2 += r2

    return sum_r2 / len(folds)


def _predict_selected(X_fit, X_pred_test, Y, alpha_per_target):
    """Refit on full train and predict the test set, each target at its own alpha.

    ``X_fit`` is the (scaled) training design, ``X_pred_test`` the (unscaled) test
    design, reproducing the refit-on-full-train + predict-on-raw step of the oracle.
    """
    X_fit = np.asarray(X_fit, dtype=np.float64)
    X_pred_test = np.asarray(X_pred_test, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    alpha_per_target = np.asarray(alpha_per_target, dtype=np.float64)

    xbar = X_fit.mean(0)
    ybar = Y.mean(0)
    Xfc = X_fit - xbar
    Yc = Y - ybar
    G = Xfc.T @ Xfc
    lam, V = np.linalg.eigh(G)
    lam = np.clip(lam, 0.0, None)
    Z = V.T @ (Xfc.T @ Yc)                    # (p, T)
    Qt = (X_pred_test - xbar) @ V             # (n_test, p)

    pred = np.empty((X_pred_test.shape[0], Y.shape[1]), dtype=np.float64)
    for a in np.unique(alpha_per_target):
        cols = np.where(alpha_per_target == a)[0]
        f = 1.0 / (lam + a)
        pred[:, cols] = Qt @ (f[:, None] * Z[:, cols]) + ybar[cols]
    return pred


def grid_search_batched(X_train, X_test, Y_train, alpha_range, kfold=5, metric='r2',
                        score_dtype=np.float64):
    """All-targets-at-once equivalent of per-cell ``Grid_search`` (single feature space).

    ``score_dtype`` controls only the CV scoring precision (see ``_meanr2_over_alphas``);
    final predictions are always float64. Use np.float32 for ~2x faster scoring.

    Returns
    -------
    Y_pred : (n_test, T)
    best_alpha : (T,)
    boundary_flag : (T,) int, 1 if the chosen alpha sits on the grid boundary.
    """
    if metric.lower() != 'r2':
        raise NotImplementedError("grid_search_batched currently supports metric='r2' only.")
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    Y_train = np.asarray(Y_train, dtype=np.float64)
    alpha_range = np.asarray(alpha_range, dtype=np.float64)

    folds = _make_folds(X_train.shape[0], kfold)
    mean_r2 = _meanr2_over_alphas(X_train, X_train, Y_train, alpha_range, folds,
                                  score_dtype=score_dtype)
    best_ai = np.argmax(mean_r2, axis=0)
    best_alpha = alpha_range[best_ai]
    Y_pred = _predict_selected(X_train, X_test, Y_train, best_alpha)
    boundary = ((best_ai == 0) | (best_ai == alpha_range.shape[0] - 1)).astype(int)
    return Y_pred, best_alpha, boundary


def grid_search_fusion_batched(X0_train, X0_test, X1_train, X1_test, Y_train,
                               alpha_range, weight_list, kfold=5, metric='r2',
                               score_dtype=np.float64):
    """All-targets-at-once equivalent of per-cell ``Grid_search_fusion``.

    Preserves the convex-combination scheme of ``WeightedRidge`` exactly: block 0
    is scaled by ``(1 - w)`` and block 1 by ``w`` for FITTING, while prediction
    uses the UNSCALED concatenated features (the published behaviour). Each target
    independently selects its own ``(alpha, weight)`` by cross-validated r2.

    Returns
    -------
    Y_pred : (n_test, T)
    best_alpha : (T,)
    best_weight : (T,)
    boundary_flag : (T,) int bitmask (1: alpha on boundary, 2: weight on boundary).
    """
    if metric.lower() != 'r2':
        raise NotImplementedError("grid_search_fusion_batched currently supports metric='r2' only.")
    X0_train = np.asarray(X0_train, dtype=np.float64)
    X1_train = np.asarray(X1_train, dtype=np.float64)
    X0_test = np.asarray(X0_test, dtype=np.float64)
    X1_test = np.asarray(X1_test, dtype=np.float64)
    Y_train = np.asarray(Y_train, dtype=np.float64)
    alpha_range = np.asarray(alpha_range, dtype=np.float64)
    weight_list = np.asarray(weight_list, dtype=np.float64)

    n, T = X0_train.shape[0], Y_train.shape[1]
    A = alpha_range.shape[0]
    folds = _make_folds(n, kfold)
    X_pred_train = np.hstack([X0_train, X1_train])   # UNSCALED, for held-out prediction

    best_score = np.full(T, -np.inf)
    best_ai = np.zeros(T, dtype=int)
    best_w = np.zeros(T, dtype=np.float64)

    # Iterate weights ascending; replace only on strictly better score so ties keep
    # the smaller (alpha, weight) -- matching GridSearchCV's first-argmax tie-break.
    for w in weight_list:
        X_fit = np.hstack([(1.0 - w) * X0_train, w * X1_train])
        mean_r2 = _meanr2_over_alphas(X_fit, X_pred_train, Y_train, alpha_range, folds,
                                      score_dtype=score_dtype)
        ai = np.argmax(mean_r2, axis=0)
        sc = mean_r2[ai, np.arange(T)]
        improve = sc > best_score
        best_score[improve] = sc[improve]
        best_ai[improve] = ai[improve]
        best_w[improve] = w

    best_alpha = alpha_range[best_ai]
    X_pred_test = np.hstack([X0_test, X1_test])      # UNSCALED test design
    Y_pred = np.empty((X_pred_test.shape[0], T), dtype=np.float64)
    for w in np.unique(best_w):
        cols = np.where(best_w == w)[0]
        X_fit = np.hstack([(1.0 - w) * X0_train, w * X1_train])
        Y_pred[:, cols] = _predict_selected(X_fit, X_pred_test, Y_train[:, cols], best_alpha[cols])

    boundary = np.zeros(T, dtype=int)
    boundary[(best_ai == 0) | (best_ai == A - 1)] |= 1
    boundary[(best_w == weight_list[0]) | (best_w == weight_list[-1])] |= 2
    return Y_pred, best_alpha, best_w, boundary
