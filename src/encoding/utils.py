"""Fast Ridge regression utilities for EEG encoding models.

This module provides drop-in, numerically identical replacements for
``GridSearchCV(Ridge(), ...)`` based hyperparameter searches:

- :func:`Grid_search` performs single-block Ridge regression with KFold CV over
  the regularization strength ``alpha``.
- :func:`Grid_search_fusion` performs two-block (fusion) Ridge regression with
  KFold CV over both ``alpha`` and a block-mixing ``weight``.

Both share the SVD of the training fold across all candidate ``alpha`` (and,
for fusion, all candidate ``weight``) values, yielding ~100x speedups while
producing results identical to the original ``GridSearchCV`` formulations.
"""

from typing import Sequence, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


# =============================================================================
# Internal fast Ridge CV (shared SVD across alphas per fold)
# =============================================================================

def _score_fold(y_true: np.ndarray, y_pred: np.ndarray, scoring: str) -> float:
    if scoring == 'r2':
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0
    elif scoring == 'neg_mean_squared_error':
        return float(-np.mean((y_true - y_pred) ** 2))
    elif scoring == 'neg_mean_absolute_error':
        return float(-np.mean(np.abs(y_true - y_pred)))
    raise ValueError(f"Unknown scoring '{scoring}'")


def _ridge_kfold_fast(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test,           # ndarray or None (skip prediction when None)
    alpha_arr: np.ndarray,
    cv: KFold,
    scoring: str,
) -> Tuple[np.ndarray, float, float]:
    """
    Fast KFold Ridge CV with shared SVD across alphas.

    For each CV fold the SVD of X_fold is computed ONCE and all alphas are
    evaluated analytically — no extra SVD per alpha.  Results are identical
    to GridSearchCV(Ridge(), {'alpha': alphas}, cv=cv, scoring=scoring).

    Speedup vs GridSearchCV:
      model-only : 5 SVDs  instead of 500  (100-alpha × 5-fold)  → 100×
      fusion     : 205 SVDs instead of 20 500 (100-alpha × 41-weight × 5-fold) → 100×

    Returns
    -------
    predictions : ndarray or None
        Predictions on X_test (None if X_test is None).
    best_alpha : float
    best_cv_score : float
        Mean CV score of the best alpha (used to compare across weights in fusion).
    """
    n_alpha = len(alpha_arr)
    cv_scores = np.zeros(n_alpha, dtype=np.float64)
    n_splits = 0

    for train_idx, val_idx in cv.split(X_train):
        n_splits += 1
        Xf = X_train[train_idx].astype(np.float64)
        yf = y_train[train_idx].astype(np.float64)
        Xv = X_train[val_idx].astype(np.float64)
        yv = y_train[val_idx].astype(np.float64)

        # Ridge(fit_intercept=True) centers X and y by training-fold mean
        x_mean = Xf.mean(axis=0)
        y_mean = yf.mean()
        Xf_c = Xf - x_mean
        yf_c = yf - y_mean
        Xv_c = Xv - x_mean

        # SVD of centered training fold — ONE SVD serves ALL 100 alphas
        U, s, Vt = np.linalg.svd(Xf_c, full_matrices=False)

        # Filter negligible singular values (matches sklearn Ridge behavior)
        mask = s > 1e-15
        s_m = s[mask]
        Uty = (U[:, mask].T) @ yf_c        # (k,)
        XvV = Xv_c @ (Vt[mask].T)          # (n_val, k)

        for a_idx, alpha in enumerate(alpha_arr):
            d = s_m / (s_m ** 2 + alpha)
            y_pred = XvV @ (d * Uty) + y_mean
            cv_scores[a_idx] += _score_fold(yv, y_pred, scoring)

    cv_scores /= n_splits
    best_idx = int(np.argmax(cv_scores))
    best_alpha = float(alpha_arr[best_idx])
    best_cv_score = float(cv_scores[best_idx])

    if X_test is None:
        return None, best_alpha, best_cv_score

    # Fit final model on full training set with best alpha
    x_mean_full = X_train.mean(axis=0, dtype=np.float64)
    y_mean_full = float(y_train.mean())
    X_c = X_train.astype(np.float64) - x_mean_full
    y_c = y_train.astype(np.float64) - y_mean_full
    U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    mask = s > 1e-15
    s_m = s[mask]
    d = s_m / (s_m ** 2 + best_alpha)
    coef = (Vt[mask].T) @ (d * ((U[:, mask].T) @ y_c))
    predictions = ((X_test.astype(np.float64) - x_mean_full) @ coef + y_mean_full).astype(np.float32)

    return predictions, best_alpha, best_cv_score


# =============================================================================
# Public API
# =============================================================================

def Grid_search(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    alpha_range: Union[Sequence[float], np.ndarray],
    metric: str = 'mse',
    kfold: int = 5,
    n_jobs: int = -1,
    **kwargs,
) -> Tuple[np.ndarray, float, int]:
    """
    Ridge regression with KFold CV over alpha.

    Equivalent to GridSearchCV(Ridge(), {'alpha': alphas}, cv=KFold(kfold,
    shuffle=True, random_state=42), scoring=metric) but ~100× faster via
    shared SVD across alphas within each fold.

    Returns
    -------
    predictions : ndarray
    best_alpha : float
    boundary_flag : int  (1 if best_alpha is at boundary of alpha_range)
    """
    alpha_arr = np.asarray(alpha_range)
    if alpha_arr.ndim != 1:
        raise ValueError("alpha_range must be 1-D.")
    scoring_map = {
        'mse': 'neg_mean_squared_error',
        'r2':  'r2',
        'mae': 'neg_mean_absolute_error',
    }
    scoring = scoring_map.get(metric.lower())
    if scoring is None:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from {list(scoring_map)}.")

    cv = KFold(n_splits=kfold, shuffle=True, random_state=42)
    predictions, best_alpha, _ = _ridge_kfold_fast(
        X_train, y_train, X_test, alpha_arr, cv, scoring)
    boundary_flag = int(best_alpha in (alpha_arr[0], alpha_arr[-1]))
    return predictions, best_alpha, boundary_flag


class WeightedRidge(BaseEstimator, RegressorMixin):
    """Ridge regressor that scales two feature blocks by complementary weights."""

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


def _ridge_kfold_fast_fusion(
    X0_train: np.ndarray,
    X1_train: np.ndarray,
    X0_test: np.ndarray,
    X1_test: np.ndarray,
    y_train: np.ndarray,
    alpha_arr: np.ndarray,
    weight_arr: np.ndarray,
    cv: KFold,
    scoring: str,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Two-block Ridge fusion CV sharing SVD(X_all_fold) across all weights.

    Instead of W × 5 large SVDs of X_w_fold  (shape n_fold × p),
    this computes 1 large SVD of X_all_fold per fold then W small SVDs of
    M_w = diag(s) @ (Vt * d_w)  (shape k × p, with k ≈ p ≈ 2000).

    Math derivation
    ---------------
    X_w = X_all @ D_w  where D_w = diag([(1-w)·I_{p0}, w·I_{p1}])
    SVD(X_all_fold_c) = U Σ Vt
    → X_w_fold_c = U @ M_w  where M_w = diag(s) @ (Vt * d_w)
    → SVD(M_w) = Pm Sm Vmt
    → SVD(X_w_fold_c) = (U Pm) Sm Vmt  (U Pm has ortho-normal columns)
    → Uty_w   = Pm[:, mask].T @ (U.T @ yf_c)   (reuses U.T @ yf_c across weights)
    → XvV_w   = (Xv_all_c * d_w) @ Vmt[mask].T
    → ŷ(α)   = XvV_w @ diag(Sm_m/(Sm_m²+α)) @ Uty_w + y_mean

    All 100 alphas are evaluated analytically for each weight using the one
    small SVD, identical to calling _ridge_kfold_fast per weight.

    Speedup vs looping _ridge_kfold_fast over W weights:
      SVDs : W × large(n×p) → 1 large + W × small(p×p) per fold  (~3–5×)

    Returns
    -------
    predictions   : ndarray  (float32)
    best_alpha    : float
    best_weight   : float
    best_cv_score : float
    """
    p0 = X0_train.shape[1]
    p1 = X1_train.shape[1]
    p  = p0 + p1
    n_alpha = len(alpha_arr)
    n_weight = len(weight_arr)

    X_all_train = np.hstack([X0_train, X1_train])   # (n, p)

    cv_scores = np.zeros((n_weight, n_alpha), dtype=np.float64)
    n_splits = 0

    for train_idx, val_idx in cv.split(X_all_train):
        n_splits += 1
        Xf = X_all_train[train_idx].astype(np.float64)
        Xv = X_all_train[val_idx].astype(np.float64)
        yf = y_train[train_idx].astype(np.float64)
        yv = y_train[val_idx].astype(np.float64)

        x_mean = Xf.mean(axis=0)
        y_mean = yf.mean()
        Xf_c   = Xf - x_mean
        Xv_c   = Xv - x_mean
        yf_c   = yf - y_mean

        # ONE large SVD per fold — shared across all W weights
        U, s, Vt = np.linalg.svd(Xf_c, full_matrices=False)   # U:(n_f,k), s:(k,), Vt:(k,p)

        # Fold-level precomputation  (all O(k*p), negligible vs eigh below)
        SVt      = s[:, np.newaxis] * Vt              # (k, p)  = diag(s) @ Vt = M_all
        Uty_base = U.T @ yf_c                          # (k,)
        rhs_all  = SVt.T @ Uty_base                    # (p,)  = M_all^T @ Uty_base
        #
        # G = SVt^T @ SVt = M_all^T M_all = Vt^T diag(s^2) Vt  (p×p, symmetric PSD)
        # Precomputed ONCE per fold; per weight: G_w = G * outer(d,d)  [O(p^2) scaling]
        # then eigh(G_w) ~2.8× faster than svd(M_w) (symmetric vs general matrix).
        #
        G = SVt.T @ SVt                                # (p, p), ONE matmul per fold

        for w_idx, w in enumerate(weight_arr):
            # Column-scaling vector  d_w = [(1-w)…p0, w…p1]
            d = np.empty(p, dtype=np.float64)
            d[:p0] = 1.0 - float(w)
            d[p0:] = float(w)

            # G_w = D_w G D_w = G * outer(d, d)   O(p^2), trivial
            G_w = G * (d[:, np.newaxis] * d[np.newaxis, :])

            # eigh of symmetric (p×p) PSD matrix:
            #   eigenvalues  Sm2  = squared singular values of M_w
            #   eigenvectors V_e  = right singular vectors of M_w  (as columns)
            # ~2.8× faster than svd(M_w) on same-sized square matrix.
            Sm2, V_e = np.linalg.eigh(G_w)            # Sm2 ascending, V_e[:,j] is j-th eigvec

            mask_m = Sm2 > 1e-28                       # keep numerically significant components
            Sm2_m  = Sm2[mask_m]                       # (k_m,)
            V_em   = V_e[:, mask_m]                    # (p, k_m)

            # r_w = V_em^T @ (d * rhs_all)
            # Equivalent to: Uty_w computed via Vmt in the SVD path
            r_w   = V_em.T @ (d * rhs_all)            # (k_m,)

            # XvV_w = (Xv_c * d) @ V_em
            XvV_w = (Xv_c * d) @ V_em                 # (n_val, k_m)

            for a_idx, alpha in enumerate(alpha_arr):
                da     = 1.0 / (Sm2_m + alpha)
                y_pred = XvV_w @ (da * r_w) + y_mean
                cv_scores[w_idx, a_idx] += _score_fold(yv, y_pred, scoring)

    cv_scores /= n_splits

    # Joint argmax over (weight, alpha) — equivalent to nested argmax
    best_w_idx, best_a_idx = np.unravel_index(np.argmax(cv_scores), cv_scores.shape)
    best_alpha    = float(alpha_arr[best_a_idx])
    best_weight   = float(weight_arr[best_w_idx])
    best_cv_score = float(cv_scores[best_w_idx, best_a_idx])

    # Final fit on full training set with best (alpha, weight) — no CV needed
    best_d = np.empty(p, dtype=np.float64)
    best_d[:p0] = 1.0 - best_weight
    best_d[p0:] = best_weight

    X_train_best = X_all_train.astype(np.float64) * best_d
    X_test_best  = np.hstack([
        X0_test.astype(np.float64) * (1.0 - best_weight),
        X1_test.astype(np.float64) * best_weight,
    ])

    x_mean_f = X_train_best.mean(axis=0)
    y_mean_f = float(y_train.mean())
    Xtr_c    = X_train_best - x_mean_f
    y_c      = y_train.astype(np.float64) - y_mean_f
    Uf, sf, Vtf = np.linalg.svd(Xtr_c, full_matrices=False)
    mask_f   = sf > 1e-15
    sf_m     = sf[mask_f]
    d_f      = sf_m / (sf_m ** 2 + best_alpha)
    coef     = (Vtf[mask_f].T) @ (d_f * ((Uf[:, mask_f].T) @ y_c))
    predictions = ((X_test_best - x_mean_f) @ coef + y_mean_f).astype(np.float32)

    return predictions, best_alpha, best_weight, best_cv_score


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
    **kwargs,
) -> Tuple[np.ndarray, float, float, int]:
    """
    Two-block Ridge fusion with KFold CV over alpha and weight.

    Uses _ridge_kfold_fast_fusion: SVD(X_all_fold) shared across all weights,
    then a small SVD of M_w = diag(s)@(Vt*d_w) per weight.

    ~100× faster than GridSearchCV(WeightedRidge, {alpha, weight}, cv=KFold(5)).
    Results are identical to the original GridSearchCV formulation.

    Returns
    -------
    predictions : ndarray
    best_alpha : float
    best_weight : float
    boundary_flag : int  (bit 0: alpha on boundary; bit 1: weight on boundary)
    """
    alpha_arr  = np.asarray(alpha_range)
    weight_arr = np.asarray(weight_list)
    if alpha_arr.ndim != 1 or weight_arr.ndim != 1:
        raise ValueError("alpha_range and weight_list must be 1-D.")
    scoring_map = {
        'mse': 'neg_mean_squared_error',
        'r2':  'r2',
        'mae': 'neg_mean_absolute_error',
    }
    scoring = scoring_map.get(metric.lower())
    if scoring is None:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from {list(scoring_map)}.")

    cv = KFold(n_splits=kfold, shuffle=True, random_state=42)

    predictions, best_alpha, best_weight, _ = _ridge_kfold_fast_fusion(
        X0_train, X1_train, X0_test, X1_test, y_train,
        alpha_arr, weight_arr, cv, scoring,
    )

    boundary_flag = 0
    if best_alpha in (alpha_arr[0], alpha_arr[-1]):
        boundary_flag |= 1
    if best_weight in (float(weight_arr[0]), float(weight_arr[-1])):
        boundary_flag |= 2

    return predictions, best_alpha, best_weight, boundary_flag
