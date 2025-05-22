import time
from typing import Tuple, Sequence, Union
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
    """
    Perform grid search over alpha for Ridge regression.

    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_samples, n_features).
    X_test : np.ndarray
        Test features of shape (m_samples, n_features).
    y_train : np.ndarray
        Training targets of shape (n_samples,) or (n_samples, n_targets).
    y_test : np.ndarray
        Test targets of shape (m_samples,) or (m_samples, n_targets).
    alpha_range : array-like of float
        1D array or sequence of alpha values to search (e.g., np.ndarray, list).
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
    # Ensure numpy array
    alpha_arr = np.asarray(alpha_range)
    if alpha_arr.ndim != 1:
        raise ValueError("alpha_range must be a 1D sequence or array.")

    # Map metric to scorers
    scorers = {
        'mse': make_scorer(mean_squared_error, greater_is_better=False),
        'r2': make_scorer(r2_score),
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
    }
    scoring = scorers.get(metric.lower())
    if scoring is None:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from {list(scorers)}.")

    # Cross-validation
    cv = KFold(n_splits=kfold, shuffle=True, random_state=42)
    model = Ridge()
    param_grid = {'alpha': alpha_arr}
    grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)

    # Fit
    grid.fit(X_train, y_train)
    best_alpha = grid.best_params_['alpha']
    predictions = grid.best_estimator_.predict(X_test)
    boundary_flag = int(best_alpha in (alpha_arr[0], alpha_arr[-1]))

    return predictions, best_alpha, boundary_flag


class WeightedRidge(BaseEstimator, RegressorMixin):
    """
    Ridge regressor that weights two blocks of features.

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
    """
    Grid search over alpha and weight for two-feature-block Ridge fusion.

    Parameters
    ----------
    X0_train, X1_train : np.ndarray
        Training features for block 1 and block 2.
    X0_test, X1_test : np.ndarray
        Test features for block 1 and block 2.
    y_train : np.ndarray
        Training targets.
    y_test : np.ndarray
        Test targets.
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
    # Convert ranges to arrays
    alpha_arr = np.asarray(alpha_range)
    weight_arr = np.asarray(weight_list)
    if alpha_arr.ndim != 1 or weight_arr.ndim != 1:
        raise ValueError("alpha_range and weight_list must be 1D sequences or arrays.")

    # Stack features and define split
    X_train = np.hstack([X0_train, X1_train])
    X_test = np.hstack([X0_test, X1_test])
    feature_one_index = X0_train.shape[1]

    # Scorers
    scorers = {
        'mse': make_scorer(mean_squared_error, greater_is_better=False),
        'r2': make_scorer(r2_score),
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
    }
    scoring = scorers.get(metric.lower())
    if scoring is None:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from {list(scorers)}.")

    cv = KFold(n_splits=kfold, shuffle=True, random_state=42)
    estimator = WeightedRidge(alpha=alpha_arr[0], weight=weight_arr[0], feature_one_index=feature_one_index)
    param_grid = {
        'alpha': alpha_arr,
        'weight': weight_arr,
        'feature_one_index': [feature_one_index],
    }
    grid = GridSearchCV(estimator, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)

    # Fit
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
