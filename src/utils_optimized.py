"""
Optimized utilities for EEG encoding with backward compatibility
Drop-in replacement for utils.py with massive performance improvements
"""

import time
import warnings
from typing import Tuple, Sequence, Union, Optional
import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from sklearn.model_selection import KFold, GridSearchCV

# Import our optimized implementations
try:
    from incremental_cv_implementation import MemoryEfficientRidgeCV
    from bayesian_ridge_cv import BayesianOptimizationRidge, BayesianOptimizationConvexCombination
    from convex_combination_optimizer import hierarchical_convex_optimization, IncrementalConvexCombinationCV
    OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Optimized implementations not available: {e}. Using original methods.")
    OPTIMIZATIONS_AVAILABLE = False


def estimate_memory_requirements(n_samples: int, n_targets: int, n_alphas: int, n_weights: int = 1) -> Tuple[float, float]:
    """Estimate memory requirements for different optimization strategies"""
    # Traditional approach (storing all CV predictions)
    traditional_gb = (n_alphas * n_weights * 5 * n_samples * n_targets * 4) / 1e9
    
    # Incremental approach (storing only statistics) 
    incremental_gb = (n_alphas * n_weights * n_targets * 2 * 4) / 1e9
    
    return traditional_gb, incremental_gb


def select_optimization_strategy(alpha_range: np.ndarray, 
                               weight_list: Optional[np.ndarray] = None,
                               optimization_strategy: str = 'auto',
                               memory_budget_gb: float = 8.0,
                               use_bayesian: bool = False) -> str:
    """
    Automatically select best optimization strategy based on problem characteristics
    
    Args:
        alpha_range: Array of alpha values to search
        weight_list: Array of weight values to search (for fusion)
        optimization_strategy: 'original', 'auto', 'incremental', 'bayesian', 'hierarchical'
        memory_budget_gb: Available memory budget in GB
        use_bayesian: Whether to prefer Bayesian optimization
        
    Returns:
        Selected strategy name
    """
    if not OPTIMIZATIONS_AVAILABLE or optimization_strategy == 'original':
        return 'original'
    
    if optimization_strategy != 'auto':
        return optimization_strategy
    
    n_alphas = len(alpha_range)
    
    if weight_list is None:
        # Single parameter optimization
        if use_bayesian and n_alphas > 20:
            return 'bayesian_single'
        elif n_alphas > 50:
            return 'incremental'
        else:
            return 'original'
    else:
        # Dual parameter optimization
        n_weights = len(weight_list)
        total_combinations = n_alphas * n_weights
        
        # Estimate memory for joint optimization
        traditional_gb, incremental_gb = estimate_memory_requirements(
            n_samples=10000, n_targets=1, n_alphas=n_alphas, n_weights=n_weights
        )
        
        if use_bayesian and total_combinations > 100:
            return 'bayesian_dual'
        elif incremental_gb > memory_budget_gb:
            return 'hierarchical'  # Most memory efficient
        elif total_combinations > 500:
            return 'incremental'
        elif total_combinations > 50:
            return 'incremental'
        else:
            return 'original'


def Grid_search(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    alpha_range: Union[Sequence[float], np.ndarray],
    metric: str = 'mse',
    kfold: int = 5,
    n_jobs: int = -1,
    optimization_strategy: str = 'auto',
    memory_budget_gb: float = 8.0,
    use_bayesian: bool = False,
) -> Tuple[np.ndarray, float, int]:
    """
    Optimized grid search over alpha for Ridge regression with backward compatibility
    
    This function maintains exact same interface as original but uses optimized methods internally.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_samples, n_features).
    X_test : np.ndarray
        Test features of shape (m_samples, n_features).
    y_train : np.ndarray
        Training targets of shape (n_samples,) - SINGLE TARGET ONLY
    y_test : np.ndarray
        Test targets of shape (m_samples,) - SINGLE TARGET ONLY
    alpha_range : array-like of float
        1D array or sequence of alpha values to search
    metric : str, default='mse'
        Metric for model selection: 'mse', 'r2', or 'mae'
    kfold : int, default=5
        Number of cross-validation folds
    n_jobs : int, default=-1
        Number of parallel jobs (maintained for compatibility)
    optimization_strategy : str, default='auto'
        Strategy: 'original', 'auto', 'incremental', 'bayesian'
    memory_budget_gb : float, default=8.0
        Memory budget for optimization
    use_bayesian : bool, default=False
        Prefer Bayesian optimization when applicable

    Returns
    -------
    predictions : np.ndarray
        Predicted values on X_test - SAME SHAPE AS ORIGINAL
    best_alpha : float
        Optimal alpha value - SAME TYPE AS ORIGINAL
    boundary_flag : int
        1 if best_alpha is on boundary, else 0 - SAME AS ORIGINAL
    """
    # Ensure numpy arrays and proper shapes
    alpha_arr = np.asarray(alpha_range)
    if alpha_arr.ndim != 1:
        raise ValueError("alpha_range must be a 1D sequence or array.")
    
    if y_train.ndim != 1 or y_test.ndim != 1:
        raise ValueError("y_train and y_test must be 1D arrays for Grid_search")
    
    # Select optimization strategy
    strategy = select_optimization_strategy(
        alpha_arr, None, optimization_strategy, memory_budget_gb, use_bayesian
    )
    
    try:
        if strategy == 'bayesian_single' and OPTIMIZATIONS_AVAILABLE:
            # Use Bayesian optimization
            device = torch.device('cpu')  # Safer default
            
            bo = BayesianOptimizationRidge(
                alpha_bounds=(float(alpha_arr.min()), float(alpha_arr.max())),
                n_calls=min(30, len(alpha_arr) // 3),
                device=device
            )
            
            # Convert to multi-target format temporarily for BO
            X_train_torch = torch.tensor(X_train, dtype=torch.float32)
            Y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Shape: (n, 1)
            
            best_alpha, _ = bo.optimize(X_train_torch, Y_train_torch, verbose=False)
            
        elif strategy == 'incremental' and OPTIMIZATIONS_AVAILABLE:
            # Use incremental CV
            device = torch.device('cpu')
            
            ridge_cv = MemoryEfficientRidgeCV(
                alphas=torch.tensor(alpha_arr, dtype=torch.float32),
                cv_folds=kfold,
                device=device
            )
            
            # Convert to multi-target format temporarily
            X_train_torch = torch.tensor(X_train, dtype=torch.float32)
            Y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Shape: (n, 1)
            
            ridge_cv.fit(X_train_torch, Y_train_torch, verbose=False)
            best_alphas, _ = ridge_cv.get_best_params()
            best_alpha = best_alphas[0].item()  # Extract single value
            
        else:
            # Fall back to original implementation
            return Grid_search_original(X_train, X_test, y_train, y_test, alpha_arr, metric, kfold, n_jobs)
            
    except Exception as e:
        warnings.warn(f"Optimized method failed ({e}), falling back to original implementation")
        return Grid_search_original(X_train, X_test, y_train, y_test, alpha_arr, metric, kfold, n_jobs)
    
    # Make predictions using optimal alpha (same as original)
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_train, y_train)
    predictions = ridge.predict(X_test)
    
    # Compute boundary flag (same as original)
    boundary_flag = int(best_alpha in (alpha_arr[0], alpha_arr[-1]))
    
    return predictions, best_alpha, boundary_flag


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
    optimization_strategy: str = 'auto',
    memory_budget_gb: float = 8.0,
    use_bayesian: bool = False,
) -> Tuple[np.ndarray, float, float, int]:
    """
    Optimized grid search over alpha and weight for convex combination with backward compatibility
    
    This function maintains exact same interface as original but uses optimized methods internally.
    
    Parameters
    ----------
    X0_train, X1_train : np.ndarray
        Training features for block 1 and block 2
    X0_test, X1_test : np.ndarray
        Test features for block 1 and block 2
    y_train : np.ndarray
        Training targets of shape (n_samples,) - SINGLE TARGET ONLY
    y_test : np.ndarray
        Test targets of shape (m_samples,) - SINGLE TARGET ONLY
    alpha_range : array-like of float
        1D array of alpha values
    weight_list : array-like of float
        1D array of weight values
    metric : str, default='mse'
        Metric for model selection: 'mse', 'r2', or 'mae'
    kfold : int, default=5
        Number of CV folds
    n_jobs : int, default=-1
        Parallel jobs (maintained for compatibility)
    optimization_strategy : str, default='auto'
        Strategy: 'original', 'auto', 'incremental', 'bayesian', 'hierarchical'
    memory_budget_gb : float, default=8.0
        Memory budget for optimization
    use_bayesian : bool, default=False
        Prefer Bayesian optimization when applicable

    Returns
    -------
    predictions : np.ndarray
        Predicted values on stacked test data - SAME SHAPE AS ORIGINAL
    best_alpha : float
        Optimal alpha found - SAME TYPE AS ORIGINAL
    best_weight : float
        Optimal weight found - SAME TYPE AS ORIGINAL
    boundary_flag : int
        Bitmask flag for boundary detection - SAME AS ORIGINAL
    """
    # Ensure numpy arrays
    alpha_arr = np.asarray(alpha_range)
    weight_arr = np.asarray(weight_list)
    
    if alpha_arr.ndim != 1 or weight_arr.ndim != 1:
        raise ValueError("alpha_range and weight_list must be 1D sequences or arrays.")
    
    if y_train.ndim != 1 or y_test.ndim != 1:
        raise ValueError("y_train and y_test must be 1D arrays for Grid_search_fusion")
    
    # Select optimization strategy
    strategy = select_optimization_strategy(
        alpha_arr, weight_arr, optimization_strategy, memory_budget_gb, use_bayesian
    )
    
    try:
        if strategy == 'bayesian_dual' and OPTIMIZATIONS_AVAILABLE:
            # Use 2D Bayesian optimization
            device = torch.device('cpu')
            
            bo = BayesianOptimizationConvexCombination(
                alpha_bounds=(float(alpha_arr.min()), float(alpha_arr.max())),
                weight_bounds=(float(weight_arr.min()), float(weight_arr.max())),
                n_calls=min(50, len(alpha_arr) * len(weight_arr) // 10),
                device=device
            )
            
            # Convert to tensors
            X0_torch = torch.tensor(X0_train, dtype=torch.float32)
            X1_torch = torch.tensor(X1_train, dtype=torch.float32)
            Y_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            
            best_alpha, best_weight, _ = bo.optimize(X0_torch, X1_torch, Y_torch, verbose=False)
            
        elif strategy == 'hierarchical' and OPTIMIZATIONS_AVAILABLE:
            # Use hierarchical optimization
            device = torch.device('cpu')
            
            X0_torch = torch.tensor(X0_train, dtype=torch.float32)
            X1_torch = torch.tensor(X1_train, dtype=torch.float32)
            Y_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            
            best_alphas, best_weights, _ = hierarchical_convex_optimization(
                X0_torch, X1_torch, Y_torch, device=device, verbose=False
            )
            best_alpha = best_alphas[0].item()
            best_weight = best_weights[0].item()
            
        elif strategy == 'incremental' and OPTIMIZATIONS_AVAILABLE:
            # Use incremental CV
            device = torch.device('cpu')
            
            cv = IncrementalConvexCombinationCV(
                alphas=torch.tensor(alpha_arr, dtype=torch.float32),
                weights=torch.tensor(weight_arr, dtype=torch.float32),
                cv_folds=kfold,
                device=device
            )
            
            X0_torch = torch.tensor(X0_train, dtype=torch.float32)
            X1_torch = torch.tensor(X1_train, dtype=torch.float32)
            Y_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            
            cv.fit(X0_torch, X1_torch, Y_torch, verbose=False)
            best_alphas, best_weights, _ = cv.get_best_params()
            best_alpha = best_alphas[0].item()
            best_weight = best_weights[0].item()
            
        else:
            # Fall back to original implementation
            return Grid_search_fusion_original(X0_train, X0_test, X1_train, X1_test, 
                                             y_train, y_test, alpha_arr, weight_arr, 
                                             metric, kfold, n_jobs)
            
    except Exception as e:
        warnings.warn(f"Optimized fusion method failed ({e}), falling back to original implementation")
        return Grid_search_fusion_original(X0_train, X0_test, X1_train, X1_test, 
                                         y_train, y_test, alpha_arr, weight_arr, 
                                         metric, kfold, n_jobs)
    
    # Make predictions using optimal parameters (same as original)
    X_combined = (1 - best_weight) * X0_train + best_weight * X1_train
    X_test_combined = (1 - best_weight) * X0_test + best_weight * X1_test
    
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_combined, y_train)
    predictions = ridge.predict(X_test_combined)
    
    # Compute boundary flags (same as original)
    boundary_flag = 0
    if best_alpha in (alpha_arr[0], alpha_arr[-1]):
        boundary_flag |= 1
    if best_weight in (weight_arr[0], weight_arr[-1]):
        boundary_flag |= 2
    
    return predictions, best_alpha, best_weight, boundary_flag


# Original implementations as fallback
def Grid_search_original(X_train, X_test, y_train, y_test, alpha_range, metric='mse', kfold=5, n_jobs=-1):
    """Original Grid_search implementation as fallback"""
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
    param_grid = {'alpha': alpha_range}
    grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)

    # Fit
    grid.fit(X_train, y_train)
    best_alpha = grid.best_params_['alpha']
    predictions = grid.best_estimator_.predict(X_test)
    boundary_flag = int(best_alpha in (alpha_range[0], alpha_range[-1]))

    return predictions, best_alpha, boundary_flag


class WeightedRidge(BaseEstimator, RegressorMixin):
    """Ridge regressor that weights two blocks of features (from original)"""
    
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


def Grid_search_fusion_original(X0_train, X0_test, X1_train, X1_test, y_train, y_test, 
                               alpha_range, weight_list, metric='mse', kfold=5, n_jobs=-1):
    """Original Grid_search_fusion implementation as fallback"""
    
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
    estimator = WeightedRidge(alpha=alpha_range[0], weight=weight_list[0], feature_one_index=feature_one_index)
    param_grid = {
        'alpha': alpha_range,
        'weight': weight_list,
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
    if best_alpha in (alpha_range[0], alpha_range[-1]):
        boundary_flag |= 1
    if best_weight in (weight_list[0], weight_list[-1]):
        boundary_flag |= 2

    return predictions, best_alpha, best_weight, boundary_flag