"""
Optimized Convex Combination Parameter Search for EEG Encoding
Jointly optimizes alpha (regularization) and weight (feature combination) parameters
"""

import torch
import numpy as np
from typing import Tuple, List, Optional, Union, Dict
from sklearn.model_selection import KFold
import time
import warnings
warnings.filterwarnings('ignore')

from incremental_cv_implementation import IncrementalCVStats
from bayesian_ridge_cv import BayesianOptimizationConvexCombination


class WeightedMultiRidge:
    """
    Multi-output ridge regression with weighted feature combination
    Extends NSD MultiRidge to handle (1-w)*X1 + w*X2 feature combinations
    """
    
    def __init__(self, alphas: torch.Tensor, weight: float, 
                 scale_X: bool = True, scale_thresh: float = 1e-8, device=None):
        self.alphas = alphas
        self.weight = weight
        self.scale_X = scale_X
        self.scale_thresh = scale_thresh
        self.device = device or torch.device('cpu')
        
        # Fitted parameters
        self.Xm = None
        self.Xs = None 
        self.Ym = None
        self.e = None
        self.Q = None
        self.Y = None
        self.X_t = None
    
    def fit(self, X1: torch.Tensor, X2: torch.Tensor, Y: torch.Tensor):
        """
        Fit weighted ridge regression
        
        Args:
            X1: First feature set (n_samples, n_features1)
            X2: Second feature set (n_samples, n_features2)
            Y: Targets (n_samples, n_targets)
        """
        # Create weighted combination
        X_combined = (1 - self.weight) * X1 + self.weight * X2
        
        # Move to device
        if self.device is not None:
            X_combined = X_combined.to(self.device)
            Y = Y.to(self.device)
        
        # Center features
        self.Xm = X_combined.mean(dim=0, keepdim=True)
        X = X_combined - self.Xm
        
        # Scale features if requested
        if self.scale_X:
            self.Xs = X.std(dim=0, keepdim=True)
            self.Xs[self.Xs < self.scale_thresh] = 1
            X = X / self.Xs
        
        # SVD decomposition (key to NSD efficiency)
        self.X_t = X.t()
        _, S, V = self.X_t.svd()
        self.e = S.pow_(2)
        self.Q = self.X_t @ V
        
        # Center targets
        self.Y = Y
        self.Ym = Y.mean(dim=0)
        
        return self
    
    def get_prediction_scores(self, X1_test: torch.Tensor, X2_test: torch.Tensor, 
                            Y_test: torch.Tensor, scoring_func) -> torch.Tensor:
        """
        Get prediction scores for all alphas simultaneously
        
        Returns:
            scores: (n_targets, n_alphas) tensor of validation scores
        """
        # Create weighted test features
        X_test_combined = (1 - self.weight) * X1_test + self.weight * X2_test
        
        if self.device is not None:
            X_test_combined = X_test_combined.to(self.device)
            Y_test = Y_test.to(self.device)
        
        # Apply same preprocessing as training
        X_test = X_test_combined - self.Xm
        if self.scale_X:
            X_test = X_test / self.Xs
        
        M_test = X_test @ self.Q
        
        scores = torch.zeros(Y_test.shape[1], len(self.alphas), 
                           dtype=X_test.dtype, device=X_test.device)
        
        for j, Y_test_j in enumerate(Y_test.t()):
            Ym_j, r_j, N_test_j = self._compute_pred_interms(j, X_test)
            
            for k, alpha in enumerate(self.alphas):
                # Ridge prediction using Woodbury identity
                Y_pred_j = (1 / alpha) * (N_test_j - M_test @ (r_j / (self.e + alpha))) + Ym_j
                scores[j, k] = scoring_func(Y_test_j, Y_pred_j).item()
        
        return scores
    
    def _compute_pred_interms(self, y_idx: int, X_test: torch.Tensor):
        """Compute intermediate terms for prediction (from NSD implementation)"""
        Y_j, Ym_j = self.Y[:, y_idx], self.Ym[y_idx]
        p_j = self.X_t @ (Y_j - Ym_j)
        r_j = self.Q.t() @ p_j
        N_test_j = X_test @ p_j
        return Ym_j, r_j, N_test_j
    
    def predict_single(self, X1_test: torch.Tensor, X2_test: torch.Tensor, 
                      alpha_indices: List[int]) -> torch.Tensor:
        """Make predictions with specified alpha per target"""
        X_test_combined = (1 - self.weight) * X1_test + self.weight * X2_test
        
        if self.device is not None:
            X_test_combined = X_test_combined.to(self.device)
        
        X_test = X_test_combined - self.Xm
        if self.scale_X:
            X_test = X_test / self.Xs
        
        M_test = X_test @ self.Q
        
        Y_pred_list = []
        for j, alpha_idx in enumerate(alpha_indices):
            Ym_j, r_j, N_test_j = self._compute_pred_interms(j, X_test)
            alpha = self.alphas[alpha_idx]
            Y_pred_j = (1 / alpha) * (N_test_j - M_test @ (r_j / (self.e + alpha))) + Ym_j
            Y_pred_list.append(Y_pred_j)
        
        return torch.stack(Y_pred_list, dim=1)


class IncrementalConvexCombinationCV:
    """
    Memory-efficient cross-validation for convex combination optimization
    Uses incremental statistics to avoid storing all predictions
    """
    
    def __init__(self,
                 alphas: torch.Tensor,
                 weights: torch.Tensor,
                 cv_folds: int = 5,
                 scoring_func=None,
                 device=None,
                 early_stop_confidence: float = 0.95,
                 min_folds_before_stop: int = 3):
        
        self.alphas = alphas
        self.weights = weights
        self.cv_folds = cv_folds
        self.device = device or torch.device('cpu')
        self.early_stop_confidence = early_stop_confidence
        self.min_folds_before_stop = min_folds_before_stop
        
        # Default scoring function (negative MSE for maximization)
        self.scoring_func = scoring_func or (lambda y_true, y_pred: -torch.nn.functional.mse_loss(y_pred, y_true))
        
        # Total parameter combinations
        self.n_params = len(alphas) * len(weights)
        
        # Results storage
        self.cv_stats = None
        self.best_alphas_ = None
        self.best_weights_ = None
        self.best_scores_ = None
        self.parameter_map_ = None
        
    def _create_parameter_map(self):
        """Create mapping between flattened parameter index and (alpha, weight)"""
        param_map = {}
        param_idx = 0
        
        for w_idx, weight in enumerate(self.weights):
            for a_idx, alpha in enumerate(self.alphas):
                param_map[param_idx] = {
                    'alpha_idx': a_idx, 
                    'weight_idx': w_idx,
                    'alpha': alpha.item(),
                    'weight': weight.item()
                }
                param_idx += 1
                
        return param_map
    
    def _compute_fold_scores(self, 
                           X1_train: torch.Tensor, X2_train: torch.Tensor, Y_train: torch.Tensor,
                           X1_val: torch.Tensor, X2_val: torch.Tensor, Y_val: torch.Tensor) -> torch.Tensor:
        """
        Compute validation scores for all (alpha, weight) combinations
        
        Returns:
            fold_scores: (n_targets, n_params) tensor
        """
        n_targets = Y_val.shape[1]
        fold_scores = torch.zeros(n_targets, self.n_params, device=self.device)
        
        param_idx = 0
        for weight in self.weights:
            # Fit weighted ridge for all alphas
            weighted_ridge = WeightedMultiRidge(
                self.alphas, weight.item(), device=self.device
            )
            weighted_ridge.fit(X1_train, X2_train, Y_train)
            
            # Get scores for all alphas
            alpha_scores = weighted_ridge.get_prediction_scores(
                X1_val, X2_val, Y_val, self.scoring_func
            )
            
            # Store in flattened parameter space
            n_alphas = len(self.alphas)
            fold_scores[:, param_idx:param_idx+n_alphas] = alpha_scores
            param_idx += n_alphas
            
            # Free GPU memory
            del weighted_ridge
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return fold_scores
    
    def fit(self, X1: torch.Tensor, X2: torch.Tensor, Y: torch.Tensor, 
           verbose: bool = True) -> 'IncrementalConvexCombinationCV':
        """
        Fit convex combination with incremental cross-validation
        
        Args:
            X1: First feature set (vision)
            X2: Second feature set (language)
            Y: EEG targets (n_samples, n_channels*n_timepoints)
        """
        # Move data to device
        X1 = X1.to(self.device)
        X2 = X2.to(self.device)
        Y = Y.to(self.device)
        
        n_targets = Y.shape[1]
        
        if verbose:
            print(f"Convex combination CV: {len(self.alphas)} alphas × {len(self.weights)} weights = {self.n_params} combinations")
            print(f"Targets: {n_targets}, CV folds: {self.cv_folds}")
        
        # Initialize incremental statistics
        self.cv_stats = IncrementalCVStats(n_targets, self.n_params, self.device)
        
        # Create parameter mapping
        self.parameter_map_ = self._create_parameter_map()
        
        # Cross-validation loop
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X1)):
            if verbose:
                print(f"  Processing fold {fold_idx + 1}/{self.cv_folds}...")
            
            # Split data
            X1_train, X1_val = X1[train_idx], X1[val_idx]
            X2_train, X2_val = X2[train_idx], X2[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
            # Compute fold scores for all parameter combinations
            fold_scores = self._compute_fold_scores(
                X1_train, X2_train, Y_train,
                X1_val, X2_val, Y_val
            )
            
            # Update incremental statistics (key memory optimization!)
            self.cv_stats.update(fold_scores)
            
            # Free memory immediately
            del fold_scores
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Check for early stopping
            if self._check_early_stopping(fold_idx):
                if verbose:
                    print(f"  Early stopping at fold {fold_idx + 1}")
                break
        
        # Extract best parameters
        self._extract_best_parameters()
        
        if verbose:
            print(f"  CV completed with {self.cv_stats.n_folds_seen} folds")
            print(f"  Mean best score: {self.best_scores_.mean():.4f}")
        
        return self
    
    def _check_early_stopping(self, fold_idx: int) -> bool:
        """Check if early stopping criteria are met"""
        if fold_idx < self.min_folds_before_stop:
            return False
            
        # Placeholder for early stopping logic
        # Could implement confidence interval-based stopping
        return False
    
    def _extract_best_parameters(self):
        """Extract best alpha and weight for each target"""
        mean_scores, _ = self.cv_stats.get_statistics()
        
        # Find best parameter combination per target
        best_param_indices = torch.argmax(mean_scores, dim=1)
        
        self.best_scores_ = torch.tensor([
            mean_scores[t, best_param_indices[t]] for t in range(len(best_param_indices))
        ], device=self.device)
        
        # Map back to alpha and weight
        best_alphas = []
        best_weights = []
        
        for param_idx in best_param_indices:
            param_info = self.parameter_map_[param_idx.item()]
            best_alphas.append(param_info['alpha'])
            best_weights.append(param_info['weight'])
        
        self.best_alphas_ = torch.tensor(best_alphas, device=self.device)
        self.best_weights_ = torch.tensor(best_weights, device=self.device)
    
    def get_best_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return best parameters for each target"""
        if self.best_alphas_ is None:
            raise RuntimeError("Must call fit() first")
        
        return self.best_alphas_, self.best_weights_, self.best_scores_


# High-level optimization strategies

def hierarchical_convex_optimization(
    X1: torch.Tensor, X2: torch.Tensor, Y: torch.Tensor,
    device=None, verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hierarchical optimization: weight first, then alpha
    Memory-efficient for large problems
    """
    if verbose:
        print("Hierarchical Convex Optimization")
        print("="*50)
    
    # Stage 1: Find best weights with fixed alpha
    if verbose:
        print("Stage 1: Weight optimization (α = 1.0)")
    
    fixed_alpha = torch.tensor([1.0], device=device)
    weights_coarse = torch.linspace(0, 1, 11, device=device)  # 11 points
    
    weight_cv = IncrementalConvexCombinationCV(
        alphas=fixed_alpha,
        weights=weights_coarse,
        cv_folds=5,
        device=device
    )
    weight_cv.fit(X1, X2, Y, verbose=False)
    
    # Get best weight per target
    _, best_weights_stage1, _ = weight_cv.get_best_params()
    
    # Stage 2: Find best alphas with optimal weights
    if verbose:
        print("Stage 2: Alpha optimization with optimal weights")
    
    alphas = torch.logspace(-6, 6, 50, device=device)  # 50 alphas
    
    # Use median weight for simplicity (could optimize per target)
    median_weight = torch.median(best_weights_stage1)
    
    alpha_cv = IncrementalConvexCombinationCV(
        alphas=alphas,
        weights=torch.tensor([median_weight], device=device),
        cv_folds=5,
        device=device
    )
    alpha_cv.fit(X1, X2, Y, verbose=False)
    
    best_alphas_stage2, _, best_scores = alpha_cv.get_best_params()
    
    if verbose:
        print(f"Optimal median weight: {median_weight:.4f}")
        print(f"Mean optimal alpha: {best_alphas_stage2.mean():.2e}")
        print(f"Mean best score: {best_scores.mean():.4f}")
    
    # Return per-target results (simplified: use median weight for all)
    best_weights = torch.full_like(best_alphas_stage2, median_weight)
    
    return best_alphas_stage2, best_weights, best_scores


def coarse_to_fine_convex_optimization(
    X1: torch.Tensor, X2: torch.Tensor, Y: torch.Tensor,
    device=None, verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Coarse-to-fine grid search
    Reduces parameter combinations from 4100 to ~200
    """
    if verbose:
        print("Coarse-to-Fine Convex Optimization")
        print("="*50)
    
    # Stage 1: Coarse grid
    alphas_coarse = torch.logspace(-4, 4, 5, device=device)  # 5 alphas
    weights_coarse = torch.linspace(0, 1, 5, device=device)   # 5 weights
    
    if verbose:
        print(f"Stage 1: Coarse grid ({len(alphas_coarse)}×{len(weights_coarse)} = {len(alphas_coarse)*len(weights_coarse)} combinations)")
    
    coarse_cv = IncrementalConvexCombinationCV(
        alphas=alphas_coarse,
        weights=weights_coarse,
        cv_folds=5,
        device=device
    )
    coarse_cv.fit(X1, X2, Y, verbose=False)
    
    best_alphas_coarse, best_weights_coarse, _ = coarse_cv.get_best_params()
    
    # Stage 2: Fine grid around median best parameters
    median_best_alpha = torch.median(best_alphas_coarse)
    median_best_weight = torch.median(best_weights_coarse)
    
    # Create fine grids around medians
    alpha_factor = 3.0  # Search ±3x around best
    weight_margin = 0.2  # Search ±0.2 around best
    
    alpha_min = max(median_best_alpha / alpha_factor, 1e-6)
    alpha_max = min(median_best_alpha * alpha_factor, 1e6)
    alphas_fine = torch.logspace(
        torch.log10(alpha_min), torch.log10(alpha_max), 20, device=device
    )
    
    weight_min = max(median_best_weight - weight_margin, 0.0)
    weight_max = min(median_best_weight + weight_margin, 1.0)
    weights_fine = torch.linspace(weight_min, weight_max, 10, device=device)
    
    if verbose:
        print(f"Stage 2: Fine grid around α={median_best_alpha:.2e}, w={median_best_weight:.3f}")
        print(f"  Fine grid: {len(alphas_fine)}×{len(weights_fine)} = {len(alphas_fine)*len(weights_fine)} combinations")
    
    fine_cv = IncrementalConvexCombinationCV(
        alphas=alphas_fine,
        weights=weights_fine,
        cv_folds=5,
        device=device
    )
    fine_cv.fit(X1, X2, Y, verbose=False)
    
    best_alphas, best_weights, best_scores = fine_cv.get_best_params()
    
    if verbose:
        print(f"Final: α∈[{best_alphas.min():.2e}, {best_alphas.max():.2e}], "
              f"w∈[{best_weights.min():.3f}, {best_weights.max():.3f}]")
        print(f"Mean best score: {best_scores.mean():.4f}")
    
    return best_alphas, best_weights, best_scores


def optimize_convex_combination_auto(
    X1: torch.Tensor, X2: torch.Tensor, Y: torch.Tensor,
    memory_budget_gb: float = 8.0,
    optimization_strategy: str = 'auto',
    device=None,
    verbose: bool = True
) -> Dict:
    """
    Automatically choose and run optimal convex combination strategy
    
    Returns:
        Dictionary with results and metadata
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move data to device
    X1 = X1.to(device)
    X2 = X2.to(device)  
    Y = Y.to(device)
    
    n_samples, n_targets = Y.shape
    
    if verbose:
        print("CONVEX COMBINATION AUTO-OPTIMIZER")
        print("="*60)
        print(f"Data: {X1.shape} + {X2.shape} → {Y.shape}")
        print(f"Memory budget: {memory_budget_gb}GB")
        print(f"Device: {device}")
    
    # Estimate memory requirements for different strategies
    full_grid_memory_gb = (
        100 * 41 * 5 * n_samples * n_targets * 4  # 100α × 41w × 5CV × samples × targets × 4bytes
    ) / 1e9
    
    bayesian_memory_gb = (
        50 * 5 * n_samples * n_targets * 4  # 50 evaluations
    ) / 1e9
    
    hierarchical_memory_gb = (
        (11 + 50) * 5 * n_samples * n_targets * 4  # 11w + 50α evaluations  
    ) / 1e9
    
    if verbose:
        print(f"\nMemory estimates:")
        print(f"  Full grid: {full_grid_memory_gb:.1f}GB")
        print(f"  Bayesian: {bayesian_memory_gb:.1f}GB")
        print(f"  Hierarchical: {hierarchical_memory_gb:.1f}GB")
    
    # Choose strategy
    if optimization_strategy == 'auto':
        if bayesian_memory_gb < memory_budget_gb * 0.8:
            strategy = 'bayesian'
        elif hierarchical_memory_gb < memory_budget_gb * 0.8:
            strategy = 'hierarchical'
        elif full_grid_memory_gb < memory_budget_gb * 0.8:
            strategy = 'coarse_to_fine'
        else:
            strategy = 'hierarchical'  # Fallback to most memory-efficient
    else:
        strategy = optimization_strategy
    
    if verbose:
        print(f"Selected strategy: {strategy}")
        print()
    
    # Run optimization
    start_time = time.time()
    
    if strategy == 'bayesian':
        bo = BayesianOptimizationConvexCombination(
            alpha_bounds=(1e-6, 1e6),
            weight_bounds=(0.0, 1.0),
            n_calls=50,
            device=device
        )
        best_alpha, best_weight, best_score = bo.optimize(X1, X2, Y, verbose=verbose)
        
        # Convert single values to per-target tensors (simplified)
        best_alphas = torch.full((n_targets,), best_alpha, device=device)
        best_weights = torch.full((n_targets,), best_weight, device=device)
        best_scores = torch.full((n_targets,), best_score, device=device)
        
    elif strategy == 'hierarchical':
        best_alphas, best_weights, best_scores = hierarchical_convex_optimization(
            X1, X2, Y, device=device, verbose=verbose
        )
        
    elif strategy == 'coarse_to_fine':
        best_alphas, best_weights, best_scores = coarse_to_fine_convex_optimization(
            X1, X2, Y, device=device, verbose=verbose
        )
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    end_time = time.time()
    
    results = {
        'strategy': strategy,
        'best_alphas': best_alphas.cpu(),
        'best_weights': best_weights.cpu(), 
        'best_scores': best_scores.cpu(),
        'optimization_time': end_time - start_time,
        'mean_score': best_scores.mean().item(),
        'alpha_range': (best_alphas.min().item(), best_alphas.max().item()),
        'weight_range': (best_weights.min().item(), best_weights.max().item())
    }
    
    if verbose:
        print(f"\nOptimization completed in {results['optimization_time']:.1f}s")
        print(f"Best mean score: {results['mean_score']:.4f}")
        print(f"Alpha range: [{results['alpha_range'][0]:.2e}, {results['alpha_range'][1]:.2e}]")
        print(f"Weight range: [{results['weight_range'][0]:.3f}, {results['weight_range'][1]:.3f}]")
    
    return results


# Example usage
if __name__ == "__main__":
    print("Testing Convex Combination Optimizer")
    
    # Create synthetic EEG-like data
    n_samples = 800
    n_vision_features = 500
    n_language_features = 500
    n_eeg_targets = 200  # Reduced for testing
    
    X1 = torch.randn(n_samples, n_vision_features)  # Vision features
    X2 = torch.randn(n_samples, n_language_features)  # Language features
    Y = torch.randn(n_samples, n_eeg_targets)  # EEG responses
    
    print(f"Test data: Vision{X1.shape} + Language{X2.shape} → EEG{Y.shape}")
    
    # Test automatic optimization
    results = optimize_convex_combination_auto(
        X1, X2, Y, 
        memory_budget_gb=4.0,
        optimization_strategy='auto',
        verbose=True
    )
    
    print("\nTest completed successfully!")