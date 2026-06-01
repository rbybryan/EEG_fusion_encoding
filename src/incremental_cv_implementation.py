"""
Memory-efficient incremental cross-validation implementation
Based on NSD ridge regression approach but with streaming CV
"""

import torch
import numpy as np
from typing import Tuple, List, Callable, Optional
from sklearn.model_selection import KFold
import gc


class IncrementalCVStats:
    """Track running statistics for cross-validation without storing all results"""
    
    def __init__(self, n_targets: int, n_params: int, device=None):
        self.device = device or torch.device('cpu')
        self.n_folds_seen = 0
        self.mean_scores = torch.zeros(n_targets, n_params, device=self.device)
        self.m2_scores = torch.zeros(n_targets, n_params, device=self.device)  # For variance
        
    def update(self, fold_scores: torch.Tensor):
        """Update running mean and variance using Welford's online algorithm"""
        self.n_folds_seen += 1
        delta = fold_scores - self.mean_scores
        self.mean_scores += delta / self.n_folds_seen
        delta2 = fold_scores - self.mean_scores
        self.m2_scores += delta * delta2
        
    def get_statistics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean scores and standard errors"""
        if self.n_folds_seen < 2:
            return self.mean_scores, torch.zeros_like(self.mean_scores)
        
        variance = self.m2_scores / (self.n_folds_seen - 1)
        std_error = torch.sqrt(variance / self.n_folds_seen)
        return self.mean_scores, std_error


class MemoryEfficientRidgeCV:
    """
    Memory-efficient ridge regression with cross-validation
    Uses incremental statistics instead of storing all predictions
    """
    
    def __init__(self, 
                 alphas: torch.Tensor,
                 cv_folds: int = 5,
                 scoring_func: Optional[Callable] = None,
                 device=None,
                 early_stop_confidence: float = 0.95,
                 min_folds_before_stop: int = 3):
        
        self.alphas = alphas
        self.cv_folds = cv_folds
        self.device = device or torch.device('cpu')
        self.early_stop_confidence = early_stop_confidence
        self.min_folds_before_stop = min_folds_before_stop
        
        # Default scoring function (negative MSE)
        self.scoring_func = scoring_func or (lambda y_true, y_pred: -torch.nn.functional.mse_loss(y_pred, y_true))
        
        # Results
        self.best_scores_ = None
        self.best_alphas_ = None
        self.stats_ = None
        
    def _compute_ridge_scores(self, X_train: torch.Tensor, Y_train: torch.Tensor,
                             X_val: torch.Tensor, Y_val: torch.Tensor) -> torch.Tensor:
        """
        Compute validation scores for all alphas using SVD-based ridge regression
        Returns: (n_targets, n_alphas) tensor of scores
        """
        # Center the data
        X_mean = X_train.mean(dim=0, keepdim=True)
        X_train_centered = X_train - X_mean
        X_val_centered = X_val - X_mean
        
        Y_mean = Y_train.mean(dim=0, keepdim=True)
        Y_train_centered = Y_train - Y_mean
        
        # SVD of centered training features
        U, S, Vt = torch.svd(X_train_centered.T)  # SVD of X^T
        S_squared = S.pow(2)
        
        # Compute scores for all alphas efficiently
        scores = torch.zeros(Y_val.shape[1], len(self.alphas), device=self.device)
        
        for target_idx in range(Y_val.shape[1]):
            y_target = Y_train_centered[:, target_idx]
            
            # Precompute terms that don't depend on alpha
            XTy = X_train_centered.T @ y_target
            UTXTy = U.T @ XTy
            
            for alpha_idx, alpha in enumerate(self.alphas):
                # Ridge solution: (X^T X + alpha I)^{-1} X^T y
                # Using Woodbury identity: (1/alpha) * (XTy - U * (UTXTy / (S^2 + alpha)))
                ridge_coef = (1.0 / alpha) * (XTy - U @ (UTXTy / (S_squared + alpha)))
                
                # Make predictions
                y_pred = X_val_centered @ ridge_coef + Y_mean[0, target_idx]
                
                # Compute score
                scores[target_idx, alpha_idx] = self.scoring_func(
                    Y_val[:, target_idx], y_pred
                ).item()
        
        return scores
    
    def _check_early_stopping(self, stats: IncrementalCVStats, fold_idx: int) -> bool:
        """Check if we can stop CV early based on statistical confidence"""
        if fold_idx < self.min_folds_before_stop:
            return False
            
        mean_scores, std_errors = stats.get_statistics()
        
        # Find current best alpha per target
        best_alpha_indices = torch.argmax(mean_scores, dim=1)
        
        # Critical value for confidence interval (approximate)
        z_score = 1.96 if self.early_stop_confidence == 0.95 else 2.58  # 99%
        
        confident_targets = 0
        total_targets = mean_scores.shape[0]
        
        for target_idx in range(total_targets):
            best_idx = best_alpha_indices[target_idx]
            best_score = mean_scores[target_idx, best_idx]
            best_se = std_errors[target_idx, best_idx]
            
            # Check if best score is significantly better than second best
            other_scores = mean_scores[target_idx].clone()
            other_scores[best_idx] = float('-inf')
            second_best_idx = torch.argmax(other_scores)
            second_best_score = other_scores[second_best_idx]
            second_best_se = std_errors[target_idx, second_best_idx]
            
            # If confidence intervals don't overlap, we're confident
            if best_score - z_score * best_se > second_best_score + z_score * second_best_se:
                confident_targets += 1
        
        # Stop if we're confident about 80% of targets
        return confident_targets / total_targets > 0.8
    
    def fit(self, X: torch.Tensor, Y: torch.Tensor, 
            verbose: bool = False) -> 'MemoryEfficientRidgeCV':
        """
        Fit ridge regression with memory-efficient cross-validation
        
        Args:
            X: Features (n_samples, n_features)
            Y: Targets (n_samples, n_targets)
            verbose: Print progress information
            
        Returns:
            self
        """
        # Move data to device
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Initialize CV statistics
        stats = IncrementalCVStats(Y.shape[1], len(self.alphas), self.device)
        
        # Create CV folds
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # CV loop with potential early stopping
        stopped_early = False
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            if verbose:
                print(f"Processing fold {fold_idx + 1}/{self.cv_folds}...")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
            # Compute scores for this fold
            fold_scores = self._compute_ridge_scores(X_train, Y_train, X_val, Y_val)
            
            # Update running statistics
            stats.update(fold_scores)
            
            # Check for early stopping
            if self._check_early_stopping(stats, fold_idx):
                stopped_early = True
                if verbose:
                    print(f"Early stopping at fold {fold_idx + 1}")
                break
            
            # Clear GPU memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Store results
        self.stats_ = stats
        mean_scores, _ = stats.get_statistics()
        self.best_scores_ = torch.max(mean_scores, dim=1)[0]
        self.best_alphas_ = self.alphas[torch.argmax(mean_scores, dim=1)]
        
        if verbose:
            folds_used = stats.n_folds_seen
            print(f"Completed CV with {folds_used} folds {'(early stop)' if stopped_early else ''}")
            print(f"Memory saved by not storing predictions: "
                  f"{self._estimate_memory_saved(X.shape[0], Y.shape[1]):.2f} GB")
        
        return self
    
    def _estimate_memory_saved(self, n_samples: int, n_targets: int) -> float:
        """Estimate memory saved by using incremental CV"""
        # Memory that would be used storing all CV predictions
        full_storage = len(self.alphas) * self.cv_folds * n_samples * n_targets * 4  # 4 bytes per float32
        
        # Memory actually used (just aggregated statistics)
        actual_storage = len(self.alphas) * n_targets * 2 * 4  # mean + variance per (target, alpha)
        
        return (full_storage - actual_storage) / 1e9
    
    def get_best_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return best alpha and scores for each target"""
        if self.best_alphas_ is None:
            raise RuntimeError("Must call fit() before get_best_params()")
        return self.best_alphas_, self.best_scores_


class BatchStreamingRidgeCV(MemoryEfficientRidgeCV):
    """
    Batch-streaming version that processes alphas in small batches
    Balances memory efficiency with computational efficiency
    """
    
    def __init__(self, alphas: torch.Tensor, batch_size: int = 10, **kwargs):
        super().__init__(alphas, **kwargs)
        self.batch_size = batch_size
    
    def fit(self, X: torch.Tensor, Y: torch.Tensor, verbose: bool = False):
        """Fit using batch processing of alphas"""
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        n_targets = Y.shape[1]
        final_scores = torch.zeros(n_targets, len(self.alphas), device=self.device)
        
        # Process alphas in batches
        n_batches = (len(self.alphas) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.alphas))
            
            if verbose:
                print(f"Processing alpha batch {batch_idx + 1}/{n_batches} "
                      f"(alphas {start_idx}-{end_idx-1})")
            
            # Create mini-CV for this batch
            batch_alphas = self.alphas[start_idx:end_idx]
            batch_cv = MemoryEfficientRidgeCV(
                batch_alphas, self.cv_folds, self.scoring_func, self.device,
                early_stop_confidence=self.early_stop_confidence,
                min_folds_before_stop=self.min_folds_before_stop
            )
            
            # Fit batch
            batch_cv.fit(X, Y, verbose=False)
            batch_mean_scores, _ = batch_cv.stats_.get_statistics()
            
            # Store results
            final_scores[:, start_idx:end_idx] = batch_mean_scores
            
            # Clear memory
            del batch_cv, batch_mean_scores
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Find best alphas across all batches
        self.best_scores_ = torch.max(final_scores, dim=1)[0]
        self.best_alphas_ = self.alphas[torch.argmax(final_scores, dim=1)]
        
        if verbose:
            print(f"Batch processing complete. Peak memory usage reduced by batching.")
        
        return self


def memory_optimal_ridge_cv(X: torch.Tensor, Y: torch.Tensor, 
                          alphas: torch.Tensor, memory_budget_gb: float = 8.0,
                          **kwargs) -> MemoryEfficientRidgeCV:
    """
    Automatically choose the best CV strategy based on memory budget
    
    Args:
        X: Features tensor
        Y: Targets tensor  
        alphas: Alpha values to search
        memory_budget_gb: Available memory budget in GB
        
    Returns:
        Fitted ridge CV model
    """
    # Estimate memory requirements for different approaches
    n_samples, n_features = X.shape
    n_targets = Y.shape[1]
    
    # Full approach memory (storing all CV predictions)
    full_memory_gb = (len(alphas) * 5 * n_samples * n_targets * 4) / 1e9  # 5 CV folds, 4 bytes per float
    
    if full_memory_gb < memory_budget_gb * 0.5:
        # Plenty of memory - use standard approach with early stopping
        return MemoryEfficientRidgeCV(alphas, **kwargs).fit(X, Y, verbose=True)
    
    elif full_memory_gb < memory_budget_gb * 2.0:
        # Moderate memory pressure - use batch streaming
        batch_size = max(5, int(memory_budget_gb * 10 / full_memory_gb))
        return BatchStreamingRidgeCV(alphas, batch_size=batch_size, **kwargs).fit(X, Y, verbose=True)
    
    else:
        # High memory pressure - use small batches
        return BatchStreamingRidgeCV(alphas, batch_size=5, **kwargs).fit(X, Y, verbose=True)


# Example usage
if __name__ == "__main__":
    # Simulate EEG data
    n_stimuli = 1000  # Smaller for testing
    n_channels = 63
    n_timepoints = 180
    n_features = 1000
    
    X = torch.randn(n_stimuli, n_features)
    Y = torch.randn(n_stimuli, n_channels * n_timepoints)
    alphas = torch.logspace(-3, 7, 20)  # Smaller grid for testing
    
    print("Testing memory-efficient ridge CV...")
    
    # Automatic strategy selection
    cv_model = memory_optimal_ridge_cv(X, Y, alphas, memory_budget_gb=2.0)
    
    best_alphas, best_scores = cv_model.get_best_params()
    print(f"Best alphas shape: {best_alphas.shape}")
    print(f"Best scores shape: {best_scores.shape}")
    print(f"Mean best score: {best_scores.mean():.4f}")