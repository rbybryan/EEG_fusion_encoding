"""
Memory-efficient batch multi-target ridge regression with incremental CV
Integrates incremental CV approach into batch processing for massive memory savings
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional
from sklearn.model_selection import KFold
import warnings
import gc

# Import the incremental CV classes
from incremental_cv_implementation import (
    MemoryEfficientRidgeCV, 
    IncrementalCVStats,
    memory_optimal_ridge_cv
)


class BatchMultiRidgeIncremental:
    """
    Multi-target Ridge regression with memory-efficient incremental CV
    Combines batch processing with incremental statistics for optimal memory usage
    """
    
    def __init__(self, alphas: Union[np.ndarray, torch.Tensor], 
                 scale_X: bool = True, device: Optional[torch.device] = None,
                 memory_budget_gb: float = 8.0):
        self.alphas = torch.tensor(alphas, dtype=torch.float32) if isinstance(alphas, np.ndarray) else alphas
        self.scale_X = scale_X
        self.device = device if device is not None else torch.device('cpu')
        self.memory_budget_gb = memory_budget_gb
        
        # Move to device
        self.alphas = self.alphas.to(self.device)
        
        # Storage for fitted models
        self.weights_ = None
        self.bias_ = None
        self.best_alphas_ = None
        self.X_mean_ = None
        self.X_std_ = None
        
    def fit(self, X: torch.Tensor, Y: torch.Tensor, cv_folds: int = 5) -> 'BatchMultiRidgeIncremental':
        """
        Fit Ridge regression using memory-efficient incremental CV
        """
        X, Y = X.to(self.device), Y.to(self.device)
        n_samples, n_features = X.shape
        n_targets = Y.shape[1]
        
        print(f"BatchMultiRidgeIncremental: Training on {n_samples} samples, {n_features} features, {n_targets} targets")
        print(f"Memory budget: {self.memory_budget_gb:.1f} GB")
        
        # Standardize features if requested
        if self.scale_X:
            self.X_mean_ = X.mean(dim=0)
            self.X_std_ = X.std(dim=0)
            self.X_std_[self.X_std_ == 0] = 1
            X = (X - self.X_mean_) / self.X_std_
        
        # Use memory-optimal incremental CV
        print("Using memory-efficient incremental CV...")
        cv_model = memory_optimal_ridge_cv(
            X, Y, self.alphas, 
            memory_budget_gb=self.memory_budget_gb,
            cv_folds=cv_folds,
            device=self.device,
            early_stop_confidence=0.95,
            min_folds_before_stop=3
        )
        
        # Get best parameters
        self.best_alphas_, best_scores = cv_model.get_best_params()
        
        print(f"Selected alphas range: {self.best_alphas_.min():.2e} to {self.best_alphas_.max():.2e}")
        print(f"CV scores range: {best_scores.min():.3f} to {best_scores.max():.3f}")
        
        # Train final models with best alphas
        self._fit_final_models(X, Y)
        
        return self
    
    def _fit_final_models(self, X: torch.Tensor, Y: torch.Tensor):
        """Train final models with selected alphas"""
        n_features, n_targets = X.shape[1], Y.shape[1]
        
        self.weights_ = torch.zeros(n_features, n_targets, device=self.device)
        self.bias_ = torch.zeros(n_targets, device=self.device)
        
        # Group targets by alpha value for efficiency
        unique_alphas, inverse_indices = torch.unique(self.best_alphas_, return_inverse=True)
        
        for alpha_idx, alpha in enumerate(unique_alphas):
            target_mask = inverse_indices == alpha_idx
            target_indices = torch.where(target_mask)[0]
            
            if len(target_indices) > 0:
                # Solve ridge system for this alpha
                XtX = X.T @ X
                XtY = X.T @ Y[:, target_indices]
                reg_matrix = XtX + alpha * torch.eye(X.shape[1], device=self.device)
                
                try:
                    weights = torch.linalg.solve(reg_matrix, XtY)
                    self.weights_[:, target_indices] = weights
                    
                    # Compute bias
                    Y_pred_no_bias = X @ weights
                    self.bias_[target_indices] = Y[:, target_indices].mean(dim=0) - Y_pred_no_bias.mean(dim=0)
                    
                except torch.linalg.LinAlgError:
                    # Fallback to pseudo-inverse
                    weights = torch.pinverse(reg_matrix) @ XtY
                    self.weights_[:, target_indices] = weights
                    
                    Y_pred_no_bias = X @ weights
                    self.bias_[target_indices] = Y[:, target_indices].mean(dim=0) - Y_pred_no_bias.mean(dim=0)
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions for all targets"""
        if self.weights_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = X.to(self.device)
        
        # Apply same standardization as training
        if self.scale_X:
            X = (X - self.X_mean_) / self.X_std_
        
        return X @ self.weights_ + self.bias_


class IncrementalFusionCV:
    """
    Incremental CV for fusion with 2D parameter space (alpha, weight)
    Extends incremental approach to handle convex combination optimization
    """
    
    def __init__(self, alphas: torch.Tensor, weights: torch.Tensor, 
                 cv_folds: int = 5, device=None, memory_budget_gb: float = 8.0):
        self.alphas = alphas.to(device) if device else alphas
        self.weights = weights.to(device) if device else weights  
        self.cv_folds = cv_folds
        self.device = device or torch.device('cpu')
        self.memory_budget_gb = memory_budget_gb
        
        # Results
        self.best_alphas_ = None
        self.best_weights_ = None
        self.best_scores_ = None
        
    def fit(self, X0_train: torch.Tensor, X1_train: torch.Tensor, Y_train: torch.Tensor) -> 'IncrementalFusionCV':
        """
        Fit fusion model with incremental CV over (alpha, weight) parameter space
        """
        n_samples, n_targets = Y_train.shape
        n_params = len(self.alphas) * len(self.weights)
        
        print(f"IncrementalFusionCV: {len(self.alphas)} alphas × {len(self.weights)} weights = {n_params} combinations")
        
        # Initialize incremental statistics for 2D parameter space
        best_scores = torch.full((n_targets,), -float('inf'), device=self.device)
        best_alphas = torch.zeros(n_targets, device=self.device)
        best_weights = torch.zeros(n_targets, device=self.device)
        
        # Create CV folds
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # Estimate memory usage and decide strategy
        estimated_memory_gb = self._estimate_memory_usage(n_samples, n_targets, n_params)
        
        if estimated_memory_gb > self.memory_budget_gb:
            print(f"High memory usage estimated ({estimated_memory_gb:.1f} GB), using streaming approach")
            return self._fit_streaming(X0_train, X1_train, Y_train, kfold, best_scores, best_alphas, best_weights)
        else:
            print(f"Memory usage manageable ({estimated_memory_gb:.1f} GB), using batch approach")
            return self._fit_batch(X0_train, X1_train, Y_train, kfold, best_scores, best_alphas, best_weights)
    
    def _estimate_memory_usage(self, n_samples: int, n_targets: int, n_params: int) -> float:
        """Estimate memory usage in GB"""
        # Memory for storing CV statistics per (target, param) combination
        cv_memory = n_targets * n_params * 2 * 4 / 1e9  # mean + variance, 4 bytes each
        return cv_memory
    
    def _fit_streaming(self, X0_train, X1_train, Y_train, kfold, best_scores, best_alphas, best_weights):
        """Streaming approach: process weights one at a time"""
        
        for weight in self.weights:
            print(f"Processing weight {weight:.3f}...")
            
            # Create combined features for this weight
            X_combined = self._create_combined_features(X0_train, X1_train, weight)
            
            # Use incremental CV for alpha optimization at this weight
            cv_model = memory_optimal_ridge_cv(
                X_combined, Y_train, self.alphas,
                memory_budget_gb=self.memory_budget_gb / len(self.weights),
                cv_folds=self.cv_folds,
                device=self.device
            )
            
            # Update best parameters for each target
            weight_best_alphas, weight_best_scores = cv_model.get_best_params()
            
            for target_idx in range(len(weight_best_scores)):
                if weight_best_scores[target_idx] > best_scores[target_idx]:
                    best_scores[target_idx] = weight_best_scores[target_idx]
                    best_alphas[target_idx] = weight_best_alphas[target_idx]
                    best_weights[target_idx] = weight
            
            # Clear memory
            del cv_model, X_combined
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        self.best_scores_ = best_scores
        self.best_alphas_ = best_alphas  
        self.best_weights_ = best_weights
        return self
    
    def _fit_batch(self, X0_train, X1_train, Y_train, kfold, best_scores, best_alphas, best_weights):
        """Batch approach: process all combinations with incremental statistics"""
        n_targets = Y_train.shape[1]
        
        # Initialize incremental stats for all (alpha, weight) combinations  
        param_combinations = [(alpha, weight) for alpha in self.alphas for weight in self.weights]
        stats = IncrementalCVStats(n_targets, len(param_combinations), self.device)
        
        # CV loop
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X0_train.cpu().numpy())):
            print(f"Processing fold {fold_idx + 1}/{self.cv_folds}...")
            
            X0_fold_train, X1_fold_train = X0_train[train_idx], X1_train[train_idx]
            X0_fold_val, X1_fold_val = X0_train[val_idx], X1_train[val_idx]
            Y_fold_train, Y_fold_val = Y_train[train_idx], Y_train[val_idx]
            
            # Compute scores for all parameter combinations
            fold_scores = torch.zeros(n_targets, len(param_combinations), device=self.device)
            
            for param_idx, (alpha, weight) in enumerate(param_combinations):
                # Create combined features
                X_combined_train = self._create_combined_features(X0_fold_train, X1_fold_train, weight)
                X_combined_val = self._create_combined_features(X0_fold_val, X1_fold_val, weight)
                
                # Fit ridge and compute scores
                scores = self._compute_ridge_scores(X_combined_train, Y_fold_train, 
                                                 X_combined_val, Y_fold_val, alpha)
                fold_scores[:, param_idx] = scores
            
            # Update incremental statistics
            stats.update(fold_scores)
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Extract best parameters
        mean_scores, _ = stats.get_statistics()
        best_param_indices = torch.argmax(mean_scores, dim=1)
        
        for target_idx in range(n_targets):
            param_idx = best_param_indices[target_idx]
            alpha, weight = param_combinations[param_idx]
            best_alphas[target_idx] = alpha
            best_weights[target_idx] = weight
            best_scores[target_idx] = mean_scores[target_idx, param_idx]
        
        self.best_scores_ = best_scores
        self.best_alphas_ = best_alphas
        self.best_weights_ = best_weights
        return self
    
    def _create_combined_features(self, X0: torch.Tensor, X1: torch.Tensor, weight: float) -> torch.Tensor:
        """Create weighted combination of features"""
        X_combined = torch.cat([X0, X1], dim=1)
        n_feat0 = X0.shape[1]
        X_combined[:, :n_feat0] *= (1 - weight)  # Vision features
        X_combined[:, n_feat0:] *= weight        # Language features  
        return X_combined
    
    def _compute_ridge_scores(self, X_train: torch.Tensor, Y_train: torch.Tensor, 
                             X_val: torch.Tensor, Y_val: torch.Tensor, alpha: float) -> torch.Tensor:
        """Compute ridge regression scores for all targets"""
        # Ridge regression: β = (X'X + αI)^(-1)X'Y
        XtX = X_train.T @ X_train
        XtY = X_train.T @ Y_train
        reg_matrix = XtX + alpha * torch.eye(X_train.shape[1], device=self.device)
        
        try:
            weights = torch.linalg.solve(reg_matrix, XtY)
            Y_pred = X_val @ weights
            
            # Compute R² scores for each target
            scores = torch.zeros(Y_val.shape[1], device=self.device)
            for target_idx in range(Y_val.shape[1]):
                y_true = Y_val[:, target_idx]
                y_pred = Y_pred[:, target_idx]
                
                ss_res = ((y_true - y_pred) ** 2).sum()
                ss_tot = ((y_true - y_true.mean()) ** 2).sum()
                scores[target_idx] = 1 - ss_res / (ss_tot + 1e-8)
            
            return scores
            
        except torch.linalg.LinAlgError:
            # Return very low scores for singular matrices
            return torch.full((Y_val.shape[1],), -1e6, device=self.device)


def batch_encoding_incremental(
    X_train: np.ndarray, X_test: np.ndarray, 
    y_train: np.ndarray, y_test: np.ndarray,
    alpha_range: np.ndarray,
    metric: str = 'r2', cv_folds: int = 5,
    device: Optional[torch.device] = None,
    memory_budget_gb: float = 8.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Memory-efficient batch encoding with incremental CV
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Memory-efficient batch encoding using device: {device}")
    
    # Reshape data
    n_train, n_channels, n_timepoints = y_train.shape
    n_test = y_test.shape[0]
    n_targets = n_channels * n_timepoints
    
    Y_train_flat = torch.tensor(y_train.reshape(n_train, n_targets), dtype=torch.float32)
    Y_test_flat = torch.tensor(y_test.reshape(n_test, n_targets), dtype=torch.float32)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    print(f"Training {n_targets} targets with memory budget {memory_budget_gb:.1f} GB")
    
    # Use incremental batch model
    model = BatchMultiRidgeIncremental(
        alphas=alpha_range, device=device, memory_budget_gb=memory_budget_gb
    )
    model.fit(X_train_tensor, Y_train_flat, cv_folds=cv_folds)
    
    # Make predictions
    Y_pred_flat = model.predict(X_test_tensor)
    
    # Reshape results
    predictions = Y_pred_flat.cpu().numpy().reshape(n_test, n_channels, n_timepoints)
    best_alphas_reshaped = model.best_alphas_.cpu().numpy().reshape(n_channels, n_timepoints)
    
    # Compute scores  
    scores = np.zeros((n_channels, n_timepoints))
    for c in range(n_channels):
        for t in range(n_timepoints):
            target_idx = c * n_timepoints + t
            y_true = Y_test_flat[:, target_idx].cpu().numpy()
            y_pred = Y_pred_flat[:, target_idx].cpu().numpy()
            
            if metric.lower() == 'r2':
                from sklearn.metrics import r2_score
                scores[c, t] = r2_score(y_true, y_pred)
    
    return predictions, best_alphas_reshaped, scores


def batch_fusion_encoding_incremental(
    X0_train: np.ndarray, X0_test: np.ndarray,
    X1_train: np.ndarray, X1_test: np.ndarray,  
    y_train: np.ndarray, y_test: np.ndarray,
    alpha_range: np.ndarray, weight_list: np.ndarray,
    metric: str = 'r2', cv_folds: int = 5,
    device: Optional[torch.device] = None,
    memory_budget_gb: float = 8.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Memory-efficient batch fusion encoding with incremental CV
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Memory-efficient batch fusion encoding using device: {device}")
    
    # Reshape data
    n_train, n_channels, n_timepoints = y_train.shape
    n_test = y_test.shape[0]
    n_targets = n_channels * n_timepoints
    
    Y_train_flat = torch.tensor(y_train.reshape(n_train, n_targets), dtype=torch.float32)
    Y_test_flat = torch.tensor(y_test.reshape(n_test, n_targets), dtype=torch.float32)
    
    X0_train_tensor = torch.tensor(X0_train, dtype=torch.float32, device=device)
    X1_train_tensor = torch.tensor(X1_train, dtype=torch.float32, device=device)
    X0_test_tensor = torch.tensor(X0_test, dtype=torch.float32, device=device)
    X1_test_tensor = torch.tensor(X1_test, dtype=torch.float32, device=device)
    
    Y_train_flat = Y_train_flat.to(device)
    Y_test_flat = Y_test_flat.to(device)
    
    # Use incremental fusion CV
    fusion_cv = IncrementalFusionCV(
        alphas=torch.tensor(alpha_range, device=device),
        weights=torch.tensor(weight_list, device=device),
        cv_folds=cv_folds, device=device, memory_budget_gb=memory_budget_gb
    )
    
    fusion_cv.fit(X0_train_tensor, X1_train_tensor, Y_train_flat)
    
    # Train final models and make predictions
    predictions = torch.zeros(n_test, n_targets, device=device)
    
    # Group by parameter combinations for efficiency
    unique_params = torch.unique(torch.stack([fusion_cv.best_alphas_, fusion_cv.best_weights_], dim=1), dim=0)
    
    for param_combo in unique_params:
        alpha, weight = param_combo[0], param_combo[1]
        
        mask = (fusion_cv.best_alphas_ == alpha) & (fusion_cv.best_weights_ == weight)
        target_indices = torch.where(mask)[0]
        
        if len(target_indices) > 0:
            # Create combined features
            X_combined_train = fusion_cv._create_combined_features(X0_train_tensor, X1_train_tensor, weight)
            X_combined_test = fusion_cv._create_combined_features(X0_test_tensor, X1_test_tensor, weight)
            
            # Fit and predict
            XtX = X_combined_train.T @ X_combined_train
            XtY = X_combined_train.T @ Y_train_flat[:, target_indices]
            reg_matrix = XtX + alpha * torch.eye(X_combined_train.shape[1], device=device)
            weights_matrix = torch.linalg.solve(reg_matrix, XtY)
            
            predictions[:, target_indices] = X_combined_test @ weights_matrix
    
    # Convert results
    predictions_np = predictions.cpu().numpy().reshape(n_test, n_channels, n_timepoints)
    best_alphas_np = fusion_cv.best_alphas_.cpu().numpy().reshape(n_channels, n_timepoints)
    best_weights_np = fusion_cv.best_weights_.cpu().numpy().reshape(n_channels, n_timepoints)
    scores_np = fusion_cv.best_scores_.cpu().numpy().reshape(n_channels, n_timepoints)
    
    return predictions_np, best_alphas_np, best_weights_np, scores_np