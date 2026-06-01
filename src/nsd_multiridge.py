"""
Multi-target Ridge regression implementation following NSD approach
Trains on all EEG channels and timepoints simultaneously instead of individually
"""

import numpy as np
import torch
from typing import Union, Tuple, Optional, List
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

class MultiRidge:
    """
    Multi-target Ridge regression that trains on all EEG (channel,timepoint) simultaneously
    Based on NSD approach using Woodbury matrix identity for efficiency
    """
    
    def __init__(self, alphas: Union[np.ndarray, torch.Tensor], 
                 scale_X: bool = True, device: Optional[torch.device] = None):
        """
        Initialize MultiRidge
        
        Args:
            alphas: Regularization parameters to search over
            scale_X: Whether to standardize features
            device: PyTorch device for computation
        """
        self.alphas = torch.tensor(alphas, dtype=torch.float32) if isinstance(alphas, np.ndarray) else alphas
        self.scale_X = scale_X
        self.device = device if device is not None else torch.device('cpu')
        
        # Move to device
        self.alphas = self.alphas.to(self.device)
        
        # Storage for fitted models
        self.weights_ = None
        self.bias_ = None
        self.best_alphas_ = None
        self.X_mean_ = None
        self.X_std_ = None
        
    def fit(self, X: torch.Tensor, Y: torch.Tensor, cv_folds: int = 5) -> 'MultiRidge':
        """
        Fit Ridge regression for all targets simultaneously using cross-validation
        
        Args:
            X: Features of shape (n_samples, n_features)
            Y: Targets of shape (n_samples, n_targets) where n_targets = n_channels * n_timepoints
            cv_folds: Number of CV folds for alpha selection
            
        Returns:
            self: Fitted MultiRidge instance
        """
        X, Y = X.to(self.device), Y.to(self.device)
        n_samples, n_features = X.shape
        n_targets = Y.shape[1]
        
        print(f"MultiRidge: Training on {n_samples} samples, {n_features} features, {n_targets} targets")
        print(f"MultiRidge: Testing {len(self.alphas)} alpha values with {cv_folds}-fold CV")
        
        # Standardize features if requested
        if self.scale_X:
            self.X_mean_ = X.mean(dim=0)
            self.X_std_ = X.std(dim=0)
            self.X_std_[self.X_std_ == 0] = 1  # Avoid division by zero
            X = (X - self.X_mean_) / self.X_std_
        
        # Perform cross-validation for each target independently
        self.best_alphas_ = torch.zeros(n_targets, device=self.device)
        
        # Use KFold CV to select best alpha for each target
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = torch.zeros(len(self.alphas), n_targets, device=self.device)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X.cpu().numpy())):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
            # Fit ridge for all alphas and targets in this fold
            for alpha_idx, alpha in enumerate(self.alphas):
                # Compute ridge solution using Woodbury identity: β = (X'X + αI)^(-1)X'Y
                XtX = X_train.T @ X_train
                XtY = X_train.T @ Y_train
                
                # Add regularization: (X'X + αI)^(-1)
                reg_matrix = XtX + alpha * torch.eye(n_features, device=self.device)
                
                try:
                    # Solve for weights: β = (X'X + αI)^(-1)X'Y
                    weights = torch.linalg.solve(reg_matrix, XtY)
                    
                    # Make predictions on validation set
                    Y_pred = X_val @ weights
                    
                    # Compute R² score for each target
                    for target_idx in range(n_targets):
                        y_true = Y_val[:, target_idx]
                        y_pred = Y_pred[:, target_idx]
                        
                        # R² = 1 - SS_res / SS_tot
                        ss_res = ((y_true - y_pred) ** 2).sum()
                        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
                        r2 = 1 - ss_res / (ss_tot + 1e-8)  # Small epsilon to avoid division by zero
                        
                        cv_scores[alpha_idx, target_idx] += r2
                        
                except torch.linalg.LinAlgError:
                    # If matrix is singular, set very low score
                    cv_scores[alpha_idx, :] += -1e6
        
        # Average across CV folds
        cv_scores /= cv_folds
        
        # Select best alpha for each target
        best_alpha_indices = cv_scores.argmax(dim=0)
        self.best_alphas_ = self.alphas[best_alpha_indices]
        
        print(f"MultiRidge: Selected alphas range from {self.best_alphas_.min():.2e} to {self.best_alphas_.max():.2e}")
        
        # Train final models with best alphas on full training data
        self.weights_ = torch.zeros(n_features, n_targets, device=self.device)
        self.bias_ = torch.zeros(n_targets, device=self.device)
        
        # For efficiency, group targets by alpha value
        unique_alphas, inverse_indices = torch.unique(self.best_alphas_, return_inverse=True)
        
        for alpha_idx, alpha in enumerate(unique_alphas):
            target_mask = inverse_indices == alpha_idx
            target_indices = torch.where(target_mask)[0]
            
            if len(target_indices) > 0:
                # Compute ridge solution for this alpha
                XtX = X.T @ X
                XtY = X.T @ Y[:, target_indices]
                reg_matrix = XtX + alpha * torch.eye(n_features, device=self.device)
                
                try:
                    weights = torch.linalg.solve(reg_matrix, XtY)
                    self.weights_[:, target_indices] = weights
                    
                    # Compute bias (intercept)
                    Y_pred_no_bias = X @ weights
                    self.bias_[target_indices] = Y[:, target_indices].mean(dim=0) - Y_pred_no_bias.mean(dim=0)
                    
                except torch.linalg.LinAlgError:
                    warnings.warn(f"Singular matrix for alpha={alpha}, using pseudo-inverse")
                    weights = torch.pinverse(reg_matrix) @ XtY
                    self.weights_[:, target_indices] = weights
                    
                    Y_pred_no_bias = X @ weights
                    self.bias_[target_indices] = Y[:, target_indices].mean(dim=0) - Y_pred_no_bias.mean(dim=0)
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Make predictions for all targets
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples, n_targets)
        """
        if self.weights_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = X.to(self.device)
        
        # Apply same standardization as training
        if self.scale_X:
            X = (X - self.X_mean_) / self.X_std_
        
        # Make predictions: Y = Xβ + bias
        predictions = X @ self.weights_ + self.bias_
        return predictions
    
    def get_weights_and_bias(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get fitted model parameters
        
        Returns:
            Tuple of (weights, bias, best_alphas)
        """
        return self.weights_, self.bias_, self.best_alphas_


def nsd_style_encoding(
    X_train: np.ndarray,
    X_test: np.ndarray, 
    y_train: np.ndarray,
    y_test: np.ndarray,
    alpha_range: np.ndarray,
    metric: str = 'r2',
    cv_folds: int = 5,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    NSD-style encoding that trains on all EEG (channel,timepoint) pairs simultaneously
    
    Args:
        X_train: Training features (n_samples, n_features)
        X_test: Test features (n_samples, n_features)  
        y_train: Training EEG data (n_samples, n_channels, n_timepoints)
        y_test: Test EEG data (n_samples, n_channels, n_timepoints)
        alpha_range: Alpha values to search over
        metric: Metric for evaluation ('r2', 'mse', 'mae')
        cv_folds: Number of CV folds
        device: PyTorch device
        
    Returns:
        Tuple of (predictions, best_alphas_reshaped, scores)
        - predictions: (n_test_samples, n_channels, n_timepoints) 
        - best_alphas_reshaped: (n_channels, n_timepoints) optimal alpha per location
        - scores: (n_channels, n_timepoints) performance scores
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"NSD-style encoding using device: {device}")
    
    # Reshape EEG data from (n_samples, channels, timepoints) to (n_samples, targets)
    n_train, n_channels, n_timepoints = y_train.shape
    n_test = y_test.shape[0]
    n_targets = n_channels * n_timepoints
    
    # Flatten targets: each (channel, timepoint) becomes one target
    Y_train_flat = torch.tensor(y_train.reshape(n_train, n_targets), dtype=torch.float32)
    Y_test_flat = torch.tensor(y_test.reshape(n_test, n_targets), dtype=torch.float32)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    print(f"Reshaped data: Y_train {Y_train_flat.shape}, Y_test {Y_test_flat.shape}")
    print(f"Training {n_targets} targets simultaneously (vs {n_targets} individual models)")
    
    # Fit MultiRidge model
    model = MultiRidge(alphas=alpha_range, device=device)
    model.fit(X_train_tensor, Y_train_flat, cv_folds=cv_folds)
    
    # Make predictions
    Y_pred_flat = model.predict(X_test_tensor)
    
    # Reshape predictions back to (n_samples, channels, timepoints)
    predictions = Y_pred_flat.cpu().numpy().reshape(n_test, n_channels, n_timepoints)
    
    # Reshape best alphas back to (channels, timepoints)
    best_alphas_reshaped = model.best_alphas_.cpu().numpy().reshape(n_channels, n_timepoints)
    
    # Compute performance scores for each (channel, timepoint)
    scores = np.zeros((n_channels, n_timepoints))
    
    for c in range(n_channels):
        for t in range(n_timepoints):
            target_idx = c * n_timepoints + t
            y_true = Y_test_flat[:, target_idx].cpu().numpy()
            y_pred = Y_pred_flat[:, target_idx].cpu().numpy()
            
            if metric.lower() == 'r2':
                scores[c, t] = r2_score(y_true, y_pred)
            elif metric.lower() == 'mse':
                scores[c, t] = -mean_squared_error(y_true, y_pred)  # Negative for consistency
            elif metric.lower() == 'mae':
                scores[c, t] = -mean_absolute_error(y_true, y_pred)  # Negative for consistency
    
    print(f"NSD encoding completed. Performance range: {scores.min():.3f} to {scores.max():.3f}")
    
    return predictions, best_alphas_reshaped, scores


def nsd_fusion_encoding(
    X0_train: np.ndarray, X0_test: np.ndarray,  # Vision features
    X1_train: np.ndarray, X1_test: np.ndarray,  # Language features  
    y_train: np.ndarray, y_test: np.ndarray,    # EEG data
    alpha_range: np.ndarray,
    weight_list: np.ndarray,
    metric: str = 'r2',
    cv_folds: int = 5,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    NSD-style fusion encoding with convex combination optimization
    Optimizes both alpha and weight simultaneously for all targets
    
    Returns:
        Tuple of (predictions, best_alphas, best_weights, scores)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"NSD-style fusion encoding using device: {device}")
    
    # Reshape EEG data 
    n_train, n_channels, n_timepoints = y_train.shape
    n_test = y_test.shape[0]
    n_targets = n_channels * n_timepoints
    
    Y_train_flat = torch.tensor(y_train.reshape(n_train, n_targets), dtype=torch.float32)
    Y_test_flat = torch.tensor(y_test.reshape(n_test, n_targets), dtype=torch.float32)
    
    # Convert features to tensors
    X0_train_tensor = torch.tensor(X0_train, dtype=torch.float32, device=device)
    X1_train_tensor = torch.tensor(X1_train, dtype=torch.float32, device=device)
    X0_test_tensor = torch.tensor(X0_test, dtype=torch.float32, device=device)
    X1_test_tensor = torch.tensor(X1_test, dtype=torch.float32, device=device)
    
    Y_train_flat = Y_train_flat.to(device)
    Y_test_flat = Y_test_flat.to(device)
    
    print(f"Fusion optimization: {len(alpha_range)} alphas × {len(weight_list)} weights = {len(alpha_range)*len(weight_list)} combinations")
    
    # Grid search over alpha and weight combinations
    best_scores = torch.full((n_targets,), -float('inf'), device=device)
    best_alphas = torch.zeros(n_targets, device=device)
    best_weights = torch.zeros(n_targets, device=device)
    
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for weight in weight_list:
        print(f"Testing weight {weight:.3f}...")
        
        # Concatenate features and apply weighted combination (following original approach)
        X_stacked_train = torch.cat([X0_train_tensor, X1_train_tensor], dim=1)
        # Apply weights to the two feature blocks
        n_feat0 = X0_train_tensor.shape[1]
        X_combined_train = X_stacked_train.clone()
        X_combined_train[:, :n_feat0] *= (1 - weight)  # Weight first block (vision)
        X_combined_train[:, n_feat0:] *= weight        # Weight second block (language)
        
        # Use MultiRidge for this weight
        model = MultiRidge(alphas=alpha_range, device=device)
        
        # Perform CV manually to get scores for each target
        cv_scores = torch.zeros(len(alpha_range), n_targets, device=device)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_combined_train.cpu().numpy())):
            X_fold_train = X_combined_train[train_idx]
            X_fold_val = X_combined_train[val_idx] 
            Y_fold_train = Y_train_flat[train_idx]
            Y_fold_val = Y_train_flat[val_idx]
            
            for alpha_idx, alpha in enumerate(alpha_range):
                # Fit ridge model
                XtX = X_fold_train.T @ X_fold_train
                XtY = X_fold_train.T @ Y_fold_train
                n_features = XtX.shape[0]
                
                reg_matrix = XtX + alpha * torch.eye(n_features, device=device)
                
                try:
                    weights_matrix = torch.linalg.solve(reg_matrix, XtY)
                    Y_pred = X_fold_val @ weights_matrix
                    
                    # Compute R² for each target
                    for target_idx in range(n_targets):
                        y_true = Y_fold_val[:, target_idx]
                        y_pred = Y_pred[:, target_idx]
                        
                        ss_res = ((y_true - y_pred) ** 2).sum()
                        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
                        r2 = 1 - ss_res / (ss_tot + 1e-8)
                        
                        cv_scores[alpha_idx, target_idx] += r2
                        
                except torch.linalg.LinAlgError:
                    cv_scores[alpha_idx, :] += -1e6
        
        # Average across folds
        cv_scores /= cv_folds
        
        # Update best parameters for each target
        for target_idx in range(n_targets):
            best_alpha_idx = cv_scores[:, target_idx].argmax()
            best_score = cv_scores[best_alpha_idx, target_idx]
            
            if best_score > best_scores[target_idx]:
                best_scores[target_idx] = best_score
                best_alphas[target_idx] = alpha_range[best_alpha_idx]
                best_weights[target_idx] = weight
    
    print(f"Best weights range: {best_weights.min():.3f} to {best_weights.max():.3f}")
    print(f"Best alphas range: {best_alphas.min():.2e} to {best_alphas.max():.2e}")
    
    # Train final model with best parameters for each target
    predictions = torch.zeros(n_test, n_targets, device=device)
    
    # Group by (alpha, weight) combinations for efficiency
    unique_params = torch.unique(torch.stack([best_alphas, best_weights], dim=1), dim=0)
    
    for param_combo in unique_params:
        alpha, weight = param_combo[0], param_combo[1]
        
        # Find targets with this parameter combination
        mask = (best_alphas == alpha) & (best_weights == weight)
        target_indices = torch.where(mask)[0]
        
        if len(target_indices) > 0:
            # Create combined features (concatenate then weight blocks)
            X_stacked_train = torch.cat([X0_train_tensor, X1_train_tensor], dim=1)
            X_stacked_test = torch.cat([X0_test_tensor, X1_test_tensor], dim=1)
            
            n_feat0 = X0_train_tensor.shape[1]
            X_combined_train = X_stacked_train.clone()
            X_combined_train[:, :n_feat0] *= (1 - weight)
            X_combined_train[:, n_feat0:] *= weight
            
            X_combined_test = X_stacked_test.clone()
            X_combined_test[:, :n_feat0] *= (1 - weight)
            X_combined_test[:, n_feat0:] *= weight
            
            # Fit ridge model
            XtX = X_combined_train.T @ X_combined_train
            XtY = X_combined_train.T @ Y_train_flat[:, target_indices]
            n_features = XtX.shape[0]
            
            reg_matrix = XtX + alpha * torch.eye(n_features, device=device)
            weights_matrix = torch.linalg.solve(reg_matrix, XtY)
            
            # Make predictions
            predictions[:, target_indices] = X_combined_test @ weights_matrix
    
    # Convert back to numpy and reshape
    predictions_np = predictions.cpu().numpy().reshape(n_test, n_channels, n_timepoints)
    best_alphas_np = best_alphas.cpu().numpy().reshape(n_channels, n_timepoints)
    best_weights_np = best_weights.cpu().numpy().reshape(n_channels, n_timepoints)
    scores_np = best_scores.cpu().numpy().reshape(n_channels, n_timepoints)
    
    print(f"NSD fusion encoding completed. Performance range: {scores_np.min():.3f} to {scores_np.max():.3f}")
    
    return predictions_np, best_alphas_np, best_weights_np, scores_np