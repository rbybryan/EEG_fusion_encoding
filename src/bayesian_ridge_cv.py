"""
Bayesian Optimization for Ridge Regression Parameter Search
Integrates with the incremental CV approach for memory efficiency
"""

import torch
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from typing import Tuple, Callable, Optional, List
import warnings
warnings.filterwarnings('ignore')

from incremental_cv_implementation import MemoryEfficientRidgeCV


class BayesianOptimizationRidge:
    """
    Bayesian Optimization for Ridge regression parameter search
    Uses Gaussian Process to intelligently select alpha parameters
    """
    
    def __init__(self,
                 alpha_bounds: Tuple[float, float] = (1e-6, 1e6),
                 n_calls: int = 30,
                 n_initial_points: int = 5,
                 acquisition_func: str = 'EI',
                 kappa: float = 2.0,
                 device=None):
        """
        Args:
            alpha_bounds: Min and max alpha values to search
            n_calls: Total number of evaluations 
            n_initial_points: Number of random initial evaluations
            acquisition_func: 'EI' (Expected Improvement) or 'UCB' (Upper Confidence Bound)
            kappa: Exploration parameter for UCB
            device: torch device for computations
        """
        self.alpha_bounds = alpha_bounds
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.acquisition_func = acquisition_func
        self.kappa = kappa
        self.device = device or torch.device('cpu')
        
        # Transform bounds to log space for better GP modeling
        self.log_bounds = [np.log10(alpha_bounds[0]), np.log10(alpha_bounds[1])]
        
        # Storage for evaluations
        self.evaluated_log_alphas = []
        self.evaluated_scores = []
        self.gp_model = None
        self.best_alpha_ = None
        self.best_score_ = None
        
    def _objective_function(self, log_alpha: float, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Evaluate ridge regression performance for given alpha
        
        Args:
            log_alpha: Log10 of alpha parameter
            X: Feature matrix
            Y: Target matrix
            
        Returns:
            Negative CV score (to minimize)
        """
        alpha = 10**log_alpha
        
        # Use memory-efficient CV
        ridge_cv = MemoryEfficientRidgeCV(
            alphas=torch.tensor([alpha], device=self.device),
            cv_folds=5,
            device=self.device
        )
        
        ridge_cv.fit(X, Y, verbose=False)
        _, scores = ridge_cv.get_best_params()
        
        # Return negative mean score (BO minimizes)
        return -scores.mean().item()
    
    def _expected_improvement(self, log_alphas: np.ndarray) -> np.ndarray:
        """Expected Improvement acquisition function"""
        if log_alphas.ndim == 1:
            log_alphas = log_alphas.reshape(-1, 1)
            
        mu, sigma = self.gp_model.predict(log_alphas, return_std=True)
        
        # Current best score
        best_score = np.min(self.evaluated_scores)
        
        # Expected improvement calculation
        with np.errstate(divide='warn', invalid='warn'):
            z = (best_score - mu) / sigma
            ei = (best_score - mu) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
            ei[sigma == 0] = 0  # Handle zero variance
            
        return ei
    
    def _upper_confidence_bound(self, log_alphas: np.ndarray) -> np.ndarray:
        """Upper Confidence Bound acquisition function"""
        if log_alphas.ndim == 1:
            log_alphas = log_alphas.reshape(-1, 1)
            
        mu, sigma = self.gp_model.predict(log_alphas, return_std=True)
        return mu + self.kappa * sigma
    
    def _acquisition_function(self, log_alphas: np.ndarray) -> np.ndarray:
        """Select and evaluate acquisition function"""
        if self.acquisition_func == 'EI':
            return -self._expected_improvement(log_alphas)  # Minimize negative EI
        elif self.acquisition_func == 'UCB':
            return -self._upper_confidence_bound(log_alphas)  # Minimize negative UCB
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_func}")
    
    def optimize(self, X: torch.Tensor, Y: torch.Tensor, verbose: bool = True) -> Tuple[float, float]:
        """
        Run Bayesian optimization to find best alpha
        
        Args:
            X: Feature matrix (n_samples, n_features)
            Y: Target matrix (n_samples, n_targets)
            verbose: Print progress
            
        Returns:
            (best_alpha, best_score)
        """
        if verbose:
            print(f"Starting Bayesian Optimization for alpha in [{self.alpha_bounds[0]:.2e}, {self.alpha_bounds[1]:.2e}]")
            print(f"Budget: {self.n_calls} evaluations ({self.n_initial_points} initial + {self.n_calls - self.n_initial_points} BO)")
        
        # Phase 1: Random initial evaluations
        if verbose:
            print("\nPhase 1: Random initialization")
            
        for i in range(self.n_initial_points):
            log_alpha = np.random.uniform(self.log_bounds[0], self.log_bounds[1])
            score = self._objective_function(log_alpha, X, Y)
            
            self.evaluated_log_alphas.append(log_alpha)
            self.evaluated_scores.append(score)
            
            if verbose:
                alpha = 10**log_alpha
                print(f"  {i+1}/{self.n_initial_points}: α={alpha:.2e}, score={-score:.4f}")
        
        # Initialize GP model
        kernel = Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(0.1, 10.0))
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=5,
            normalize_y=True
        )
        
        # Phase 2: Bayesian optimization
        if verbose:
            print(f"\nPhase 2: Bayesian optimization")
            
        for i in range(self.n_calls - self.n_initial_points):
            # Fit GP to current data
            X_gp = np.array(self.evaluated_log_alphas).reshape(-1, 1)
            y_gp = np.array(self.evaluated_scores)
            self.gp_model.fit(X_gp, y_gp)
            
            # Optimize acquisition function
            result = minimize(
                self._acquisition_function,
                x0=[np.random.uniform(self.log_bounds[0], self.log_bounds[1])],
                bounds=[self.log_bounds],
                method='L-BFGS-B',
                options={'maxiter': 100}
            )
            
            # Evaluate at best acquisition point
            next_log_alpha = result.x[0]
            score = self._objective_function(next_log_alpha, X, Y)
            
            self.evaluated_log_alphas.append(next_log_alpha)
            self.evaluated_scores.append(score)
            
            if verbose:
                alpha = 10**next_log_alpha
                best_so_far = -np.min(self.evaluated_scores)
                print(f"  {i+1}/{self.n_calls - self.n_initial_points}: α={alpha:.2e}, "
                      f"score={-score:.4f}, best={best_so_far:.4f}")
        
        # Find best result
        best_idx = np.argmin(self.evaluated_scores)
        self.best_alpha_ = 10**self.evaluated_log_alphas[best_idx]
        self.best_score_ = -self.evaluated_scores[best_idx]
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"Best α: {self.best_alpha_:.2e}")
            print(f"Best score: {self.best_score_:.4f}")
            print(f"Evaluations used: {len(self.evaluated_scores)}")
        
        return self.best_alpha_, self.best_score_


class BayesianOptimizationConvexCombination:
    """
    Bayesian Optimization for convex combination of two feature sets
    Jointly optimizes alpha and weight parameters
    """
    
    def __init__(self,
                 alpha_bounds: Tuple[float, float] = (1e-6, 1e6),
                 weight_bounds: Tuple[float, float] = (0.0, 1.0),
                 n_calls: int = 50,
                 n_initial_points: int = 10,
                 acquisition_func: str = 'EI',
                 device=None):
        
        self.alpha_bounds = alpha_bounds
        self.weight_bounds = weight_bounds
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.acquisition_func = acquisition_func
        self.device = device or torch.device('cpu')
        
        # Transform bounds
        self.log_alpha_bounds = [np.log10(alpha_bounds[0]), np.log10(alpha_bounds[1])]
        self.bounds = [self.log_alpha_bounds, list(weight_bounds)]
        
        # Storage
        self.evaluated_params = []  # [(log_alpha, weight), ...]
        self.evaluated_scores = []
        self.gp_model = None
        self.best_alpha_ = None
        self.best_weight_ = None
        self.best_score_ = None
    
    def _objective_function(self, params: List[float], 
                          X1: torch.Tensor, X2: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Evaluate convex combination performance
        
        Args:
            params: [log_alpha, weight]
            X1: First feature set (e.g., vision)
            X2: Second feature set (e.g., language)
            Y: Target matrix
            
        Returns:
            Negative CV score
        """
        log_alpha, weight = params
        alpha = 10**log_alpha
        
        # Create weighted combination
        X_combined = (1 - weight) * X1 + weight * X2
        
        # Evaluate with CV
        ridge_cv = MemoryEfficientRidgeCV(
            alphas=torch.tensor([alpha], device=self.device),
            cv_folds=5,
            device=self.device
        )
        
        ridge_cv.fit(X_combined, Y, verbose=False)
        _, scores = ridge_cv.get_best_params()
        
        return -scores.mean().item()
    
    def _expected_improvement(self, params_array: np.ndarray) -> np.ndarray:
        """2D Expected Improvement"""
        if params_array.ndim == 1:
            params_array = params_array.reshape(1, -1)
            
        mu, sigma = self.gp_model.predict(params_array, return_std=True)
        best_score = np.min(self.evaluated_scores)
        
        with np.errstate(divide='warn', invalid='warn'):
            z = (best_score - mu) / sigma
            ei = (best_score - mu) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
            ei[sigma == 0] = 0
            
        return ei
    
    def optimize(self, X1: torch.Tensor, X2: torch.Tensor, Y: torch.Tensor, 
                verbose: bool = True) -> Tuple[float, float, float]:
        """
        Run 2D Bayesian optimization
        
        Returns:
            (best_alpha, best_weight, best_score)
        """
        if verbose:
            print(f"2D Bayesian Optimization: α ∈ [{self.alpha_bounds[0]:.2e}, {self.alpha_bounds[1]:.2e}], "
                  f"w ∈ [{self.weight_bounds[0]}, {self.weight_bounds[1]}]")
        
        # Phase 1: Random initialization
        if verbose:
            print(f"\nPhase 1: Random initialization ({self.n_initial_points} points)")
            
        for i in range(self.n_initial_points):
            log_alpha = np.random.uniform(*self.log_alpha_bounds)
            weight = np.random.uniform(*self.weight_bounds)
            params = [log_alpha, weight]
            
            score = self._objective_function(params, X1, X2, Y)
            
            self.evaluated_params.append(params)
            self.evaluated_scores.append(score)
            
            if verbose:
                alpha = 10**log_alpha
                print(f"  {i+1}: α={alpha:.2e}, w={weight:.3f}, score={-score:.4f}")
        
        # Initialize 2D GP with anisotropic kernel
        kernel_alpha = Matern(length_scale=1.0, nu=2.5)  # For log-alpha dimension
        kernel_weight = Matern(length_scale=0.3, nu=2.5)  # For weight dimension
        kernel = kernel_alpha * kernel_weight  # Product kernel
        
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=10,
            normalize_y=True
        )
        
        # Phase 2: Bayesian optimization
        if verbose:
            print(f"\nPhase 2: BO iterations ({self.n_calls - self.n_initial_points})")
            
        for i in range(self.n_calls - self.n_initial_points):
            # Fit GP
            X_gp = np.array(self.evaluated_params)
            y_gp = np.array(self.evaluated_scores)
            self.gp_model.fit(X_gp, y_gp)
            
            # Optimize acquisition function
            def acquisition(params):
                return -self._expected_improvement(params.reshape(1, -1))[0]
            
            # Multi-start optimization of acquisition function
            best_acq_value = np.inf
            best_next_params = None
            
            for _ in range(10):  # Multiple random starts
                x0 = [
                    np.random.uniform(*self.log_alpha_bounds),
                    np.random.uniform(*self.weight_bounds)
                ]
                
                result = minimize(
                    acquisition,
                    x0=x0,
                    bounds=self.bounds,
                    method='L-BFGS-B'
                )
                
                if result.fun < best_acq_value:
                    best_acq_value = result.fun
                    best_next_params = result.x
            
            # Evaluate at best point
            score = self._objective_function(best_next_params, X1, X2, Y)
            
            self.evaluated_params.append(best_next_params.tolist())
            self.evaluated_scores.append(score)
            
            if verbose:
                alpha = 10**best_next_params[0]
                weight = best_next_params[1]
                best_so_far = -np.min(self.evaluated_scores)
                print(f"  {i+1}: α={alpha:.2e}, w={weight:.3f}, "
                      f"score={-score:.4f}, best={best_so_far:.4f}")
        
        # Extract best result
        best_idx = np.argmin(self.evaluated_scores)
        best_params = self.evaluated_params[best_idx]
        
        self.best_alpha_ = 10**best_params[0]
        self.best_weight_ = best_params[1]
        self.best_score_ = -self.evaluated_scores[best_idx]
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"Best α: {self.best_alpha_:.2e}")
            print(f"Best weight: {self.best_weight_:.4f}")
            print(f"Best score: {self.best_score_:.4f}")
        
        return self.best_alpha_, self.best_weight_, self.best_score_


# Example usage and testing
if __name__ == "__main__":
    print("Testing Bayesian Optimization for Ridge Regression")
    
    # Create synthetic data
    n_samples = 500
    n_features = 100
    n_targets = 50
    
    X = torch.randn(n_samples, n_features)
    Y = torch.randn(n_samples, n_targets)
    
    print(f"Data: {X.shape} features → {Y.shape} targets")
    
    # Test single parameter optimization
    print("\n" + "="*60)
    print("Testing Single Parameter (Alpha) Optimization")
    print("="*60)
    
    bo_single = BayesianOptimizationRidge(
        alpha_bounds=(1e-4, 1e2),
        n_calls=15,  # Reduced for quick test
        n_initial_points=5
    )
    
    best_alpha, best_score = bo_single.optimize(X, Y, verbose=True)
    
    # Test dual parameter optimization
    print("\n" + "="*60)
    print("Testing Dual Parameter (Alpha + Weight) Optimization")
    print("="*60)
    
    # Create two feature sets
    X1 = X[:, :50]  # Vision features
    X2 = X[:, 50:]  # Language features
    
    bo_dual = BayesianOptimizationConvexCombination(
        alpha_bounds=(1e-4, 1e2),
        weight_bounds=(0.0, 1.0),
        n_calls=20,  # Reduced for quick test
        n_initial_points=8
    )
    
    best_alpha, best_weight, best_score = bo_dual.optimize(X1, X2, Y, verbose=True)
    
    print("\nAll tests completed successfully!")