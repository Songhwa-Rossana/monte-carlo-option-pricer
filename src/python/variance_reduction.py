"""
Variance reduction techniques for Monte Carlo simulation.

Implements antithetic variates and control variate methods
to reduce the variance of Monte Carlo estimators. 
"""

from typing import Dict, Tuple
import numpy as np
from .options import Option, EuropeanOption
from .monte_carlo import MonteCarloEngine


class VarianceReducer:
    """
    Variance reduction techniques for Monte Carlo option pricing.
    
    Implements:
    - Antithetic variates: Use both Z and -Z for each random draw
    - Control variates: Use correlated asset with known value
    """
    
    @staticmethod
    def price_with_antithetic(
        engine: MonteCarloEngine,
        option: Option
    ) -> Tuple[float, float]:
        """
        Price option using antithetic variates. 
        
        Antithetic variates reduce variance by using paired random
        variables: for each Z ~ N(0,1), also use -Z. 
        
        Args:
            engine: Monte Carlo engine
            option: Option to price
        
        Returns:
            Tuple of (price, standard_error)
        """
        return engine.price_option(option, antithetic=True)
    
    @staticmethod
    def price_with_control_variate(
        engine: MonteCarloEngine,
        target_option: Option,
        control_option: Option
    ) -> Tuple[float, float]:
        """
        Price option using control variate method.
        
        Control variate reduces variance using a correlated estimator
        with known expectation:
            Y_adjusted = Y - c*(X - E[X])
        where c = Cov(X,Y) / Var(X)
        
        Args:
            engine: Monte Carlo engine
            target_option: Option to price (unknown value)
            control_option: Control option with known analytical price
        
        Returns:
            Tuple of (adjusted_price, adjusted_standard_error)
        """
        # Must have analytical price for control
        if not isinstance(control_option, EuropeanOption):
            raise ValueError("Control option must be EuropeanOption with analytical price")
        
        # Save original seed
        original_seed = engine.seed
        
        # Generate paths (use same seed for both)
        engine.set_seed(original_seed)
        paths = engine.generate_paths(
            S0=target_option.params. S0,
            r=target_option.params.r,
            sigma=target_option.params.sigma,
            T=target_option.params.T,
            antithetic=False
        )
        
        # Calculate target payoffs
        target_payoffs = target_option.payoff(paths)
        discount = np.exp(-target_option.params.r * target_option.params.T)
        target_discounted = target_payoffs * discount
        
        # Calculate control payoffs
        control_payoffs = control_option.payoff(paths)
        control_discounted = control_payoffs * discount
        
        # Known analytical value of control
        control_analytical = control_option.analytical_price()
        
        # Calculate optimal control coefficient
        # c = Cov(X, Y) / Var(X)
        cov = np.cov(control_discounted, target_discounted)[0, 1]
        var = np.var(control_discounted, ddof=1)
        
        if var > 1e-10:  # Avoid division by zero
            c = cov / var
        else:
            c = 0
        
        # Apply control variate adjustment
        # Y_adjusted = Y - c*(X - E[X])
        adjusted_payoffs = target_discounted - c * (control_discounted - control_analytical)
        
        # Calculate adjusted price and standard error
        adjusted_price = np.mean(adjusted_payoffs)
        adjusted_std = np.std(adjusted_payoffs, ddof=1)
        adjusted_se = 1.96 * adjusted_std / np.sqrt(len(adjusted_payoffs))
        
        return adjusted_price, adjusted_se
    
    @staticmethod
    def compare_methods(
        option: Option,
        n_paths: int,
        seed: int = 42
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare different pricing methods for the same option.
        
        Args:
            option: Option to price
            n_paths: Number of paths for simulation
            seed: Random seed
        
        Returns:
            Dictionary with results from different methods:
            - 'standard': Standard Monte Carlo
            - 'antithetic': With antithetic variates
            - 'analytical': Analytical price (if available)
        """
        # Create engine
        engine = MonteCarloEngine(n_paths=n_paths, n_steps=100, seed=seed)
        
        results = {}
        
        # Standard Monte Carlo
        engine.set_seed(seed)
        price_std, se_std = engine.price_option(option, antithetic=False)
        results['standard'] = {
            'price': price_std,
            'std_error': se_std,
            'method': 'Standard MC'
        }
        
        # Antithetic variates
        engine.set_seed(seed)
        price_anti, se_anti = engine.price_option(option, antithetic=True)
        results['antithetic'] = {
            'price': price_anti,
            'std_error': se_anti,
            'method': 'Antithetic Variates'
        }
        
        # Analytical if available
        if isinstance(option, EuropeanOption):
            analytical_price = option.analytical_price()
            results['analytical'] = {
                'price': analytical_price,
                'std_error': 0.0,
                'method': 'Black-Scholes'
            }
        
        return results
    
    @staticmethod
    def variance_reduction_ratio(
        std_error_standard: float,
        std_error_reduced: float
    ) -> float:
        """
        Calculate variance reduction ratio.
        
        Variance reduction ratio = Var(standard) / Var(reduced)
        Since Var = SE^2 (up to constant), ratio = (SE_std / SE_red)^2
        
        Args:
            std_error_standard: Standard error from standard MC
            std_error_reduced: Standard error from variance-reduced MC
        
        Returns:
            Variance reduction ratio (> 1 means improvement)
        """
        if std_error_reduced == 0:
            return float('inf')
        
        ratio = (std_error_standard / std_error_reduced) ** 2
        return ratio
