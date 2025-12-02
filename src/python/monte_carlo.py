"""
Monte Carlo simulation engine for option pricing.

Implements geometric Brownian motion path generation and
option pricing with variance reduction techniques.
"""

from typing import Optional, Tuple
import numpy as np
from .options import Option


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for pricing options.
    
    Uses geometric Brownian motion to simulate stock price paths
    and prices options by averaging discounted payoffs.
    """
    
    def __init__(self, n_paths: int, n_steps: int, seed: int = 42):
        """
        Initialize Monte Carlo engine.
        
        Args:
            n_paths: Number of simulation paths
            n_steps: Number of time steps per path
            seed: Random seed for reproducibility
        """
        if n_paths <= 0:
            raise ValueError(f"n_paths must be positive, got {n_paths}")
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        
        self._n_paths = n_paths
        self._n_steps = n_steps
        self._seed = seed
        self._rng = np.random.default_rng(seed)
    
    @property
    def n_paths(self) -> int:
        """Get number of simulation paths."""
        return self._n_paths
    
    @property
    def n_steps(self) -> int:
        """Get number of time steps."""
        return self._n_steps
    
    @property
    def seed(self) -> int:
        """Get random seed."""
        return self._seed
    
    def set_n_paths(self, n_paths: int) -> None:
        """Set number of simulation paths."""
        if n_paths <= 0:
            raise ValueError(f"n_paths must be positive, got {n_paths}")
        self._n_paths = n_paths
    
    def set_n_steps(self, n_steps: int) -> None:
        """Set number of time steps."""
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        self._n_steps = n_steps
    
    def set_seed(self, seed: int) -> None:
        """Set random seed and reset RNG."""
        self._seed = seed
        self._rng = np.random.default_rng(seed)
    
    def generate_paths(
        self,
        S0: float,
        r: float,
        sigma: float,
        T: float,
        antithetic: bool = False
    ) -> np.ndarray:
        """
        Generate stock price paths using geometric Brownian motion.
        
        The stochastic differential equation:
            dS = r*S*dt + sigma*S*dW
        
        Solution:
            S(t+dt) = S(t) * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        
        Args:
            S0: Initial stock price
            r: Risk-free rate (annualized)
            sigma: Volatility (annualized)
            T: Time to maturity in years
            antithetic: If True, use antithetic variates for variance reduction
        
        Returns:
            Array of price paths, shape (n_paths, n_steps+1)
            First column is S0 for all paths
        """
        dt = T / self._n_steps
        
        if antithetic:
            # Generate half the paths, then use antithetic variates for the other half
            n_half = self._n_paths // 2
            n_actual = n_half * 2  # Ensure even number
            
            # Generate random normal samples for half the paths
            Z = self._rng.standard_normal((n_half, self._n_steps))
            
            # Create antithetic pairs: [Z, -Z]
            Z_full = np.vstack([Z, -Z])
        else:
            # Standard random number generation
            Z_full = self._rng.standard_normal((self._n_paths, self._n_steps))
            n_actual = self._n_paths
        
        # Initialize paths array with S0
        paths = np.zeros((n_actual, self._n_steps + 1))
        paths[:, 0] = S0
        
        # Calculate drift and diffusion components
        drift = (r - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np. sqrt(dt)
        
        # Generate paths using cumulative product
        # log(S(t+dt)/S(t)) = drift + diffusion * Z
        log_returns = drift + diffusion * Z_full
        
        # Cumulative sum of log returns gives log(S(t)/S0)
        cumulative_returns = np.cumsum(log_returns, axis=1)
        
        # S(t) = S0 * exp(cumulative_returns)
        paths[:, 1:] = S0 * np. exp(cumulative_returns)
        
        return paths
    
    def price_option(
        self,
        option: Option,
        antithetic: bool = False
    ) -> Tuple[float, float]:
        """
        Price an option using Monte Carlo simulation. 
        
        Args:
            option: Option object to price
            antithetic: If True, use antithetic variates
        
        Returns:
            Tuple of (price, standard_error)
            - price: Estimated option price
            - standard_error: Standard error of the estimate (95% CI width)
        """
        # Generate price paths
        paths = self.generate_paths(
            S0=option.params.S0,
            r=option.params.r,
            sigma=option.params.sigma,
            T=option.params.T,
            antithetic=antithetic
        )
        
        # Calculate payoffs
        payoffs = option. payoff(paths)
        
        # Discount to present value
        discount_factor = np.exp(-option.params. r * option.params.T)
        discounted_payoffs = payoffs * discount_factor
        
        # Calculate price as mean of discounted payoffs
        price = np.mean(discounted_payoffs)
        
        # Calculate standard error for 95% confidence interval
        # SE = std / sqrt(n), 95% CI ≈ 1.96 * SE
        std_dev = np.std(discounted_payoffs, ddof=1)
        n_actual = len(discounted_payoffs)
        standard_error = 1.96 * std_dev / np.sqrt(n_actual)
        
        return price, standard_error
