"""
Option classes and parameters for Monte Carlo pricing. 

Implements European, Asian, and Barrier options with analytical
Black-Scholes pricing and Greeks for European options.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
from . utils import norm_cdf, norm_pdf


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


class BarrierType(Enum):
    """Barrier option type enumeration."""
    UP_AND_IN = "up_and_in"
    UP_AND_OUT = "up_and_out"
    DOWN_AND_IN = "down_and_in"
    DOWN_AND_OUT = "down_and_out"


@dataclass
class OptionParameters:
    """
    Parameters for option pricing.
    
    Attributes:
        S0: Initial stock price (must be positive)
        K: Strike price (must be positive)
        T: Time to maturity in years (must be positive)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized, must be positive)
        option_type: Type of option (CALL or PUT)
    """
    S0: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: OptionType
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.S0 <= 0:
            raise ValueError(f"Initial stock price must be positive, got {self.S0}")
        if self.K <= 0:
            raise ValueError(f"Strike price must be positive, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"Time to maturity must be positive, got {self.T}")
        if self. sigma <= 0:
            raise ValueError(f"Volatility must be positive, got {self.sigma}")
        if not isinstance(self.option_type, OptionType):
            raise ValueError(f"option_type must be an OptionType enum, got {type(self.option_type)}")


class Option(ABC):
    """
    Abstract base class for all option types.
    
    Attributes:
        params: OptionParameters containing option specifications
    """
    
    def __init__(self, params: OptionParameters):
        """
        Initialize option with parameters.
        
        Args:
            params: OptionParameters object with validated parameters
        """
        self. params = params
    
    @abstractmethod
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Calculate option payoff from price paths.
        
        Args:
            paths: Array of simulated price paths, shape (n_paths, n_steps+1)
        
        Returns:
            Array of payoffs for each path, shape (n_paths,)
        """
        pass


class EuropeanOption(Option):
    """
    European option with analytical Black-Scholes pricing. 
    
    A European option can only be exercised at maturity.
    """
    
    def payoff(self, paths: np. ndarray) -> np.ndarray:
        """
        Calculate European option payoff.
        
        Args:
            paths: Price paths, shape (n_paths, n_steps+1)
        
        Returns:
            Payoffs based on final prices, shape (n_paths,)
        """
        # European option uses only the final price
        final_prices = paths[:, -1]
        
        if self.params.option_type == OptionType.CALL:
            return np.maximum(final_prices - self.params.K, 0)
        else:  # PUT
            return np.maximum(self. params.K - final_prices, 0)
    
    def _calculate_d1_d2(self) -> tuple[float, float]:
        """
        Calculate d1 and d2 for Black-Scholes formula.
        
        Returns:
            Tuple of (d1, d2)
        """
        S0 = self.params.S0
        K = self.params.K
        T = self.params. T
        r = self.params.r
        sigma = self. params.sigma
        
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np. sqrt(T)
        
        return d1, d2
    
    def analytical_price(self) -> float:
        """
        Calculate analytical Black-Scholes price.
        
        Returns:
            Theoretical option price
        """
        d1, d2 = self._calculate_d1_d2()
        
        S0 = self.params.S0
        K = self.params.K
        T = self. params.T
        r = self.params.r
        
        if self.params.option_type == OptionType.CALL:
            price = S0 * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
        else:  # PUT
            price = K * np. exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1)
        
        return price
    
    def black_scholes_delta(self) -> float:
        """
        Calculate analytical delta. 
        
        Returns:
            Delta (∂V/∂S)
        """
        d1, _ = self._calculate_d1_d2()
        
        if self.params.option_type == OptionType.CALL:
            return norm_cdf(d1)
        else:  # PUT
            return norm_cdf(d1) - 1
    
    def black_scholes_gamma(self) -> float:
        """
        Calculate analytical gamma. 
        
        Returns:
            Gamma (∂²V/∂S²)
        """
        d1, _ = self._calculate_d1_d2()
        
        S0 = self.params.S0
        sigma = self.params.sigma
        T = self.params.T
        
        gamma = norm_pdf(d1) / (S0 * sigma * np.sqrt(T))
        return gamma
    
    def black_scholes_vega(self) -> float:
        """
        Calculate analytical vega.
        
        Returns:
            Vega (∂V/∂σ) per 1% change in volatility
        """
        d1, _ = self._calculate_d1_d2()
        
        S0 = self.params.S0
        T = self.params.T
        
        vega = S0 * norm_pdf(d1) * np.sqrt(T) / 100
        return vega
    
    def black_scholes_theta(self) -> float:
        """
        Calculate analytical theta. 
        
        Returns:
            Theta (∂V/∂t) per day
        """
        d1, d2 = self._calculate_d1_d2()
        
        S0 = self.params.S0
        K = self.params.K
        T = self.params.T
        r = self.params.r
        sigma = self.params.sigma
        
        if self.params.option_type == OptionType.CALL:
            theta = (
                -S0 * norm_pdf(d1) * sigma / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm_cdf(d2)
            )
        else:  # PUT
            theta = (
                -S0 * norm_pdf(d1) * sigma / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm_cdf(-d2)
            )
        
        # Convert to per-day theta
        return theta / 365
    
    def black_scholes_rho(self) -> float:
        """
        Calculate analytical rho. 
        
        Returns:
            Rho (∂V/∂r) per 1% change in interest rate
        """
        _, d2 = self._calculate_d1_d2()
        
        K = self.params.K
        T = self.params.T
        r = self.params.r
        
        if self.params.option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * norm_cdf(d2) / 100
        else:  # PUT
            rho = -K * T * np.exp(-r * T) * norm_cdf(-d2) / 100
        
        return rho


class AsianOption(Option):
    """
    Asian option with path-dependent payoff.
    
    Asian options depend on the average price over the life of the option.
    Supports both arithmetic and geometric averaging.
    """
    
    def __init__(self, params: OptionParameters, averaging_type: str = "arithmetic"):
        """
        Initialize Asian option. 
        
        Args:
            params: Option parameters
            averaging_type: "arithmetic" or "geometric" averaging
        """
        super().__init__(params)
        if averaging_type not in ["arithmetic", "geometric"]:
            raise ValueError(f"averaging_type must be 'arithmetic' or 'geometric', got {averaging_type}")
        self.averaging_type = averaging_type
    
    def payoff(self, paths: np. ndarray) -> np.ndarray:
        """
        Calculate Asian option payoff based on average price.
        
        Args:
            paths: Price paths, shape (n_paths, n_steps+1)
        
        Returns:
            Payoffs based on average prices, shape (n_paths,)
        """
        if self.averaging_type == "arithmetic":
            avg_prices = np.mean(paths, axis=1)
        else:  # geometric
            # Geometric mean: exp(mean(log(x)))
            avg_prices = np.exp(np.mean(np.log(paths), axis=1))
        
        if self. params.option_type == OptionType.CALL:
            return np.maximum(avg_prices - self.params.K, 0)
        else:  # PUT
            return np.maximum(self.params.K - avg_prices, 0)


class BarrierOption(Option):
    """
    Barrier option with knock-in or knock-out features.
    
    Barrier options activate (knock-in) or deactivate (knock-out)
    when the underlying price crosses a barrier level.
    """
    
    def __init__(
        self,
        params: OptionParameters,
        barrier_type: BarrierType,
        barrier_level: float
    ):
        """
        Initialize barrier option.
        
        Args:
            params: Option parameters
            barrier_type: Type of barrier (UP_IN, UP_OUT, DOWN_IN, DOWN_OUT)
            barrier_level: Price level that triggers barrier condition
        """
        super().__init__(params)
        if not isinstance(barrier_type, BarrierType):
            raise ValueError(f"barrier_type must be a BarrierType enum, got {type(barrier_type)}")
        if barrier_level <= 0:
            raise ValueError(f"Barrier level must be positive, got {barrier_level}")
        
        self.barrier_type = barrier_type
        self.barrier_level = barrier_level
    
    def _check_barrier_crossed(self, paths: np.ndarray) -> np.ndarray:
        """
        Check if barrier was crossed for each path.
        
        Args:
            paths: Price paths, shape (n_paths, n_steps+1)
        
        Returns:
            Boolean array indicating barrier crossing, shape (n_paths,)
        """
        if self.barrier_type in [BarrierType.UP_AND_IN, BarrierType.UP_AND_OUT]:
            # Check if price ever went above barrier
            crossed = np.any(paths >= self.barrier_level, axis=1)
        else:  # DOWN_AND_IN or DOWN_AND_OUT
            # Check if price ever went below barrier
            crossed = np.any(paths <= self.barrier_level, axis=1)
        
        return crossed
    
    def payoff(self, paths: np. ndarray) -> np.ndarray:
        """
        Calculate barrier option payoff.
        
        Args:
            paths: Price paths, shape (n_paths, n_steps+1)
        
        Returns:
            Payoffs considering barrier conditions, shape (n_paths,)
        """
        # Calculate vanilla European payoff
        final_prices = paths[:, -1]
        if self.params.option_type == OptionType.CALL:
            vanilla_payoff = np.maximum(final_prices - self.params. K, 0)
        else:  # PUT
            vanilla_payoff = np.maximum(self.params.K - final_prices, 0)
        
        # Check barrier crossing
        barrier_crossed = self._check_barrier_crossed(paths)
        
        # Apply barrier logic
        if self.barrier_type in [BarrierType.UP_AND_IN, BarrierType.DOWN_AND_IN]:
            # Knock-in: payoff only if barrier was crossed
            payoff = np.where(barrier_crossed, vanilla_payoff, 0)
        else:  # UP_AND_OUT or DOWN_AND_OUT
            # Knock-out: payoff only if barrier was NOT crossed
            payoff = np. where(barrier_crossed, 0, vanilla_payoff)
        
        return payoff
