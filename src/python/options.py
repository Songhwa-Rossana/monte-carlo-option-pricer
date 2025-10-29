from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
import scipy.stats as norm

class OptionType(Enum):
    """Enumm for option types"""
    CALL = "call"
    PUT = "put"


class BarrierType(Enum):
    """Enum for barrier types"""
    UP_AND_IN = "up_and_in"
    UP_AND_OUT = "up_and_out"
    DOWN_AND_IN = "down_and_in"
    DOWN_AND_OUT = "down_and_out"

@dataclass
class OptionParameters:
    """
    Parameters defining an option contract.
    
    Attributes:
        S0: Initial stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: CALL or PUT
    """
    S0: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: OptionType
    
    def __post_init__(self):
        """Validate Parameter"""
        if self.S0 <= 0:
            raise ValueError("Initial stock price S0 must be positive.")
        if self.K <= 0:
            raise ValueError("Strike price K must be positive.")
        if self.T <= 0:
            raise ValueError("Time to maturity T must be positive.")
        if self.sigma <= 0:
            raise ValueError("Volatility must be positive")
        
class Option(ABC):
    """Abstract base class for options."""

    def __init__(self, params: OptionParameters):
        """
        Initialize option with parameters.

        Args:
            params: OptionParameters Object
        """

        self.params = params

    @abstractmethod
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Calculate option payoff for given price paths.
        
        Args:
            paths: Array of stock price paths, shape (n_paths, n_steps+1)
            
        Returns:
            Array of payoffs, shape (n_paths,)
        """
        pass

    def analytical_price(self) -> Optional[float]:
        """
        Calculate analytical price if available.

        Returns:
            Analytical price or None if not available.
        """
        S0 = self.params.S0
        K = self.params.K
        T = self.params.T
        r = self.params.r
        sigma = self.params.sigma

        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if self.params.option_type == OptionType.CALL:
            price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else: # PUT
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        return price
    

class AsianOption(Option):
    """Asian option (payoff depends on average price over the path)."""

    def __init__(self, params: OptionParameters, averaging_type: str = "arithmetic"):
        """
        Initialize Asian option.
        
        Args:
            params: OptionParameters object
            averaging_type: "arithmetic" or "geometric" averaging
        """
        super().__init__(params)
        if averaging_type not in ["arithmetic", "geometric"]:
            raise ValueError("averaging_type must be 'arithmetic' or 'geometric'")
        self.averaging_type = averaging_type

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Calculate Asian option payoff based on average price.
        
        Args:
            paths: Array of stock price paths, shape (n_paths, n_steps+1)
            
        Returns:
            Array of payoffs based on average prices
        """
        if self.averaging_type == "arithmetic":
            # Arithmetic average
            avg_prices = np.mean(paths, axis=1)
        else:
            # Geometric average
            avg_price = np.exp(np.mean(np.log(paths), axis = 1))
        
        if self.params.option_type == OptionType.CALL:
            return np.maximum(avg_prices - self.params.K, 0)
        else: #PUT
            return np.maximum(self.params.K - avg_prices, 0)
        

class BarrierOption(Option):
    """Barrier option (activated or deactivated when price crosses barrier)."""

    def __init__(self, params: OptionParameters, barrier: float,
                 barrier_type: BarrierType):
        """
        Initialize Barrier option.
        
        Args:
            params: OptionParameters object
            barrier: Barrier price level
            barrier_type: Type of barrier (knock-in/knock-out, up/down)
        """
        super().__init__(params)
        self.barrier = barrier
        self.barrierType = barrier_type


        # Validate barrier level
        if barrier_type in [BarrierType.UP_AND_IN, BarrierType.UP_AND_OUT]:
            if barrier <= params.S0:
                raise ValueError("Up barrier must be above initial price")
            else: # DOWN barriers
                if barrier >= params.S0:
                    raise ValueError("Down barrier must be below initial price")
                
    def payoff(self, paths:np.ndarray) -> np.ndarray:
        """
        Calculate Barrier option payoff.
        
        Args:
            paths: Array of stock price paths, shape (n_paths, n_steps+1)
            
        Returns:
            Array of payoffs considering barrier conditions
        """
        # Calculate European payoff first
        S_T = paths[:, -1]
        if self.params.option_type == OptionType.CALL:
            european_payoff = np.maximum(S_T - self.params.K, 0)
        else: # PUT
            european_payoff = np.maximum(self.params.K - S_T, 0)

        # Check if barrier was crossed
        if self.barrier_type in [BarrierType.UP_AND_IN, BarrierType.UP_AND_OUT]:
            # Check if any price crossed above the barrier
            barrier_crossed = np.any(paths >= self.barrier, axis = 1)
        else: # DOWN barriers
            # Check if any price crossed below the barrier
            barrier_crossed = np.any(paths <= self.barrier, axis = 1)

        # Apply barrier logic
        if self.barrier_type in [BarrierType.UP_AND_IN, BarrierType.UP_AND_OUT]:
            # Knock-in: option only active if barrier crossed
            payoff = np.where(barrier_crossed, european_payoff, 0)
        else: # Knock-out: option inactive if barrier crossed
            payoff = np.where(barrier_crossed, 0, european_payoff)

        return payoff

        






