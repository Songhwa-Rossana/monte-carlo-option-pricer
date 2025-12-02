"""
Greeks calculation using finite difference methods.

Implements numerical approximations for option sensitivities
when analytical formulas are not available.
"""

from typing import Dict
import numpy as np
from .options import Option, OptionParameters
from .monte_carlo import MonteCarloEngine


class GreeksCalculator:
    """
    Calculate option Greeks using finite difference methods.
    
    Greeks measure the sensitivity of option prices to various parameters.
    Finite difference methods approximate derivatives numerically.
    """
    
    def __init__(self, engine: MonteCarloEngine, bump_size: float = 0.01):
        """
        Initialize Greeks calculator.
        
        Args:
            engine: Monte Carlo engine for pricing
            bump_size: Relative size of parameter bumps (default 1%)
        """
        if bump_size <= 0:
            raise ValueError(f"bump_size must be positive, got {bump_size}")
        
        self.engine = engine
        self.bump_size = bump_size
    
    def calculate_delta(self, option: Option) -> float:
        """
        Calculate delta using finite difference. 
        
        Delta = ∂V/∂S ≈ (V(S+ΔS) - V(S-ΔS)) / (2*ΔS)
        
        Args:
            option: Option to calculate delta for
        
        Returns:
            Delta (sensitivity to stock price)
        """
        # Bump stock price up and down
        dS = option.params.S0 * self.bump_size
        
        # Price with bumped up S0
        params_up = OptionParameters(
            S0=option.params.S0 + dS,
            K=option.params.K,
            T=option.params.T,
            r=option.params.r,
            sigma=option.params.sigma,
            option_type=option.params.option_type
        )
        option_up = type(option)(params_up)
        if hasattr(option, 'averaging_type'):
            option_up.averaging_type = option.averaging_type
        if hasattr(option, 'barrier_type'):
            option_up.barrier_type = option.barrier_type
            option_up.barrier_level = option.barrier_level
        
        # Price with bumped down S0
        params_down = OptionParameters(
            S0=option.params.S0 - dS,
            K=option. params.K,
            T=option.params.T,
            r=option.params.r,
            sigma=option.params. sigma,
            option_type=option.params.option_type
        )
        option_down = type(option)(params_down)
        if hasattr(option, 'averaging_type'):
            option_down.averaging_type = option.averaging_type
        if hasattr(option, 'barrier_type'):
            option_down.barrier_type = option.barrier_type
            option_down. barrier_level = option.barrier_level
        
        # Calculate prices
        price_up, _ = self.engine.price_option(option_up)
        price_down, _ = self.engine.price_option(option_down)
        
        # Central difference
        delta = (price_up - price_down) / (2 * dS)
        
        return delta
    
    def calculate_gamma(self, option: Option) -> float:
        """
        Calculate gamma using second-order finite difference.
        
        Gamma = ∂²V/∂S² ≈ (V(S+ΔS) - 2*V(S) + V(S-ΔS)) / (ΔS)²
        
        Args:
            option: Option to calculate gamma for
        
        Returns:
            Gamma (sensitivity of delta to stock price)
        """
        # Bump stock price up and down
        dS = option.params.S0 * self.bump_size
        
        # Price with bumped up S0
        params_up = OptionParameters(
            S0=option.params.S0 + dS,
            K=option. params.K,
            T=option.params.T,
            r=option.params.r,
            sigma=option.params. sigma,
            option_type=option.params.option_type
        )
        option_up = type(option)(params_up)
        if hasattr(option, 'averaging_type'):
            option_up.averaging_type = option.averaging_type
        if hasattr(option, 'barrier_type'):
            option_up.barrier_type = option.barrier_type
            option_up. barrier_level = option.barrier_level
        
        # Price with bumped down S0
        params_down = OptionParameters(
            S0=option.params. S0 - dS,
            K=option.params.K,
            T=option.params.T,
            r=option.params.r,
            sigma=option.params.sigma,
            option_type=option.params.option_type
        )
        option_down = type(option)(params_down)
        if hasattr(option, 'averaging_type'):
            option_down.averaging_type = option.averaging_type
        if hasattr(option, 'barrier_type'):
            option_down.barrier_type = option.barrier_type
            option_down.barrier_level = option.barrier_level
        
        # Original price
        price, _ = self.engine.price_option(option)
        
        # Calculate bumped prices
        price_up, _ = self.engine.price_option(option_up)
        price_down, _ = self.engine. price_option(option_down)
        
        # Second-order central difference
        gamma = (price_up - 2 * price + price_down) / (dS ** 2)
        
        return gamma
    
    def calculate_vega(self, option: Option) -> float:
        """
        Calculate vega using finite difference.
        
        Vega = ∂V/∂σ per 1% change in volatility
        
        Args:
            option: Option to calculate vega for
        
        Returns:
            Vega (sensitivity to volatility per 1%)
        """
        # Bump volatility by bump_size
        d_sigma = option.params.sigma * self.bump_size
        
        # Price with bumped up sigma
        params_up = OptionParameters(
            S0=option.params.S0,
            K=option.params. K,
            T=option. params.T,
            r=option.params.r,
            sigma=option.params.sigma + d_sigma,
            option_type=option.params.option_type
        )
        option_up = type(option)(params_up)
        if hasattr(option, 'averaging_type'):
            option_up.averaging_type = option.averaging_type
        if hasattr(option, 'barrier_type'):
            option_up. barrier_type = option.barrier_type
            option_up.barrier_level = option.barrier_level
        
        # Price with bumped down sigma
        params_down = OptionParameters(
            S0=option.params.S0,
            K=option.params.K,
            T=option.params.T,
            r=option.params.r,
            sigma=option.params.sigma - d_sigma,
            option_type=option.params.option_type
        )
        option_down = type(option)(params_down)
        if hasattr(option, 'averaging_type'):
            option_down.averaging_type = option.averaging_type
        if hasattr(option, 'barrier_type'):
            option_down.barrier_type = option.barrier_type
            option_down.barrier_level = option.barrier_level
        
        # Calculate prices
        price_up, _ = self.engine.price_option(option_up)
        price_down, _ = self.engine.price_option(option_down)
        
        # Central difference, scaled to per-1% change
        vega = (price_up - price_down) / (2 * d_sigma) * (option.params.sigma / 100)
        
        return vega
    
    def calculate_theta(self, option: Option) -> float:
        """
        Calculate theta using finite difference.
        
        Theta = ∂V/∂t per day (note: negative time derivative)
        
        Args:
            option: Option to calculate theta for
        
        Returns:
            Theta (sensitivity to time decay per day)
        """
        # Bump time by 1 day
        dt = 1 / 365
        
        # Can't bump below zero time
        if option.params.T <= dt:
            # Use forward difference
            params_down = OptionParameters(
                S0=option.params.S0,
                K=option.params.K,
                T=max(option.params.T - dt, 0. 001),  # Avoid zero
                r=option.params. r,
                sigma=option. params.sigma,
                option_type=option.params.option_type
            )
            option_down = type(option)(params_down)
            if hasattr(option, 'averaging_type'):
                option_down.averaging_type = option.averaging_type
            if hasattr(option, 'barrier_type'):
                option_down.barrier_type = option.barrier_type
                option_down.barrier_level = option. barrier_level
            
            price, _ = self.engine.price_option(option)
            price_down, _ = self.engine.price_option(option_down)
            
            theta = (price_down - price) / dt
        else:
            # Use central difference
            params_up = OptionParameters(
                S0=option.params.S0,
                K=option.params.K,
                T=option.params.T + dt,
                r=option. params.r,
                sigma=option.params.sigma,
                option_type=option.params. option_type
            )
            option_up = type(option)(params_up)
            if hasattr(option, 'averaging_type'):
                option_up. averaging_type = option.averaging_type
            if hasattr(option, 'barrier_type'):
                option_up.barrier_type = option.barrier_type
                option_up.barrier_level = option.barrier_level
            
            params_down = OptionParameters(
                S0=option.params. S0,
                K=option.params.K,
                T=option.params.T - dt,
                r=option.params.r,
                sigma=option.params.sigma,
                option_type=option.params.option_type
            )
            option_down = type(option)(params_down)
            if hasattr(option, 'averaging_type'):
                option_down.averaging_type = option.averaging_type
            if hasattr(option, 'barrier_type'):
                option_down.barrier_type = option.barrier_type
                option_down.barrier_level = option.barrier_level
            
            price_up, _ = self. engine.price_option(option_up)
            price_down, _ = self.engine.price_option(option_down)
            
            # Note: theta is -∂V/∂T, so we flip the sign
            theta = -(price_up - price_down) / (2 * dt)
        
        return theta
    
    def calculate_rho(self, option: Option) -> float:
        """
        Calculate rho using finite difference.
        
        Rho = ∂V/∂r per 1% change in interest rate
        
        Args:
            option: Option to calculate rho for
        
        Returns:
            Rho (sensitivity to interest rate per 1%)
        """
        # Bump interest rate by bump_size (absolute, not relative)
        dr = 0.01 * self.bump_size  # 1 basis point * bump_size
        
        # Price with bumped up r
        params_up = OptionParameters(
            S0=option.params. S0,
            K=option.params.K,
            T=option.params.T,
            r=option.params.r + dr,
            sigma=option.params.sigma,
            option_type=option.params.option_type
        )
        option_up = type(option)(params_up)
        if hasattr(option, 'averaging_type'):
            option_up.averaging_type = option.averaging_type
        if hasattr(option, 'barrier_type'):
            option_up.barrier_type = option.barrier_type
            option_up.barrier_level = option.barrier_level
        
        # Price with bumped down r
        params_down = OptionParameters(
            S0=option. params.S0,
            K=option.params.K,
            T=option.params.T,
            r=option.params.r - dr,
            sigma=option.params.sigma,
            option_type=option.params.option_type
        )
        option_down = type(option)(params_down)
        if hasattr(option, 'averaging_type'):
            option_down.averaging_type = option.averaging_type
        if hasattr(option, 'barrier_type'):
            option_down.barrier_type = option.barrier_type
            option_down.barrier_level = option.barrier_level
        
        # Calculate prices
        price_up, _ = self.engine.price_option(option_up)
        price_down, _ = self.engine.price_option(option_down)
        
        # Central difference, scaled to per-1% change
        rho = (price_up - price_down) / (2 * dr) * 0.01
        
        return rho
    
    def calculate_all_greeks(self, option: Option) -> Dict[str, float]:
        """
        Calculate all Greeks for an option.
        
        Args:
            option: Option to calculate Greeks for
        
        Returns:
            Dictionary with keys: 'delta', 'gamma', 'vega', 'theta', 'rho'
        """
        return {
            'delta': self. calculate_delta(option),
            'gamma': self.calculate_gamma(option),
            'vega': self.calculate_vega(option),
            'theta': self.calculate_theta(option),
            'rho': self.calculate_rho(option),
        }
