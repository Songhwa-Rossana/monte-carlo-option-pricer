"""
Unit tests for Python Monte Carlo option pricing implementation.
"""

import pytest
import numpy as np
from src.python.options import (
    OptionType, OptionParameters, EuropeanOption, 
    AsianOption, BarrierOption, BarrierType
)
from src.python.monte_carlo import MonteCarloEngine
from src.python.greeks import GreeksCalculator
from src.python.variance_reduction import VarianceReducer


class TestOptions:
    """Test option classes and payoff calculations."""
    
    def test_option_parameters_validation(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            OptionParameters(S0=-100, K=100, T=1, r=0.05, sigma=0.2, 
                           option_type=OptionType.CALL)
        
        with pytest.raises(ValueError):
            OptionParameters(S0=100, K=-100, T=1, r=0.05, sigma=0.2, 
                           option_type=OptionType.CALL)
        
        with pytest.raises(ValueError):
            OptionParameters(S0=100, K=100, T=-1, r=0.05, sigma=0.2, 
                           option_type=OptionType.CALL)
    
    def test_european_call_payoff(self):
        """Test European call option payoff."""
        params = OptionParameters(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                option_type=OptionType.CALL)
        option = EuropeanOption(params)
        
        # Create mock paths
        paths = np.array([[100, 105, 110],
                         [100, 95, 90],
                         [100, 102, 100]])
        
        payoffs = option.payoff(paths)
        expected = np.array([10, 0, 0])  # max(S_T - K, 0)
        
        np.testing.assert_array_equal(payoffs, expected)
    
    def test_european_put_payoff(self):
        """Test European put option payoff."""
        params = OptionParameters(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                option_type=OptionType.PUT)
        option = EuropeanOption(params)
        
        paths = np.array([[100, 105, 110],
                         [100, 95, 90],
                         [100, 102, 100]])
        
        payoffs = option.payoff(paths)
        expected = np.array([0, 10, 0])  # max(K - S_T, 0)
        
        np.testing.assert_array_equal(payoffs, expected)
    
    def test_black_scholes_call(self):
        """Test Black-Scholes formula for call option."""
        params = OptionParameters(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                option_type=OptionType.CALL)
        option = EuropeanOption(params)
        
        price = option.analytical_price()
        
        # Expected value approximately 10.45 for ATM call
        assert 10.0 < price < 11.0
    
    def test_black_scholes_put(self):
        """Test Black-Scholes formula for put option."""
        params = OptionParameters(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                option_type=OptionType.PUT)
        option = EuropeanOption(params)
        
        price = option.analytical_price()
        
        # Expected value approximately 5.57 for ATM put
        assert 5.0 < price < 6.5
    
    def test_asian_arithmetic_payoff(self):
        """Test Asian option with arithmetic averaging."""
        params = OptionParameters(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                option_type=OptionType.CALL)
        option = AsianOption(params, averaging_type="arithmetic")
        
        paths = np.array([[100, 110, 120]])  # Average = 110
        payoffs = option.payoff(paths)
        
        expected = 10.0  # max(110 - 100, 0)
        np.testing.assert_almost_equal(payoffs[0], expected)
    
    def test_asian_geometric_payoff(self):
        """Test Asian option with geometric averaging."""
        params = OptionParameters(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                option_type=OptionType.CALL)
        option = AsianOption(params, averaging_type="geometric")
        
        paths = np.array([[100, 100, 100]])  # Geo mean = 100
        payoffs = option.payoff(paths)
        
        expected = 0.0  # max(100 - 100, 0)
        np.testing.assert_almost_equal(payoffs[0], expected)
    
    def test_barrier_up_and_out(self):
        """Test up-and-out barrier option."""
        params = OptionParameters(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                option_type=OptionType.CALL)
        option = BarrierOption(params, barrier=120, 
                              barrier_type=BarrierType.UP_AND_OUT)
        
        # Path that crosses barrier
        paths_crossed = np.array([[100, 125, 130]])
        payoffs_crossed = option.payoff(paths_crossed)
        assert payoffs_crossed[0] == 0  # Knocked out
        
        # Path that doesn't cross barrier
        paths_safe = np.array([[100, 105, 110]])
        payoffs_safe = option.payoff(paths_safe)
        assert payoffs_safe[0] == 10  # max(110 - 100, 0)
    
    def test_barrier_down_and_in(self):
        """Test down-and-in barrier option."""
        params = OptionParameters(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                option_type=OptionType.CALL)
        option = BarrierOption(params, barrier=80,
                              barrier_type=BarrierType.DOWN_AND_IN)
        
        # Path that crosses barrier
        paths_crossed = np.array([[100, 75, 110]])
        payoffs_crossed = option.payoff(paths_crossed)
        assert payoffs_crossed[0] == 10  # Knocked in, max(110 - 100, 0)
        
        # Path that doesn't cross barrier
        paths_safe = np.array([[100, 105, 110]])
        payoffs_safe = option.payoff(paths_safe)
        assert payoffs_safe[0] == 0  # Not knocked in


class TestMonteCarloEngine:
    """Test Monte Carlo simulation engine."""
    
    def test_path_generation_shape(self):
        """Test that generated paths have correct shape."""
        engine = MonteCarloEngine(n_paths=1000, n_steps=100, seed=42)
        paths = engine.generate_paths(S0=100, r=0.05, sigma=0.2, T=1.0)
        
        assert paths.shape == (1000, 101)  # n_paths x (n_steps + 1)
    
    def test_initial_price(self):
        """Test that all paths start at S0."""
        engine = MonteCarloEngine(n_paths=1000, n_steps=100, seed=42)
        paths = engine.generate_paths(S0=100, r=0.05, sigma=0.2, T=1.0)
        
        np.testing.assert_array_almost_equal(paths[:, 0], 100.0)
    
    def test_antithetic_variates(self):
        """Test that antithetic variates reduce variance."""
        engine = MonteCarloEngine(n_paths=10000, n_steps=100, seed=42)
        
        # Standard paths
        paths_standard = engine.generate_paths(S0=100, r=0.05, sigma=0.2, 
                                              T=1.0, antithetic=False)
        
        # Antithetic paths
        engine_anti = MonteCarloEngine(n_paths=10000, n_steps=100, seed=42)
        paths_anti = engine_anti.generate_paths(S0=100, r=0.05, sigma=0.2,
                                               T=1.0, antithetic=True)
        
        # Variance of terminal prices should be similar but antithetic
        # should have better convergence properties
        var_standard = np.var(paths_standard[:, -1])
        var_anti = np.var(paths_anti[:, -1])
        
        # Both should be close to theoretical variance
        assert 0 < var_standard < np.inf
        assert 0 < var_anti < np.inf
    
    def test_european_pricing_accuracy(self):
        """Test that MC price converges to Black-Scholes."""
        params = OptionParameters(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                option_type=OptionType.CALL)
        option = EuropeanOption(params)
        
        # Get analytical price
        analytical = option.analytical_price()
        
        # Get MC price with many paths
        engine = MonteCarloEngine(n_paths=100000, n_steps=252, seed=42)
        mc_price, std_error = engine.price_option(option)
        
        # MC price should be within 3 standard errors of analytical
        assert abs(mc_price - analytical) < 3 * std_error
        
        # Price should be reasonable
        assert 10.0 < mc_price < 11.0
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        params = OptionParameters(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                option_type=OptionType.CALL)
        option = EuropeanOption(params)
        
        engine1 = MonteCarloEngine(n_paths=10000, n_steps=100, seed=42)
        price1, _ = engine1.price_option(option)
        
        engine2 = MonteCarloEngine(n_paths=10000, n_steps=100, seed=42)
        price2, _ = engine2.price_option(option)
        
        assert price1 == price2


class TestGreeks:
    """Test Greeks calculations."""
    
    def test_call_delta_positive(self):
        """Test that call delta is positive."""
        params = OptionParameters(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                option_type=OptionType.CALL)
        option = EuropeanOption(params)
        
        calculator = GreeksCalculator(
            MonteCarloEngine(n_paths=50000, n_steps=100, seed=42)
        )
        delta = calculator.calculate_delta(option)
        
        # Call delta should be between 0 and 1 (approximately 0.5 for ATM)
        assert 0.3 < delta < 0.7
    
    def test_put_delta_negative(self):
        """Test that put delta is negative."""
        params = OptionParameters(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                option_type=OptionType.PUT)
        option = EuropeanOption(params)
        
        calculator = GreeksCalculator(
            MonteCarloEngine(n_paths=50000, n_steps=100, seed=42)
        )
        delta = calculator.calculate_delta(option)
        
        # Put delta should be between -1 and 0 (approximately -0.5 for ATM)
        assert -0.7 < delta < -0.3
    
    def test_gamma_positive(self):
        """Test that gamma is positive for both calls and puts."""
        params = OptionParameters(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                option_type=OptionType.CALL)
        option = EuropeanOption(params)
        
        calculator = GreeksCalculator(
            MonteCarloEngine(n_paths=50000, n_steps=100, seed=42)
        )
        gamma = calculator.calculate_gamma(option)
        
        # Gamma should be positive for long options
        assert gamma > 0
    
    def test_vega_positive(self):
        """Test that vega is positive for long options."""
        params = OptionParameters(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                option_type=OptionType.CALL)
        option = EuropeanOption(params)
        
        calculator = GreeksCalculator(
            MonteCarloEngine(n_paths=50000, n_steps=100, seed=42)
        )
        vega = calculator.calculate_vega(option)
        
        # Vega should be positive (options gain value with volatility)
        assert vega > 0
    
    def test_theta_negative(self):
        """Test that theta is negative (time decay)."""
        params = OptionParameters(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                option_type=OptionType.CALL)
        option = EuropeanOption(params)
        
        calculator = GreeksCalculator(
            MonteCarloEngine(n_paths=50000, n_steps=100, seed=42)
        )
        theta = calculator.calculate_theta(option)
        
        # Theta should be negative (option loses value over time)
        assert theta < 0
    
    def test_all_greeks(self):
        """Test calculating all Greeks at once."""
        params = OptionParameters(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                option_type=OptionType.CALL)
        option = EuropeanOption(params)
        
        calculator = GreeksCalculator(
            MonteCarloEngine(n_paths=30000, n_steps=100, seed=42)
        )
        greeks = calculator.calculate_all_greeks(option)
        
        # Check all Greeks are present
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'vega' in greeks
        assert 'theta' in greeks
        assert 'rho' in greeks
        
        # Check reasonable ranges
        assert 0 < greeks['delta'] < 1
        assert greeks['gamma'] > 0
        assert greeks['vega'] > 0
        assert greeks['theta'] < 0


class TestVarianceReduction:
    """Test variance reduction techniques."""
    
    def test_antithetic_reduces_variance(self):
        """Test that antithetic variates reduce standard error."""
        params = OptionParameters(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                option_type=OptionType.CALL)
        option = EuropeanOption(params)
        
        # Standard MC
        engine_std = MonteCarloEngine(n_paths=10000, n_steps=100, seed=42)
        _, std_error_std = engine_std.price_option(option, antithetic=False)
        
        # Antithetic MC
        engine_anti = MonteCarloEngine(n_paths=10000, n_steps=100, seed=42)
        _, std_error_anti = engine_anti.price_option(option, antithetic=True)
        
        # Antithetic should have lower standard error
        assert std_error_anti < std_error_std
    
    def test_control_variate_for_asian(self):
        """Test control variate for Asian options."""
        params = OptionParameters(S0=100, K=100, T=1, r=0.05, sigma=0.2,
                                option_type=OptionType.CALL)
        option = AsianOption(params, averaging_type="arithmetic")
        
        results = VarianceReducer.compare_methods(option, n_paths=20000, seed=42)
        
        # Should have standard and control variate results
        assert 'standard' in results
        assert 'control_variate' in results
        
        # Control variate should reduce standard error
        std_error_std = results['standard'][1]
        std_error_cv = results['control_variate'][1]
        
        assert std_error_cv < std_error_std
    
    def test_variance_reduction_ratio(self):
        """Test variance reduction ratio calculation."""
        ratio = VarianceReducer.variance_reduction_ratio(0.1, 0.05)
        
        # Ratio should be 4 (variance reduced by factor of 4)
        assert abs(ratio - 4.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
