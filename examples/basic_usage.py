"""
Basic usage examples for Monte Carlo option pricing.
"""

import numpy as np
from src.python.options import (
    OptionType, OptionParameters, EuropeanOption,
    AsianOption, BarrierOption, BarrierType
)
from src.python.monte_carlo import MonteCarloEngine
from src.python.greeks import GreeksCalculator
from src.python.variance_reduction import VarianceReducer


def example_european_option():
    """Price a European call option and compare with Black-Scholes."""
    print("=" * 70)
    print("EXAMPLE 1: European Call Option")
    print("=" * 70)
    
    # Define option parameters
    params = OptionParameters(
        S0=100.0,      # Stock price
        K=100.0,       # Strike price
        T=1.0,         # Time to maturity (years)
        r=0.05,        # Risk-free rate (5%)
        sigma=0.2,     # Volatility (20%)
        option_type=OptionType.CALL
    )
    
    # Create option
    option = EuropeanOption(params)
    
    # Get analytical Black-Scholes price
    analytical_price = option.analytical_price()
    print(f"\nBlack-Scholes Analytical Price: ${analytical_price:.4f}")
    
    # Price using Monte Carlo
    engine = MonteCarloEngine(n_paths=100000, n_steps=252, seed=42)
    mc_price, std_error = engine.price_option(option)
    
    print(f"Monte Carlo Price: ${mc_price:.4f} ± ${std_error:.4f}")
    print(f"Difference: ${abs(mc_price - analytical_price):.4f}")
    print(f"Within {abs(mc_price - analytical_price) / std_error:.2f} standard errors")
    print()


def example_asian_option():
    """Price an Asian option with arithmetic averaging."""
    print("=" * 70)
    print("EXAMPLE 2: Asian Call Option (Arithmetic Average)")
    print("=" * 70)
    
    params = OptionParameters(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.CALL
    )
    
    # Create Asian option
    option = AsianOption(params, averaging_type="arithmetic")
    
    # Price using Monte Carlo
    engine = MonteCarloEngine(n_paths=100000, n_steps=252, seed=42)
    price, std_error = engine.price_option(option)
    
    print(f"\nAsian Option Price: ${price:.4f} ± ${std_error:.4f}")
    print(f"Note: Asian options are typically cheaper than European options")
    print(f"      due to averaging reducing volatility.\n")


def example_barrier_option():
    """Price an up-and-out barrier option."""
    print("=" * 70)
    print("EXAMPLE 3: Up-and-Out Barrier Call Option")
    print("=" * 70)
    
    params = OptionParameters(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.CALL
    )
    
    # Create barrier option (knocked out if price reaches 120)
    option = BarrierOption(params, barrier=120.0, 
                          barrier_type=BarrierType.UP_AND_OUT)
    
    # Price using Monte Carlo
    engine = MonteCarloEngine(n_paths=100000, n_steps=252, seed=42)
    price, std_error = engine.price_option(option)
    
    print(f"\nBarrier: ${option.barrier:.2f}")
    print(f"Barrier Option Price: ${price:.4f} ± ${std_error:.4f}")
    print(f"Note: Barrier options are cheaper than vanilla options")
    print(f"      because they can be knocked out.\n")


def example_greeks():
    """Calculate Greeks for a European call option."""
    print("=" * 70)
    print("EXAMPLE 4: Greeks Calculation")
    print("=" * 70)
    
    params = OptionParameters(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.CALL
    )
    
    option = EuropeanOption(params)
    
    # Calculate Greeks
    print("\nCalculating Greeks (this may take a moment)...")
    calculator = GreeksCalculator(
        MonteCarloEngine(n_paths=50000, n_steps=100, seed=42)
    )
    
    greeks = calculator.calculate_all_greeks(option)
    
    print(f"\nOption Greeks:")
    print(f"  Delta:  {greeks['delta']:7.4f}  (sensitivity to stock price)")
    print(f"  Gamma:  {greeks['gamma']:7.4f}  (rate of change of delta)")
    print(f"  Vega:   {greeks['vega']:7.4f}  (sensitivity to volatility)")
    print(f"  Theta:  {greeks['theta']:7.4f}  (time decay per day)")
    print(f"  Rho:    {greeks['rho']:7.4f}  (sensitivity to interest rate)")
    print()


def example_variance_reduction():
    """Compare variance reduction techniques."""
    print("=" * 70)
    print("EXAMPLE 5: Variance Reduction Techniques")
    print("=" * 70)
    
    params = OptionParameters(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.CALL
    )
    
    # Use Asian option to demonstrate control variates
    option = AsianOption(params, averaging_type="arithmetic")
    
    print("\nComparing variance reduction methods for Asian option...")
    print("(Using same number of paths for fair comparison)\n")
    
    results = VarianceReducer.compare_methods(option, n_paths=50000, seed=42)
    
    # Display results
    print(f"{'Method':<20} {'Price':>10} {'Std Error':>12} {'Efficiency Gain':>16}")
    print("-" * 60)
    
    standard_error = results['standard'][1]
    
    for method, (price, std_error) in results.items():
        efficiency = VarianceReducer.efficiency_gain(standard_error, std_error)
        print(f"{method.replace('_', ' ').title():<20} "
              f"${price:>9.4f} ${std_error:>11.5f} "
              f"{efficiency:>15.2f}x")
    
    print("\nNote: Efficiency gain shows how many times fewer simulations")
    print("      are needed to achieve the same accuracy as standard MC.\n")


def example_convergence():
    """Demonstrate Monte Carlo convergence."""
    print("=" * 70)
    print("EXAMPLE 6: Monte Carlo Convergence")
    print("=" * 70)
    
    params = OptionParameters(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.CALL
    )
    
    option = EuropeanOption(params)
    analytical = option.analytical_price()
    
    print(f"\nBlack-Scholes Price: ${analytical:.4f}\n")
    print(f"{'Paths':>10} {'MC Price':>12} {'Std Error':>12} {'Error':>10}")
    print("-" * 50)
    
    path_counts = [1000, 5000, 10000, 50000, 100000]
    
    for n_paths in path_counts:
        engine = MonteCarloEngine(n_paths=n_paths, n_steps=252, seed=42)
        price, std_error = engine.price_option(option)
        error = abs(price - analytical)
        
        print(f"{n_paths:>10,} ${price:>11.4f} ${std_error:>11.5f} "
              f"${error:>9.4f}")
    
    print("\nNote: Standard error decreases as O(1/sqrt(N))")
    print("      Doubling accuracy requires 4x more simulations.\n")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  🎲 Monte Carlo Option Pricing - Usage Examples  ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")
    
    example_european_option()
    example_asian_option()
    example_barrier_option()
    example_greeks()
    example_variance_reduction()
    example_convergence()
    
    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
