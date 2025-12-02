"""
Monte Carlo Option Pricing Engine - Python Implementation

A comprehensive option pricing library implementing Monte Carlo simulation
with variance reduction techniques and analytical Greeks calculations.
"""

from .options import (
    OptionType,
    BarrierType,
    OptionParameters,
    Option,
    EuropeanOption,
    AsianOption,
    BarrierOption,
)
from .monte_carlo import MonteCarloEngine
from .greeks import GreeksCalculator
from .variance_reduction import VarianceReducer
from .utils import norm_cdf, norm_pdf

__version__ = "1.0.0"
__all__ = [
    "OptionType",
    "BarrierType",
    "OptionParameters",
    "Option",
    "EuropeanOption",
    "AsianOption",
    "BarrierOption",
    "MonteCarloEngine",
    "GreeksCalculator",
    "VarianceReducer",
    "norm_cdf",
    "norm_pdf",
]
