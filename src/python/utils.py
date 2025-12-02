"""
Mathematical utility functions for option pricing.

Implements standard normal distribution functions and
other mathematical helpers. 
"""

import numpy as np
from scipy import stats


def norm_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function.
    
    Φ(x) = P(Z ≤ x) where Z ~ N(0,1)
    
    Args:
        x: Point at which to evaluate CDF
    
    Returns:
        Cumulative probability
    """
    return stats.norm.cdf(x)


def norm_pdf(x: float) -> float:
    """
    Standard normal probability density function.
    
    φ(x) = (1/√(2π)) * exp(-x²/2)
    
    Args:
        x: Point at which to evaluate PDF
    
    Returns:
        Probability density
    """
    return stats.norm. pdf(x)
