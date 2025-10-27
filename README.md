# 🎲 Monte Carlo Option Pricing Engine

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![C++](https://img.shields.io/badge/C++-17-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)

A high-performance Monte Carlo simulation engine for pricing financial derivatives, built in both Python and C++ as a quantitative finance learning and portfolio project.

## 🎯 Project Goals

- Learn and implement Monte Carlo methods for derivative pricing
- Understand variance reduction techniques and their impact
- Gain experience with C++ optimization and Python-C++ integration
- Build a portfolio-ready quantitative finance project
- **Timeline: 5 days (Oct 27-31, 2025)**

## ✨ Features

- **Multiple Option Types**: European, Asian (arithmetic/geometric), Barrier (knock-in/knock-out)
- **Variance Reduction**: Antithetic variates, control variates for faster convergence
- **Greeks Calculation**: Delta, Gamma, Vega, Theta, Rho using finite differences
- **Dual Implementation**: Python for prototyping, C++ for performance (10-100x speedup)
- **Multi-threading**: OpenMP parallelization for maximum performance
- **Python Integration**: Seamless Python-C++ bindings via pybind11
- **Comprehensive Testing**: pytest for Python, Google Test for C++
- **Visual Analytics**: Jupyter notebooks with convergence and sensitivity analysis

## 🚀 Quick Start

```python
from src.python.options import EuropeanOption, OptionParameters, OptionType
from src.python.monte_carlo import MonteCarloEngine

# Define option parameters
params = OptionParameters(
    S0=100.0,      # Stock price
    K=100.0,       # Strike price
    T=1.0,         # Time to maturity (years)
    r=0.05,        # Risk-free rate
    sigma=0.2,     # Volatility
    option_type=OptionType.CALL
)

# Price the option
option = EuropeanOption(params)
engine = MonteCarloEngine(n_paths=100000, n_steps=252)
price, std_error = engine.price_option(option)

print(f"Option Price: ${price:.4f} ± ${std_error:.4f}")
```

## 📊 Performance

Expected performance on modern hardware:
- **Python**: ~50,000 paths/second
- **C++ (single-thread)**: ~500,000 paths/second (10x faster)
- **C++ (8 threads)**: ~3,000,000 paths/second (60x faster)

## 🗓️ Development Timeline

- **Day 1 (Oct 27)**: Project setup and structure
- **Day 2 (Oct 28)**: Python core implementation
- **Day 3 (Oct 29)**: C++ core implementation
- **Day 4 (Oct 30)**: Advanced features and testing
- **Day 5 (Oct 31)**: Documentation and polish

## 📚 Learning Outcomes

This project demonstrates:
- Understanding of derivative pricing and the Black-Scholes model
- Implementation of Monte Carlo methods and variance reduction
- Numerical methods for Greeks calculation
- C++ optimization techniques and multi-threading
- Python-C++ integration with pybind11
- Software engineering best practices (testing, documentation, CI/CD)

## 🛠️ Tech Stack

- **Python**: NumPy, SciPy, matplotlib, pandas, pytest
- **C++**: C++17, STL, OpenMP
- **Integration**: pybind11
- **Build**: CMake
- **Testing**: pytest, Google Test
- **Documentation**: Jupyter notebooks, Markdown

## 📖 Documentation

Coming soon:
- Installation guide
- API reference
- Mathematical background
- Usage examples
- Performance benchmarks

## 🤝 Contributing

This is a personal learning/portfolio project, but feedback and suggestions are welcome!

## 📄 License

MIT License - see LICENSE file for details

## 👤 Author

**Songhwa-Rossana**
- Portfolio project demonstrating quantitative finance and computational skills
- Built in 5 days as a focused learning sprint

## 🔗 References

1. Hull, J. C. (2018). *Options, Futures, and Other Derivatives*
2. Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*
3. Joshi, M. S. (2008). *C++ Design Patterns and Derivatives Pricing*

---

⭐ Star this repo if you find it helpful for learning quantitative finance!
