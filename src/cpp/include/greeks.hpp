#pragma once

#include "common.hpp"
#include "options.hpp"
#include "monte_carlo.hpp"

namespace mc {

class GreeksCalculator {
public:
    GreeksCalculator(MonteCarloEngine& engine, double bump_size = 0.01);
    
    // Calculate all Greeks using finite differences
    Greeks calculate(const Option& option);
    
    // Calculate individual Greeks
    double delta(const Option& option);
    double gamma(const Option& option);
    double vega(const Option& option);
    double theta(const Option& option);
    double rho(const Option& option);
    
private:
    MonteCarloEngine& engine_;
    double bump_size_;
};

} // namespace mc
