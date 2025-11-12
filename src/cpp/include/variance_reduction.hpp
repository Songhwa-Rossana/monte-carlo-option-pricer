#pragma once

#include "common.hpp"
#include "options.hpp"
#include "monte_carlo.hpp"

namespace mc {

// Antithetic variates pricing
PricingResult price_antithetic(
    MonteCarloEngine& engine,
    const Option& option
);

// Control variate pricing
PricingResult price_control_variate(
    MonteCarloEngine& engine,
    const Option& target_option,
    const EuropeanOption& control_option
);

} // namespace mc