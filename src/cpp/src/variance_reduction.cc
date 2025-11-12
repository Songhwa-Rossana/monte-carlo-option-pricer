#include "variance_reduction.hpp"

namespace mc {

PricingResult price_antithetic(MonteCarloEngine& engine, const Option& option) {
    return engine.price(option, true);
}

PricingResult price_control_variate(
    MonteCarloEngine& engine,
    const Option& target_option,
    const EuropeanOption& control_option) {
    
    return engine.price_with_control_variate(target_option, control_option);
}

} // namespace mc