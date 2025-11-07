#include "greeks.hpp"
#include "options.hpp"

namespace mc {

GreeksCalculator::GreeksCalculator(MonteCarloEngine& engine, double bump_size)
    : engine_(engine), bump_size_(bump_size) {}

Greeks GreeksCalculator::calculate(const Option& option) {
    Greeks greeks;
    greeks.delta = delta(option);
    greeks.gamma = gamma(option);
    greeks.vega = vega(option);
    greeks.theta = theta(option);
    greeks.rho = rho(option);
    return greeks;
}

double GreeksCalculator::delta(const Option& option) {
    auto params = option.params();
    
    // Bump spot price up
    auto params_up = params;
    params_up.S0 = params.S0 * (1.0 + bump_size_);
    EuropeanOption option_up(params_up);
    double price_up = engine_.price(option_up).price;
    
    // Bump spot price down
    auto params_down = params;
    params_down.S0 = params.S0 * (1.0 - bump_size_);
    EuropeanOption option_down(params_down);
    double price_down = engine_.price(option_down).price;
    
    return (price_up - price_down) / (2.0 * params.S0 * bump_size_);
}

double GreeksCalculator::gamma(const Option& option) {
    auto params = option.params();
    
    // Central price
    double price = engine_.price(option).price;
    
    // Bump up
    auto params_up = params;
    params_up.S0 = params.S0 * (1.0 + bump_size_);
    EuropeanOption option_up(params_up);
    double price_up = engine_.price(option_up).price;
    
    // Bump down
    auto params_down = params;
    params_down.S0 = params.S0 * (1.0 - bump_size_);
    EuropeanOption option_down(params_down);
    double price_down = engine_.price(option_down).price;
    
    double dS = params.S0 * bump_size_;
    return (price_up - 2.0 * price + price_down) / (dS * dS);
}

double GreeksCalculator::vega(const Option& option) {
    auto params = option.params();
    
    // Bump volatility up
    auto params_up = params;
    params_up.sigma = params.sigma + 0.01; // 1% absolute change
    EuropeanOption option_up(params_up);
    double price_up = engine_.price(option_up).price;
    
    // Bump volatility down
    auto params_down = params;
    params_down.sigma = params.sigma - 0.01;
    EuropeanOption option_down(params_down);
    double price_down = engine_.price(option_down).price;
    
    return (price_up - price_down) / 2.0; // Vega per 1% change
}

double GreeksCalculator::theta(const Option& option) {
    auto params = option.params();
    
    // Current price
    double price = engine_.price(option).price;
    
    // Bump time down (1 day)
    auto params_future = params;
    params_future.T = params.T - 1.0/365.0;
    if (params_future.T <= 0) return 0.0;
    
    EuropeanOption option_future(params_future);
    double price_future = engine_.price(option_future).price;
    
    return price_future - price; // Theta per day
}

double GreeksCalculator::rho(const Option& option) {
    auto params = option.params();
    
    // Bump rate up
    auto params_up = params;
    params_up.r = params.r + 0.01; // 1% absolute change
    EuropeanOption option_up(params_up);
    double price_up = engine_.price(option_up).price;
    
    // Bump rate down
    auto params_down = params;
    params_down.r = params.r - 0.01;
    EuropeanOption option_down(params_down);
    double price_down = engine_.price(option_down).price;
    
    return (price_up - price_down) / 2.0; // Rho per 1% change
}

} // namespace mc
