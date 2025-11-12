#pragma once

#include "common.hpp"
#include "options.hpp"
#include "random_generator.hpp"

namespace mc {

class MonteCarloEngine {
public:
    MonteCarloEngine(size_t n_paths, size_t n_steps, unsigned int seed = 42);
    
    // Price an option
    PricingResult price(const Option& option, bool use_antithetic = false);
    
    // Price with control variate (requires European option as control)
    PricingResult price_with_control_variate(
        const Option& option,
        const EuropeanOption& control_option
    );
    
    // Generate stock price path
    void generate_path(std::vector<double>& path, const OptionParameters& params);
    
    // Setters
    void set_n_paths(size_t n) { n_paths_ = n; }
    void set_n_steps(size_t n) { n_steps_ = n; }
    void set_seed(unsigned int seed) { rng_.reset(seed); }
    
    // Getters
    size_t n_paths() const { return n_paths_; }
    size_t n_steps() const { return n_steps_; }
    
private:
    size_t n_paths_;
    size_t n_steps_;
    RandomGenerator rng_;
    
    double simulate_single_path(const Option& option);
};

} // namespace mc