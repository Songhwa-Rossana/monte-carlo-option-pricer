#include "monte_carlo.hpp"
#include <cmath>
#include <omp.h>

namespace mc {

MonteCarloEngine::MonteCarloEngine(size_t n_paths, size_t n_steps, unsigned int seed)
    : n_paths_(n_paths), n_steps_(n_steps), rng_(seed) {}

void MonteCarloEngine::generate_path(std::vector<double>& path, const OptionParameters& params) {
    path.resize(n_steps_ + 1);
    path[0] = params.S0;
    
    double dt = params.T / n_steps_;
    double drift = (params.r - 0.5 * params.sigma * params.sigma) * dt;
    double vol_sqrt_dt = params.sigma * std::sqrt(dt);
    
    for (size_t i = 1; i <= n_steps_; ++i) {
        double z = rng_.normal();
        path[i] = path[i-1] * std::exp(drift + vol_sqrt_dt * z);
    }
}

double MonteCarloEngine::simulate_single_path(const Option& option) {
    std::vector<double> path;
    generate_path(path, option.params());
    return option.payoff(path);
}

PricingResult MonteCarloEngine::price(const Option& option, bool use_antithetic) {
    std::vector<double> payoffs(n_paths_);
    
    if (use_antithetic) {
        // Use antithetic variates
        size_t half_paths = n_paths_ / 2;
        
        #pragma omp parallel
        {
            RandomGenerator local_rng(rng_.uniform() * 1000000);
            std::vector<double> path(n_steps_ + 1);
            std::vector<double> anti_path(n_steps_ + 1);
            
            #pragma omp for
            for (size_t i = 0; i < half_paths; ++i) {
                // Generate path
                path[0] = option.params().S0;
                anti_path[0] = option.params().S0;
                
                double dt = option.params().T / n_steps_;
                double drift = (option.params().r - 0.5 * option.params().sigma * option.params().sigma) * dt;
                double vol_sqrt_dt = option.params().sigma * std::sqrt(dt);
                
                for (size_t j = 1; j <= n_steps_; ++j) {
                    double z = local_rng.normal();
                    path[j] = path[j-1] * std::exp(drift + vol_sqrt_dt * z);
                    anti_path[j] = anti_path[j-1] * std::exp(drift - vol_sqrt_dt * z);
                }
                
                payoffs[2*i] = option.payoff(path);
                payoffs[2*i + 1] = option.payoff(anti_path);
            }
        }
    } else {
        // Standard Monte Carlo
        #pragma omp parallel
        {
            RandomGenerator local_rng(rng_.uniform() * 1000000);
            std::vector<double> path;
            
            #pragma omp for
            for (size_t i = 0; i < n_paths_; ++i) {
                path.resize(n_steps_ + 1);
                path[0] = option.params().S0;
                
                double dt = option.params().T / n_steps_;
                double drift = (option.params().r - 0.5 * option.params().sigma * option.params().sigma) * dt;
                double vol_sqrt_dt = option.params().sigma * std::sqrt(dt);
                
                for (size_t j = 1; j <= n_steps_; ++j) {
                    double z = local_rng.normal();
                    path[j] = path[j-1] * std::exp(drift + vol_sqrt_dt * z);
                }
                
                payoffs[i] = option.payoff(path);
            }
        }
    }
    
    // Calculate statistics
    double sum = 0.0;
    double sum_sq = 0.0;
    
    for (double p : payoffs) {
        sum += p;
        sum_sq += p * p;
    }
    
    double mean = sum / n_paths_;
    double variance = (sum_sq / n_paths_) - (mean * mean);
    double std_error = std::sqrt(variance / n_paths_);
    
    // Discount to present value
    double discount = std::exp(-option.params().r * option.params().T);
    double price = mean * discount;
    double se = std_error * discount;
    
    // 95% confidence interval
    double z_score = 1.96;
    PricingResult result;
    result.price = price;
    result.std_error = se;
    result.confidence_lower = price - z_score * se;
    result.confidence_upper = price + z_score * se;
    
    return result;
}

PricingResult MonteCarloEngine::price_with_control_variate(
    const Option& option,
    const EuropeanOption& control_option) {
    
    std::vector<double> target_payoffs(n_paths_);
    std::vector<double> control_payoffs(n_paths_);
    
    #pragma omp parallel
    {
        RandomGenerator local_rng(rng_.uniform() * 1000000);
        std::vector<double> path;
        
        #pragma omp for
        for (size_t i = 0; i < n_paths_; ++i) {
            path.resize(n_steps_ + 1);
            path[0] = option.params().S0;
            
            double dt = option.params().T / n_steps_;
            double drift = (option.params().r - 0.5 * option.params().sigma * option.params().sigma) * dt;
            double vol_sqrt_dt = option.params().sigma * std::sqrt(dt);
            
            for (size_t j = 1; j <= n_steps_; ++j) {
                double z = local_rng.normal();
                path[j] = path[j-1] * std::exp(drift + vol_sqrt_dt * z);
            }
            
            target_payoffs[i] = option.payoff(path);
            control_payoffs[i] = control_option.payoff(path);
        }
    }
    
    // Calculate correlation and optimal c
    double mean_target = 0.0, mean_control = 0.0;
    for (size_t i = 0; i < n_paths_; ++i) {
        mean_target += target_payoffs[i];
        mean_control += control_payoffs[i];
    }
    mean_target /= n_paths_;
    mean_control /= n_paths_;
    
    double cov = 0.0, var_control = 0.0;
    for (size_t i = 0; i < n_paths_; ++i) {
        double dt = target_payoffs[i] - mean_target;
        double dc = control_payoffs[i] - mean_control;
        cov += dt * dc;
        var_control += dc * dc;
    }
    
    double c = cov / (var_control + EPSILON);
    double control_exact = control_option.black_scholes_price() / std::exp(-option.params().r * option.params().T);
    
    // Apply control variate
    std::vector<double> adjusted_payoffs(n_paths_);
    for (size_t i = 0; i < n_paths_; ++i) {
        adjusted_payoffs[i] = target_payoffs[i] - c * (control_payoffs[i] - control_exact);
    }
    
    // Calculate statistics
    double sum = 0.0, sum_sq = 0.0;
    for (double p : adjusted_payoffs) {
        sum += p;
        sum_sq += p * p;
    }
    
    double mean = sum / n_paths_;
    double variance = (sum_sq / n_paths_) - (mean * mean);
    double std_error = std::sqrt(variance / n_paths_);
    
    double discount = std::exp(-option.params().r * option.params().T);
    
    PricingResult result;
    result.price = mean * discount;
    result.std_error = std_error * discount;
    result.confidence_lower = result.price - 1.96 * result.std_error;
    result.confidence_upper = result.price + 1.96 * result.std_error;
    
    return result;
}

} // namespace mc