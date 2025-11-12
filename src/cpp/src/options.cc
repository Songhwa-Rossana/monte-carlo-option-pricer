#include "options.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace mc {

namespace {
    // Standard normal CDF
    double norm_cdf(double x) {
        return 0.5 * std::erfc(-x * M_SQRT1_2);
    }
    
    // Standard normal PDF
    double norm_pdf(double x) {
        return std::exp(-0.5 * x * x) / SQRT_2PI;
    }
}

// European Option
EuropeanOption::EuropeanOption(const OptionParameters& params) 
    : Option(params) {}

double EuropeanOption::payoff(const std::vector<double>& path) const {
    return payoff(path.back());
}

double EuropeanOption::payoff(double S_T) const {
    if (params_.type == OptionType::CALL) {
        return std::max(S_T - params_.K, 0.0);
    } else {
        return std::max(params_.K - S_T, 0.0);
    }
}

double EuropeanOption::black_scholes_price() const {
    double S = params_.S0;
    double K = params_.K;
    double T = params_.T;
    double r = params_.r;
    double sigma = params_.sigma;
    
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    
    if (params_.type == OptionType::CALL) {
        return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
    } else {
        return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
    }
}

double EuropeanOption::black_scholes_delta() const {
    double S = params_.S0;
    double K = params_.K;
    double T = params_.T;
    double r = params_.r;
    double sigma = params_.sigma;
    
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    
    if (params_.type == OptionType::CALL) {
        return norm_cdf(d1);
    } else {
        return norm_cdf(d1) - 1.0;
    }
}

double EuropeanOption::black_scholes_gamma() const {
    double S = params_.S0;
    double K = params_.K;
    double T = params_.T;
    double r = params_.r;
    double sigma = params_.sigma;
    
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    
    return norm_pdf(d1) / (S * sigma * std::sqrt(T));
}

double EuropeanOption::black_scholes_vega() const {
    double S = params_.S0;
    double K = params_.K;
    double T = params_.T;
    double r = params_.r;
    double sigma = params_.sigma;
    
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    
    return S * norm_pdf(d1) * std::sqrt(T) / 100.0; // Divide by 100 for 1% change
}

double EuropeanOption::black_scholes_theta() const {
    double S = params_.S0;
    double K = params_.K;
    double T = params_.T;
    double r = params_.r;
    double sigma = params_.sigma;
    
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    
    double term1 = -(S * norm_pdf(d1) * sigma) / (2.0 * std::sqrt(T));
    
    if (params_.type == OptionType::CALL) {
        double term2 = r * K * std::exp(-r * T) * norm_cdf(d2);
        return (term1 - term2) / 365.0; // Per day
    } else {
        double term2 = r * K * std::exp(-r * T) * norm_cdf(-d2);
        return (term1 + term2) / 365.0; // Per day
    }
}

double EuropeanOption::black_scholes_rho() const {
    double S = params_.S0;
    double K = params_.K;
    double T = params_.T;
    double r = params_.r;
    double sigma = params_.sigma;
    
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    
    if (params_.type == OptionType::CALL) {
        return K * T * std::exp(-r * T) * norm_cdf(d2) / 100.0; // For 1% change
    } else {
        return -K * T * std::exp(-r * T) * norm_cdf(-d2) / 100.0; // For 1% change
    }
}

// Asian Option
AsianOption::AsianOption(const OptionParameters& params)
    : Option(params) {}

double AsianOption::payoff(const std::vector<double>& path) const {
    double avg = std::accumulate(path.begin(), path.end(), 0.0) / path.size();
    
    if (params_.type == OptionType::CALL) {
        return std::max(avg - params_.K, 0.0);
    } else {
        return std::max(params_.K - avg, 0.0);
    }
}

double AsianOption::payoff(double S_T) const {
    // Path-dependent, cannot use single price
    return 0.0;
}

// Geometric Asian Option
GeometricAsianOption::GeometricAsianOption(const OptionParameters& params)
    : Option(params) {}

double GeometricAsianOption::payoff(const std::vector<double>& path) const {
    double log_sum = 0.0;
    for (double price : path) {
        log_sum += std::log(price + EPSILON);
    }
    double geometric_avg = std::exp(log_sum / path.size());
    
    if (params_.type == OptionType::CALL) {
        return std::max(geometric_avg - params_.K, 0.0);
    } else {
        return std::max(params_.K - geometric_avg, 0.0);
    }
}

double GeometricAsianOption::payoff(double S_T) const {
    // Path-dependent, cannot use single price
    return 0.0;
}

// Barrier Option
BarrierOption::BarrierOption(const OptionParameters& params, double barrier, BarrierType barrier_type)
    : Option(params), barrier_(barrier), barrier_type_(barrier_type) {}

bool BarrierOption::is_barrier_crossed(const std::vector<double>& path) const {
    switch (barrier_type_) {
        case BarrierType::UP_IN:
        case BarrierType::UP_OUT:
            return std::any_of(path.begin(), path.end(), 
                             [this](double s) { return s >= barrier_; });
        case BarrierType::DOWN_IN:
        case BarrierType::DOWN_OUT:
            return std::any_of(path.begin(), path.end(), 
                             [this](double s) { return s <= barrier_; });
    }
    return false;
}

double BarrierOption::payoff(const std::vector<double>& path) const {
    bool crossed = is_barrier_crossed(path);
    double terminal_payoff;
    
    if (params_.type == OptionType::CALL) {
        terminal_payoff = std::max(path.back() - params_.K, 0.0);
    } else {
        terminal_payoff = std::max(params_.K - path.back(), 0.0);
    }
    
    switch (barrier_type_) {
        case BarrierType::UP_IN:
        case BarrierType::DOWN_IN:
            return crossed ? terminal_payoff : 0.0;
        case BarrierType::UP_OUT:
        case BarrierType::DOWN_OUT:
            return crossed ? 0.0 : terminal_payoff;
    }
    return 0.0;
}

double BarrierOption::payoff(double S_T) const {
    // Path-dependent, cannot use single price
    return 0.0;
}

} // namespace mc