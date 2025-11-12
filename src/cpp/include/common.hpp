#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <memory>

namespace mc {

// Constants
constexpr double EPSILON = 1e-8;
constexpr double SQRT_2PI = 2.506628274631000502;

// Option type enumeration
enum class OptionType {
    CALL,
    PUT
};

// Barrier type enumeration
enum class BarrierType {
    UP_IN,
    UP_OUT,
    DOWN_IN,
    DOWN_OUT
};

// Option parameters structure
struct OptionParameters {
    double S0;          // Initial stock price
    double K;           // Strike price
    double T;           // Time to maturity (years)
    double r;           // Risk-free rate
    double sigma;       // Volatility
    OptionType type;    // Call or Put
    
    OptionParameters(double s0, double k, double t, double rate, double vol, OptionType opt_type)
        : S0(s0), K(k), T(t), r(rate), sigma(vol), type(opt_type) {}
};

// Pricing result structure
struct PricingResult {
    double price;
    double std_error;
    double confidence_lower;
    double confidence_upper;
    
    PricingResult(double p = 0.0, double se = 0.0, double cl = 0.0, double cu = 0.0)
        : price(p), std_error(se), confidence_lower(cl), confidence_upper(cu) {}
};

// Greeks structure
struct Greeks {
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
    
    Greeks() : delta(0), gamma(0), vega(0), theta(0), rho(0) {}
};

} // namespace mc