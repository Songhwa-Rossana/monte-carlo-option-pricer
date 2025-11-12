#pragma once

#include "common.hpp"

namespace mc {

// Base option class
class Option {
public:
    virtual ~Option() = default;
    
    virtual double payoff(const std::vector<double>& path) const = 0;
    virtual double payoff(double S_T) const = 0;
    
    const OptionParameters& params() const { return params_; }
    
protected:
    Option(const OptionParameters& params) : params_(params) {}
    OptionParameters params_;
};

// European option
class EuropeanOption : public Option {
public:
    explicit EuropeanOption(const OptionParameters& params);
    
    double payoff(const std::vector<double>& path) const override;
    double payoff(double S_T) const override;
    
    // Analytical Black-Scholes price for comparison
    double black_scholes_price() const;
    double black_scholes_delta() const;
    double black_scholes_gamma() const;
    double black_scholes_vega() const;
    double black_scholes_theta() const;
    double black_scholes_rho() const;
};

// Asian option (arithmetic average)
class AsianOption : public Option {
public:
    explicit AsianOption(const OptionParameters& params);
    
    double payoff(const std::vector<double>& path) const override;
    double payoff(double S_T) const override; // Not applicable for path-dependent
};

// Geometric Asian option
class GeometricAsianOption : public Option {
public:
    explicit GeometricAsianOption(const OptionParameters& params);
    
    double payoff(const std::vector<double>& path) const override;
    double payoff(double S_T) const override; // Not applicable for path-dependent
};

// Barrier option
class BarrierOption : public Option {
public:
    BarrierOption(const OptionParameters& params, double barrier, BarrierType barrier_type);
    
    double payoff(const std::vector<double>& path) const override;
    double payoff(double S_T) const override; // Not applicable for path-dependent
    
private:
    double barrier_;
    BarrierType barrier_type_;
    
    bool is_barrier_crossed(const std::vector<double>& path) const;
};

} // namespace mc