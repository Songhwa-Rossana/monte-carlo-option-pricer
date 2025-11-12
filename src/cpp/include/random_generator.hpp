#pragma once

#include "common.hpp"

namespace mc {

class RandomGenerator {
public:
    RandomGenerator(unsigned int seed = std::random_device{}());
    
    // Generate standard normal random variable
    double normal();
    
    // Generate uniform random variable [0, 1)
    double uniform();
    
    // Generate vector of normal random variables
    void normal_vector(std::vector<double>& out, size_t n);
    
    // Reset with new seed
    void reset(unsigned int seed);
    
private:
    std::mt19937_64 gen_;
    std::normal_distribution<double> normal_dist_;
    std::uniform_real_distribution<double> uniform_dist_;
};

} // namespace mc