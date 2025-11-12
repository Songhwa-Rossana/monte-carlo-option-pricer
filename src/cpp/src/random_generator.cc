#include "random_generator.hpp"

namespace mc {

RandomGenerator::RandomGenerator(unsigned int seed)
    : gen_(seed), 
      normal_dist_(0.0, 1.0),
      uniform_dist_(0.0, 1.0) {
}

double RandomGenerator::normal() {
    return normal_dist_(gen_);
}

double RandomGenerator::uniform() {
    return uniform_dist_(gen_);
}

void RandomGenerator::normal_vector(std::vector<double>& out, size_t n) {
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = normal();
    }
}

void RandomGenerator::reset(unsigned int seed) {
    gen_.seed(seed);
}

} // namespace mc