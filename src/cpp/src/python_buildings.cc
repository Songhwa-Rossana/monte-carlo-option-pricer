#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "common.hpp"
#include "options.hpp"
#include "monte_carlo.hpp"
#include "greeks.hpp"
#include "variance_reduction.hpp"

namespace py = pybind11;
using namespace mc;

PYBIND11_MODULE(monte_carlo_cpp, m) {
    m.doc() = "High-performance Monte Carlo option pricing engine";
    
    // Enums
    py::enum_<OptionType>(m, "OptionType")
        .value("CALL", OptionType::CALL)
        .value("PUT", OptionType::PUT);
    
    py::enum_<BarrierType>(m, "BarrierType")
        .value("UP_IN", BarrierType::UP_IN)
        .value("UP_OUT", BarrierType::UP_OUT)
        .value("DOWN_IN", BarrierType::DOWN_IN)
        .value("DOWN_OUT", BarrierType::DOWN_OUT);
    
    // Structures
    py::class_<OptionParameters>(m, "OptionParameters")
        .def(py::init<double, double, double, double, double, OptionType>(),
             py::arg("S0"), py::arg("K"), py::arg("T"), 
             py::arg("r"), py::arg("sigma"), py::arg("type"))
        .def_readwrite("S0", &OptionParameters::S0)
        .def_readwrite("K", &OptionParameters::K)
        .def_readwrite("T", &OptionParameters::T)
        .def_readwrite("r", &OptionParameters::r)
        .def_readwrite("sigma", &OptionParameters::sigma)
        .def_readwrite("type", &OptionParameters::type);
    
    py::class_<PricingResult>(m, "PricingResult")
        .def(py::init<>())
        .def_readwrite("price", &PricingResult::price)
        .def_readwrite("std_error", &PricingResult::std_error)
        .def_readwrite("confidence_lower", &PricingResult::confidence_lower)
        .def_readwrite("confidence_upper", &PricingResult::confidence_upper);
    
    py::class_<Greeks>(m, "Greeks")
        .def(py::init<>())
        .def_readwrite("delta", &Greeks::delta)
        .def_readwrite("gamma", &Greeks::gamma)
        .def_readwrite("vega", &Greeks::vega)
        .def_readwrite("theta", &Greeks::theta)
        .def_readwrite("rho", &Greeks::rho);
    
    // Options
    py::class_<Option, std::shared_ptr<Option>>(m, "Option")
        .def("payoff", py::overload_cast<const std::vector<double>&>(&Option::payoff, py::const_))
        .def("params", &Option::params);
    
    py::class_<EuropeanOption, Option, std::shared_ptr<EuropeanOption>>(m, "EuropeanOption")
        .def(py::init<const OptionParameters&>())
        .def("black_scholes_price", &EuropeanOption::black_scholes_price)
        .def("black_scholes_delta", &EuropeanOption::black_scholes_delta)
        .def("black_scholes_gamma", &EuropeanOption::black_scholes_gamma)
        .def("black_scholes_vega", &EuropeanOption::black_scholes_vega)
        .def("black_scholes_theta", &EuropeanOption::black_scholes_theta)
        .def("black_scholes_rho", &EuropeanOption::black_scholes_rho);
    
    py::class_<AsianOption, Option, std::shared_ptr<AsianOption>>(m, "AsianOption")
        .def(py::init<const OptionParameters&>());
    
    py::class_<GeometricAsianOption, Option, std::shared_ptr<GeometricAsianOption>>(m, "GeometricAsianOption")
        .def(py::init<const OptionParameters&>());
    
    py::class_<BarrierOption, Option, std::shared_ptr<BarrierOption>>(m, "BarrierOption")
        .def(py::init<const OptionParameters&, double, BarrierType>(),
             py::arg("params"), py::arg("barrier"), py::arg("barrier_type"));
    
    // Monte Carlo Engine
    py::class_<MonteCarloEngine>(m, "MonteCarloEngine")
        .def(py::init<size_t, size_t, unsigned int>(),
             py::arg("n_paths"), py::arg("n_steps"), py::arg("seed") = 42)
        .def("price", &MonteCarloEngine::price,
             py::arg("option"), py::arg("use_antithetic") = false)
        .def("price_with_control_variate", &MonteCarloEngine::price_with_control_variate)
        .def("set_n_paths", &MonteCarloEngine::set_n_paths)
        .def("set_n_steps", &MonteCarloEngine::set_n_steps)
        .def("set_seed", &MonteCarloEngine::set_seed)
        .def_property_readonly("n_paths", &MonteCarloEngine::n_paths)
        .def_property_readonly("n_steps", &MonteCarloEngine::n_steps);
    
    // Greeks Calculator
    py::class_<GreeksCalculator>(m, "GreeksCalculator")
        .def(py::init<MonteCarloEngine&, double>(),
             py::arg("engine"), py::arg("bump_size") = 0.01)
        .def("calculate", &GreeksCalculator::calculate)
        .def("delta", &GreeksCalculator::delta)
        .def("gamma", &GreeksCalculator::gamma)
        .def("vega", &GreeksCalculator::vega)
        .def("theta", &GreeksCalculator::theta)
        .def("rho", &GreeksCalculator::rho);
    
    // Variance reduction functions
    m.def("price_antithetic", &price_antithetic);
    m.def("price_control_variate", &price_control_variate);
}
