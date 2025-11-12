# Building the C++ Monte Carlo Pricer

## Requirements
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.15+
- OpenMP (optional, for multi-threading)
- pybind11 (optional, for Python bindings)

## Build Steps

### Linux/macOS

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Install (optional)
sudo make install
```

### With Python bindings

```bash
# Install pybind11
pip install pybind11

# Configure with Python bindings
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON

# Build
make -j$(nproc)

# The Python module will be in build/monte_carlo_cpp*.so
```

### Windows (Visual Studio)

```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019"
cmake --build . --config Release
```

## Usage

### C++ Example

```cpp
#include "options.hpp"
#include "monte_carlo.hpp"

int main() {
    using namespace mc;
    
    OptionParameters params(100.0, 100.0, 1.0, 0.05, 0.2, OptionType::CALL);
    EuropeanOption option(params);
    
    MonteCarloEngine engine(100000, 252);
    auto result = engine.price(option);
    
    std::cout << "Price: " << result.price << " ± " << result.std_error << std::endl;
    
    return 0;
}
```

### Python Example

```python
import monte_carlo_cpp as mc

params = mc.OptionParameters(100.0, 100.0, 1.0, 0.05, 0.2, mc.OptionType.CALL)
option = mc.EuropeanOption(params)

engine = mc.MonteCarloEngine(100000, 252)
result = engine.price(option)

print(f"Price: {result.price:.4f} ± {result.std_error:.4f}")
```