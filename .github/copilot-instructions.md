# GitHub Copilot Instructions for Reactant.jl

## Project Overview

Reactant.jl is a Julia package that compiles Julia functions into MLIR (Multi-Level Intermediate Representation) and runs advanced optimizations, including automatic differentiation with EnzymeMLIR, to create executables for CPU/GPU/TPU via XLA. It operates as a tracing system for high-performance computing.

## Language and Code Style

### Julia Code
- **Language**: Julia 1.10+
- **Code Style**: Use [Blue style](https://github.com/JuliaDiff/BlueStyle) as enforced by JuliaFormatter
- **Formatter Config**: See `.JuliaFormatter.toml` with `style = "blue"` and `always_use_return = true`
- **Type Stability**: Maintain type stability for performance-critical code
- **Broadcasting**: Prefer Julia's dot notation for array operations (e.g., `sin.(x) .+ y`)
- **Naming Conventions**: Use descriptive names; types are CamelCase, functions and variables are snake_case

### C++ Code
- **Style**: LLVM style as specified in `.clang-format`
- **Location**: C++ code lives in `deps/ReactantExtra/`
- **Build System**: Bazel is used for building C++ components

## Project Structure

```
├── src/              # Main Julia source code
│   ├── Reactant.jl   # Main module file
│   ├── TracedRArray.jl, ConcreteRArray.jl  # Core array types
│   ├── Compiler.jl   # Compilation infrastructure
│   ├── Enzyme.jl     # Automatic differentiation integration
│   ├── mlir/         # MLIR bindings and utilities
│   └── xla/          # XLA integration
├── ext/              # Package extensions (conditional loading)
├── lib/              # Sub-packages
│   └── ReactantCore/ # Core functionality
├── deps/             # Build dependencies and C++ code
│   └── ReactantExtra/ # C++ API and Bazel build files
├── test/             # Test suite
│   ├── runtests.jl   # Main test runner
│   ├── integration/  # Integration tests
│   └── nn/           # Neural network tests
└── docs/             # Documentation
```

## Core Concepts

### Array Types
- **ConcreteRArray**: Underlying buffer for device data (CPU/GPU/TPU)
- **TracedRArray**: Traced version used during compilation (no access to actual values)
- Conversion: Use `Reactant.to_rarray()` to convert Julia arrays to RArrays

### Compilation
- Use `@compile` macro to compile functions
- Compiled functions capture control flow at compile time
- Only ConcreteRArray updates are captured in compiled code

### Backends
- Default backend can be set with `Reactant.set_default_backend("cpu"/"gpu"/"tpu")`
- Supports CPU, GPU (CUDA), and TPU via XLA

## Testing

### Test Structure
- **Framework**: Uses `SafeTestsets` and Julia's built-in `Test`
- **Test Groups**: Tests are organized into three groups:
  - `core`: Basic functionality, tracing, compilation, autodiff
  - `neural_networks`: NNlib, Flux, LuxLib, Lux integration
  - `integration`: CUDA, KernelAbstractions, FFT, MPI, etc.
- **Backend Testing**: Tests can run with different backends (CPU/GPU)
- **Runtime Testing**: Tests run with both "pjrt" and "ifrt" runtimes

### Running Tests
```bash
# Run all tests
julia --project=. test/runtests.jl

# Run specific test group
REACTANT_TEST_GROUP=core julia --project=. test/runtests.jl

# Run with GPU backend
REACTANT_BACKEND_GROUP=gpu julia --project=. test/runtests.jl
```

### Writing Tests
- Use `@safetestset` for isolated test environments
- Follow existing patterns in test files
- Test both forward and reverse-mode automatic differentiation where applicable
- Include edge cases and type stability checks

## CI/CD

### Workflows
- **CI**: Main test suite runs on Julia 1.10, 1.11 on multiple platforms (Ubuntu, macOS, Windows, ARM, TPU)
- **Format Check**: Enforces Julia code style via JuliaFormatter
- **Format Check (C++)**: Enforces LLVM style for C++ code via clang-format
- **Format Check (Bazel)**: Enforces Bazel file formatting with buildifier
- **Documentation**: Builds with Documenter.jl and DocumenterVitepress
- **Benchmarks**: Performance tracking on push to main

### Continuous Integration
- Tests run on push to `main` and `release-*` branches
- PRs trigger CI on relevant file changes
- Concurrency controls prevent redundant builds

## Dependencies and Extensions

### Core Dependencies
- **EnzymeCore/Enzyme**: Automatic differentiation
- **LLVM.jl**: LLVM integration
- **Functors.jl**: Recursive structure traversal
- **Adapt.jl**: Array type adaptation

### Package Extensions
Reactant uses Julia's package extensions for optional integrations:
- CUDA, KernelAbstractions for GPU computing
- NNlib for neural network primitives
- Zygote for alternative AD (Julia < 1.12)
- MPI for distributed computing
- AbstractFFTs, SpecialFunctions, etc.

### Adding Dependencies
- Update `Project.toml` [deps] for required dependencies
- Add to [weakdeps] and create extension in `ext/` for optional dependencies
- Specify version bounds in [compat] section

## Development Workflow

### Setting Up Development Environment
```bash
# Clone the repository
git clone https://github.com/EnzymeAD/Reactant.jl.git
cd Reactant.jl

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Build (if needed)
julia --project=. deps/build_local.jl
```

### Code Formatting
```bash
# Format Julia code
julia --project=. -e 'using JuliaFormatter; format(".")'

# Format C++ code (requires clang-format)
find deps/ReactantExtra -name "*.cpp" -o -name "*.h" | xargs clang-format -i

# Format Bazel files (requires buildifier)
buildifier -r deps/ReactantExtra/
```

### Documentation
- Documentation is built with Documenter.jl and DocumenterVitepress
- Docstrings should follow Julia conventions
- Run `julia --project=docs docs/make.jl` to build locally

## Best Practices

### Performance
- Minimize allocations in hot loops
- Use `@inbounds` carefully when bounds are guaranteed
- Prefer type-stable code; avoid type unions and `Any`
- Use `@compile` to generate optimized executables for production code

### Automatic Differentiation
- Test both forward and reverse mode when adding new operations
- Ensure proper handling of mutation and aliasing
- Use EnzymeCore for defining custom derivatives when needed

### MLIR/XLA Integration
- MLIR operations are in `src/mlir/` and follow Dialect conventions
- XLA integration code is in `src/xla/`
- C++ API changes require updating Bazel build files and Julia bindings

### Error Handling
- Provide informative error messages
- Use Julia's exception system appropriately
- Document expected failure modes in docstrings

### Compatibility
- Maintain compatibility with Julia 1.10+
- Follow [ColPrac](https://github.com/SciML/ColPrac) guidelines
- Keep dependencies up-to-date via Dependabot

## Common Tasks

### Adding a New Operation
1. Implement the Julia function in appropriate file in `src/`
2. Add tracing support if needed in `src/Tracing.jl`
3. Add tests in `test/ops.jl` or appropriate test file
4. Add documentation if it's a public API
5. Format code with JuliaFormatter

### Adding a New Test
1. Create test file in `test/` (or appropriate subdirectory)
2. Add to `test/runtests.jl` with `@safetestset`
3. Group appropriately (core/integration/neural_networks)
4. Run tests locally before submitting PR

### Updating MLIR Bindings
1. Modify `deps/ReactantExtra/API.cpp` if needed
2. Update Bazel BUILD files
3. Run regeneration workflow or script
4. Test thoroughly with existing test suite

## Resources

- [Documentation](https://enzymead.github.io/Reactant.jl/dev)
- [Issue Tracker](https://github.com/EnzymeAD/Reactant.jl/issues)
- [Contributing Guide](https://github.com/SciML/ColPrac)
- [Enzyme Documentation](https://enzyme.mit.edu/)
- [XLA Documentation](https://www.tensorflow.org/xla)

## Questions?

For questions or clarifications, open an issue or discussion on GitHub. The maintainers actively monitor the repository and are happy to help!
