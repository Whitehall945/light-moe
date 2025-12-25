# Contributing to Light-MoE

Thank you for your interest in contributing to Light-MoE! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment (see below)
4. Create a branch for your changes
5. Make your changes and test them
6. Submit a pull request

## Development Setup

### Prerequisites

- CUDA 12.0+
- Python 3.10+
- CMake 3.18+
- GCC 9+ or Clang 10+

### Build from Source

```bash
# Clone with submodules
git clone --recursive https://github.com/YOUR_USERNAME/light-moe.git
cd light-moe

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Build C++ components
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### Running Tests

```bash
# Python tests
pytest tests/python -v

# C++ tests
cd build && ctest --output-on-failure
```

## Code Style

### C++/CUDA

We follow the Google C++ Style Guide with some modifications. Key points:

- Use `.cu` for CUDA source files
- Use `.cuh` for CUDA headers
- Use `clang-format` for formatting (configuration in `.clang-format`)

```bash
# Format C++ code
find src include -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.cuh" | xargs clang-format -i
```

### Python

We use `ruff` for linting and `black` for formatting:

```bash
# Lint
ruff check python/

# Format
black python/
```

## Pull Request Process

1. **Create an Issue First**: For significant changes, please open an issue to discuss your proposal before starting work.

2. **Branch Naming**: Use descriptive branch names:
   - `feature/grouped-gemm-optimization`
   - `fix/memory-leak-in-dispatcher`
   - `docs/cute-tutorial`

3. **Commit Messages**: Write clear, concise commit messages:
   ```
   [component] Brief description of change
   
   Longer explanation if needed. Wrap at 72 characters.
   ```

4. **Testing**: Ensure all tests pass and add new tests for new functionality.

5. **Documentation**: Update documentation to reflect any changes.

6. **Code Review**: Address all feedback from reviewers.

## Reporting Issues

When reporting issues, please include:

- Operating system and version
- CUDA version and GPU model
- Python version
- Steps to reproduce the issue
- Expected vs actual behavior
- Any relevant logs or error messages

## Areas for Contribution

We especially welcome contributions in:

- **CuTe Operators**: New or optimized CUDA kernels
- **Benchmarks**: Performance comparisons and analysis
- **Documentation**: Tutorials, API docs, and examples
- **Testing**: Improved test coverage
- **Bug Fixes**: Identified issues from the tracker

Thank you for contributing to Light-MoE!
