# Getting Started with Light-MoE

This guide will help you set up and run Light-MoE for the first time.

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with Compute Capability 7.5+ (Turing or newer)
  - Recommended: RTX 2080 Ti, RTX 3090, A100, H100
  - For multi-GPU: 2+ GPUs with NCCL support

- **Memory**: 
  - System RAM: 32GB+ recommended
  - GPU Memory: 11GB+ per GPU

### Software Requirements

- CUDA 12.0+
- Python 3.10+
- PyTorch 2.0+
- GCC 9+ or Clang 10+
- CMake 3.18+

## Installation

### Option 1: Install from Source (Recommended)

```bash
# Clone repository with submodules
git clone --recursive https://github.com/YOUR_USERNAME/light-moe.git
cd light-moe

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python package in editable mode
pip install -e ".[dev]"
```

### Option 2: Build C++ Library Only

```bash
git clone --recursive https://github.com/YOUR_USERNAME/light-moe.git
cd light-moe

# Build with CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Quick Start

### Single GPU Inference

```python
from light_moe import LightMoEEngine

# Initialize engine
engine = LightMoEEngine(
    model_path="path/to/mixtral-8x7b",
    tensor_parallel_size=1,
    expert_parallel_size=1,
)

# Generate text
output = engine.generate(
    prompt="Explain quantum computing in simple terms:",
    max_tokens=256,
    temperature=0.7,
)
print(output)
```

### Multi-GPU with Expert Parallelism

```bash
# Launch with torchrun for 8 GPUs
torchrun --nproc_per_node=8 examples/multi_gpu_demo.py \
    --model_path path/to/mixtral-8x7b \
    --expert_parallel_size 8
```

## Verifying Installation

Run the test suite to verify everything is working:

```bash
# Python tests
pytest tests/python -v

# Build and run C++ tests (requires CUDA)
cd build && ctest --output-on-failure
```

## Next Steps

- Read the [Architecture Guide](architecture.md) to understand the system design
- Explore [CuTe programming](cute_guide.md) for kernel development
- Check out [examples/](../examples/) for more usage patterns

## Troubleshooting

### CUDA Not Found

Ensure CUDA is properly installed and `nvcc` is in your PATH:

```bash
nvcc --version
```

### CUTLASS Submodule Missing

If you cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### Import Errors

Make sure you've installed the package:

```bash
pip install -e .
```
