# Light-MoE

<p align="center">
  <b>High-Performance Pipeline-Parallel MoE Inference Engine with CuTe</b>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="README_CN.md">中文文档</a>
</p>

---

## Overview

**Light-MoE** is a high-performance distributed inference engine specifically designed for Mixture-of-Experts (MoE) models. It addresses the unique challenges of MoE inference: dynamic load imbalance and All-to-All communication overhead.

Unlike general-purpose frameworks like vLLM that focus on KV cache management, Light-MoE targets the **expert bottleneck** through:
- Custom CuTe-based operators for maximum hardware utilization
- Asynchronous communication-computation overlap
- Dynamic token dispatching with load balancing

## Features

### Operator Level (CuTe/CUTLASS)
- 🔥 **Fused MoE Gate + TopK**: Reduces global memory access through kernel fusion
- ⚡ **Grouped GEMM**: Dynamic shape support for expert computation with optimized Tensor Core utilization
- 🎯 **W4A16 Quantized GEMM**: Weight-only INT4 quantization for memory-constrained environments

### Infrastructure Level
- 🌐 **Expert Parallelism (EP)**: Flexible expert placement across GPUs
- 🔄 **Communication-Computation Overlap**: Hide All-to-All latency with pipelined execution
- ⚖️ **Dynamic Load Balancing**: Real-time workload distribution across experts

## Installation

### Prerequisites
- CUDA 12.0+
- Python 3.10+
- PyTorch 2.0+
- NCCL 2.18+

### From Source
```bash
git clone https://github.com/YOUR_USERNAME/light-moe.git
cd light-moe
git submodule update --init --recursive

# Install Python package
pip install -e .

# Or build C++ library only
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Quick Start

```python
from light_moe import LightMoEEngine

# Initialize engine with 8 GPUs
engine = LightMoEEngine(
    model_path="path/to/mixtral-8x7b",
    tensor_parallel_size=1,
    expert_parallel_size=8,
)

# Run inference
output = engine.generate(
    prompt="Explain the theory of relativity",
    max_tokens=512,
)
print(output)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Light-MoE Engine                         │
├─────────────────────┬───────────────────────────────────────────┤
│   Python Frontend   │              C++/CUDA Core                │
├─────────────────────┼───────────────────────────────────────────┤
│  • Scheduler        │  • CuTe Operators (Grouped GEMM, Gate)    │
│  • Model Loader     │  • NCCL Communication Layer               │
│  • API Server       │  • Async Dispatcher                       │
└─────────────────────┴───────────────────────────────────────────┘
```

### Directory Structure
```
light-moe/
├── include/          # C++ public headers
├── src/              # C++/CUDA source code
│   ├── ops/cute/     # CuTe-based operators
│   └── comm/         # Communication layer
├── python/           # Python bindings and frontend
├── tests/            # Unit and integration tests
├── benchmarks/       # Performance benchmarks
└── docs/             # Documentation
```

## Benchmarks

| Configuration | Model | Throughput (tokens/s) | Improvement |
|--------------|-------|----------------------|-------------|
| 8x 2080 Ti   | Mixtral-8x7B | TBD | TBD vs baseline |

## Roadmap

- [ ] Phase 1: Core CuTe operators (Grouped GEMM, Fused Gate)
- [ ] Phase 2: Distributed infrastructure (EP, All-to-All)
- [ ] Phase 3: End-to-end integration and benchmarking

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) for the CuTe framework
- [vLLM](https://github.com/vllm-project/vllm) for inspiration on inference engine design
