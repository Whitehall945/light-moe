# Light-MoE

<p align="center">
  <b>基于 CuTe 的高性能流水线并行 MoE 推理引擎</b>
</p>

<p align="center">
  <a href="#特性">特性</a> •
  <a href="#安装">安装</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#架构">架构</a> •
  <a href="#性能测试">性能测试</a> •
  <a href="README.md">English</a>
</p>

---

## 概述

**Light-MoE** 是一个专为混合专家模型（MoE）设计的高性能分布式推理引擎。它针对 MoE 推理的核心挑战：动态负载不均衡和 All-to-All 通信开销。

与 vLLM 等通用框架专注于 KV Cache 管理不同，Light-MoE 通过以下方式解决**专家瓶颈**问题：
- 基于 CuTe 的自定义算子，最大化硬件利用率
- 异步通信-计算重叠
- 动态 Token 分发与负载均衡

## 特性

### 算子层（CuTe/CUTLASS）
- 🔥 **融合 MoE Gate + TopK**：通过算子融合减少全局内存访问
- ⚡ **Grouped GEMM**：支持动态 Shape 的专家计算，优化 Tensor Core 利用率
- 🎯 **W4A16 量化 GEMM**：仅权重 INT4 量化，适用于显存受限环境

### 架构层
- 🌐 **专家并行（EP）**：灵活的专家 GPU 放置策略
- 🔄 **通信-计算重叠**：通过流水线执行隐藏 All-to-All 延迟
- ⚖️ **动态负载均衡**：专家间实时工作负载分配

## 安装

### 环境要求
- CUDA 12.0+
- Python 3.10+
- PyTorch 2.0+
- NCCL 2.18+

### 从源码安装
```bash
git clone https://github.com/YOUR_USERNAME/light-moe.git
cd light-moe
git submodule update --init --recursive

# 安装 Python 包
pip install -e .

# 或仅构建 C++ 库
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 快速开始

```python
from light_moe import LightMoEEngine

# 使用 8 张 GPU 初始化引擎
engine = LightMoEEngine(
    model_path="path/to/mixtral-8x7b",
    tensor_parallel_size=1,
    expert_parallel_size=8,
)

# 运行推理
output = engine.generate(
    prompt="解释相对论",
    max_tokens=512,
)
print(output)
```

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Light-MoE 引擎                            │
├─────────────────────┬───────────────────────────────────────────┤
│   Python 前端       │              C++/CUDA 核心                 │
├─────────────────────┼───────────────────────────────────────────┤
│  • 调度器           │  • CuTe 算子（Grouped GEMM、Gate）         │
│  • 模型加载器       │  • NCCL 通信层                             │
│  • API 服务器       │  • 异步分发器                              │
└─────────────────────┴───────────────────────────────────────────┘
```

## 性能测试

| 配置 | 模型 | 吞吐量 (tokens/s) | 提升 |
|-----|------|------------------|------|
| 8x 2080 Ti | Mixtral-8x7B | 待测 | 对比基线待测 |

## 开发路线

- [ ] 阶段一：核心 CuTe 算子（Grouped GEMM、融合 Gate）
- [ ] 阶段二：分布式基础设施（EP、All-to-All）
- [ ] 阶段三：端到端集成与性能测试

## 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献指南。

## 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE)。

## 致谢

- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) 提供的 CuTe 框架
- [vLLM](https://github.com/vllm-project/vllm) 在推理引擎设计上的启发
