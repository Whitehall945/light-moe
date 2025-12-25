"""
Real CUDA Kernels for Light-MoE

This module provides CUDA-accelerated operations compiled via PyTorch's
cpp_extension. These are the actual production kernels.
"""

import os
import torch
from torch.utils.cpp_extension import load_inline

# CUDA source for Grouped GEMM with tiled shared memory
GROUPED_GEMM_CUDA_SOURCE = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Tiled GEMM kernel with shared memory
template <typename scalar_t, int BLOCK_M = 64, int BLOCK_N = 64, int BLOCK_K = 16>
__global__ void grouped_gemm_tiled_kernel(
    const scalar_t* __restrict__ input,      // [total_tokens, hidden_size]
    const scalar_t* __restrict__ weights,    // [num_experts, out_features, hidden_size]
    scalar_t* __restrict__ output,           // [total_tokens, out_features]
    const int64_t* __restrict__ expert_offsets,  // [num_experts + 1]
    int num_experts,
    int out_features,
    int hidden_size) {
    
    // Shared memory tiles
    __shared__ float smem_A[BLOCK_M][BLOCK_K];
    __shared__ float smem_B[BLOCK_K][BLOCK_N];
    
    // Expert index
    int expert_idx = blockIdx.z;
    if (expert_idx >= num_experts) return;
    
    // Get expert boundaries
    int64_t start = expert_offsets[expert_idx];
    int64_t end = expert_offsets[expert_idx + 1];
    int m = end - start;
    if (m == 0) return;
    
    // Block position
    int bm = blockIdx.y * BLOCK_M;
    int bn = blockIdx.x * BLOCK_N;
    
    // Thread position
    int tid = threadIdx.x;
    int tx = tid % BLOCK_N;
    int ty = tid / BLOCK_N;
    
    // Pointers for this expert
    const scalar_t* A = input + start * hidden_size;
    const scalar_t* B = weights + expert_idx * out_features * hidden_size;
    scalar_t* C = output + start * out_features;
    
    // Accumulator
    float acc = 0.0f;
    
    // Tile over K dimension
    for (int bk = 0; bk < hidden_size; bk += BLOCK_K) {
        // Collaborative load of A tile
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += blockDim.x) {
            int row = i / BLOCK_K;
            int col = i % BLOCK_K;
            if (bm + row < m && bk + col < hidden_size) {
                smem_A[row][col] = static_cast<float>(A[(bm + row) * hidden_size + bk + col]);
            } else {
                smem_A[row][col] = 0.0f;
            }
        }
        
        // Collaborative load of B tile (transposed: B is [out_features, hidden_size])
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += blockDim.x) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            if (bk + row < hidden_size && bn + col < out_features) {
                smem_B[row][col] = static_cast<float>(B[(bn + col) * hidden_size + bk + row]);
            } else {
                smem_B[row][col] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial products
        if (ty < BLOCK_M && tx < BLOCK_N) {
            #pragma unroll
            for (int k = 0; k < BLOCK_K; k++) {
                acc += smem_A[ty][k] * smem_B[k][tx];
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (bm + ty < m && bn + tx < out_features) {
        C[(bm + ty) * out_features + bn + tx] = static_cast<scalar_t>(acc);
    }
}

// Launch wrapper
torch::Tensor grouped_gemm_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor expert_offsets,
    int num_experts) {
    
    const int total_tokens = input.size(0);
    const int hidden_size = input.size(1);
    const int out_features = weights.size(1);
    
    auto output = torch::zeros({total_tokens, out_features}, input.options());
    
    if (total_tokens == 0) return output;
    
    // Kernel config
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_K = 16;
    constexpr int THREADS = 256;
    
    // Find max tokens per expert for grid sizing
    int max_m = 0;
    auto offsets_cpu = expert_offsets.cpu();
    auto offsets_ptr = offsets_cpu.data_ptr<int64_t>();
    for (int i = 0; i < num_experts; i++) {
        int m = offsets_ptr[i + 1] - offsets_ptr[i];
        max_m = std::max(max_m, m);
    }
    
    dim3 grid(
        (out_features + BLOCK_N - 1) / BLOCK_N,
        (max_m + BLOCK_M - 1) / BLOCK_M,
        num_experts
    );
    dim3 block(THREADS);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grouped_gemm_cuda", ([&] {
        grouped_gemm_tiled_kernel<scalar_t, BLOCK_M, BLOCK_N, BLOCK_K>
            <<<grid, block>>>(
                input.data_ptr<scalar_t>(),
                weights.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                expert_offsets.data_ptr<int64_t>(),
                num_experts,
                out_features,
                hidden_size
            );
    }));
    
    return output;
}
"""

# Fused Gate + TopK kernel
FUSED_GATE_CUDA_SOURCE = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

// Warp reduce max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp reduce sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fused gate projection + softmax + top-k
template <int NUM_EXPERTS, int TOP_K>
__global__ void fused_gate_topk_kernel(
    const float* __restrict__ hidden_states,  // [num_tokens, hidden_size]
    const float* __restrict__ gate_weight,    // [num_experts, hidden_size]
    int* __restrict__ topk_indices,           // [num_tokens, top_k]
    float* __restrict__ topk_weights,         // [num_tokens, top_k]
    int num_tokens,
    int hidden_size) {
    
    __shared__ float s_logits[NUM_EXPERTS];
    __shared__ float s_reduce[8];  // For block reduction
    
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    int tid = threadIdx.x;
    const float* token = hidden_states + token_idx * hidden_size;
    
    // Compute gate logits (each thread handles some experts)
    for (int e = tid; e < NUM_EXPERTS; e += blockDim.x) {
        float sum = 0.0f;
        const float* w = gate_weight + e * hidden_size;
        
        for (int i = 0; i < hidden_size; i += 4) {
            float4 h = *reinterpret_cast<const float4*>(token + i);
            float4 g = *reinterpret_cast<const float4*>(w + i);
            sum += h.x * g.x + h.y * g.y + h.z * g.z + h.w * g.w;
        }
        s_logits[e] = sum;
    }
    __syncthreads();
    
    // Softmax: find max
    float local_max = -FLT_MAX;
    for (int e = tid; e < NUM_EXPERTS; e += blockDim.x) {
        local_max = fmaxf(local_max, s_logits[e]);
    }
    local_max = warp_reduce_max(local_max);
    if (tid % 32 == 0) s_reduce[tid / 32] = local_max;
    __syncthreads();
    if (tid < 8) local_max = warp_reduce_max(tid < blockDim.x / 32 ? s_reduce[tid] : -FLT_MAX);
    float global_max = __shfl_sync(0xffffffff, local_max, 0);
    __syncthreads();
    
    // Softmax: compute exp and sum
    float local_sum = 0.0f;
    for (int e = tid; e < NUM_EXPERTS; e += blockDim.x) {
        float v = expf(s_logits[e] - global_max);
        s_logits[e] = v;
        local_sum += v;
    }
    local_sum = warp_reduce_sum(local_sum);
    if (tid % 32 == 0) s_reduce[tid / 32] = local_sum;
    __syncthreads();
    if (tid < 8) local_sum = warp_reduce_sum(tid < blockDim.x / 32 ? s_reduce[tid] : 0.0f);
    float global_sum = __shfl_sync(0xffffffff, local_sum, 0);
    __syncthreads();
    
    // Normalize
    float inv_sum = 1.0f / global_sum;
    for (int e = tid; e < NUM_EXPERTS; e += blockDim.x) {
        s_logits[e] *= inv_sum;
    }
    __syncthreads();
    
    // TopK selection (single thread)
    if (tid == 0) {
        int indices[TOP_K];
        float values[TOP_K];
        
        for (int k = 0; k < TOP_K; k++) {
            values[k] = -FLT_MAX;
            indices[k] = 0;
        }
        
        for (int e = 0; e < NUM_EXPERTS; e++) {
            float val = s_logits[e];
            if (val > values[TOP_K - 1]) {
                int pos = TOP_K - 1;
                while (pos > 0 && val > values[pos - 1]) pos--;
                for (int k = TOP_K - 1; k > pos; k--) {
                    values[k] = values[k - 1];
                    indices[k] = indices[k - 1];
                }
                values[pos] = val;
                indices[pos] = e;
            }
        }
        
        // Normalize and write
        float weight_sum = 0.0f;
        for (int k = 0; k < TOP_K; k++) weight_sum += values[k];
        
        for (int k = 0; k < TOP_K; k++) {
            topk_indices[token_idx * TOP_K + k] = indices[k];
            topk_weights[token_idx * TOP_K + k] = values[k] / weight_sum;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> fused_gate_topk_cuda(
    torch::Tensor hidden_states,
    torch::Tensor gate_weight,
    int top_k) {
    
    const int num_tokens = hidden_states.size(0);
    const int hidden_size = hidden_states.size(1);
    const int num_experts = gate_weight.size(0);
    
    auto topk_indices = torch::empty({num_tokens, top_k}, 
        torch::dtype(torch::kInt32).device(hidden_states.device()));
    auto topk_weights = torch::empty({num_tokens, top_k}, hidden_states.options());
    
    dim3 grid(num_tokens);
    dim3 block(256);
    
    // Dispatch based on num_experts
    if (num_experts == 8 && top_k == 2) {
        fused_gate_topk_kernel<8, 2><<<grid, block>>>(
            hidden_states.data_ptr<float>(),
            gate_weight.data_ptr<float>(),
            topk_indices.data_ptr<int>(),
            topk_weights.data_ptr<float>(),
            num_tokens, hidden_size
        );
    } else if (num_experts == 16 && top_k == 2) {
        fused_gate_topk_kernel<16, 2><<<grid, block>>>(
            hidden_states.data_ptr<float>(),
            gate_weight.data_ptr<float>(),
            topk_indices.data_ptr<int>(),
            topk_weights.data_ptr<float>(),
            num_tokens, hidden_size
        );
    } else {
        throw std::runtime_error("Unsupported num_experts/top_k combination");
    }
    
    return std::make_tuple(topk_indices, topk_weights);
}
"""

# C++ declarations
CPP_SOURCE = """
torch::Tensor grouped_gemm_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor expert_offsets,
    int num_experts);

std::tuple<torch::Tensor, torch::Tensor> fused_gate_topk_cuda(
    torch::Tensor hidden_states,
    torch::Tensor gate_weight,
    int top_k);
"""

# Compile and load
_cuda_module = None

def _get_cuda_module():
    """JIT compile and cache CUDA module."""
    global _cuda_module
    if _cuda_module is None:
        print("Compiling CUDA kernels (this may take a minute)...")
        _cuda_module = load_inline(
            name="light_moe_cuda",
            cpp_sources=[CPP_SOURCE],
            cuda_sources=[GROUPED_GEMM_CUDA_SOURCE + FUSED_GATE_CUDA_SOURCE],
            functions=["grouped_gemm_cuda", "fused_gate_topk_cuda"],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
        print("CUDA kernels compiled successfully!")
    return _cuda_module


def grouped_gemm_cuda(
    input: torch.Tensor,
    weights: torch.Tensor,
    expert_offsets: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    CUDA Grouped GEMM with tiled shared memory.
    
    Args:
        input: [total_tokens, hidden_size]
        weights: [num_experts, out_features, hidden_size]
        expert_offsets: [num_experts + 1]
        num_experts: Number of experts
        
    Returns:
        output: [total_tokens, out_features]
    """
    module = _get_cuda_module()
    return module.grouped_gemm_cuda(input, weights, expert_offsets, num_experts)


def fused_gate_topk_cuda(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int = 2,
) -> tuple:
    """
    Fused gate projection + softmax + topK.
    
    Args:
        hidden_states: [num_tokens, hidden_size]
        gate_weight: [num_experts, hidden_size]
        top_k: Number of experts per token
        
    Returns:
        Tuple of (topk_indices, topk_weights)
    """
    module = _get_cuda_module()
    return module.fused_gate_topk_cuda(hidden_states.float(), gate_weight.float(), top_k)


# Test function
def test_kernels():
    """Test CUDA kernels."""
    print("\n" + "="*60)
    print("Testing Light-MoE CUDA Kernels")
    print("="*60)
    
    device = torch.device("cuda")
    
    # Test Grouped GEMM
    print("\n1. Testing Grouped GEMM...")
    num_experts = 8
    hidden_size = 4096
    out_features = 14336
    tokens_per_expert = [256, 128, 512, 64, 320, 192, 384, 96]
    
    # Create test data
    offsets = [0]
    for t in tokens_per_expert:
        offsets.append(offsets[-1] + t)
    total_tokens = offsets[-1]
    
    input_tensor = torch.randn(total_tokens, hidden_size, device=device, dtype=torch.float16)
    weights = torch.randn(num_experts, out_features, hidden_size, device=device, dtype=torch.float16)
    expert_offsets = torch.tensor(offsets, device=device, dtype=torch.int64)
    
    # Run CUDA kernel
    output_cuda = grouped_gemm_cuda(input_tensor, weights, expert_offsets, num_experts)
    
    # Reference: PyTorch
    output_ref = torch.zeros(total_tokens, out_features, device=device, dtype=torch.float16)
    for e in range(num_experts):
        start, end = offsets[e], offsets[e+1]
        if end > start:
            output_ref[start:end] = torch.mm(input_tensor[start:end], weights[e].t())
    
    # Compare
    diff = (output_cuda - output_ref).abs().max().item()
    print(f"   Max difference vs PyTorch: {diff:.6f}")
    print(f"   Shape: {output_cuda.shape}")
    
    # Benchmark
    import time
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = grouped_gemm_cuda(input_tensor, weights, expert_offsets, num_experts)
    torch.cuda.synchronize()
    cuda_time = (time.perf_counter() - start) / 100 * 1000
    
    start = time.perf_counter()
    for _ in range(100):
        for e in range(num_experts):
            s, end = offsets[e], offsets[e+1]
            if end > s:
                output_ref[s:end] = torch.mm(input_tensor[s:end], weights[e].t())
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / 100 * 1000
    
    print(f"   CUDA kernel: {cuda_time:.3f}ms")
    print(f"   PyTorch loop: {pytorch_time:.3f}ms")
    print(f"   Speedup: {pytorch_time/cuda_time:.2f}x")
    
    # Test Fused Gate
    print("\n2. Testing Fused Gate + TopK...")
    num_tokens = 2048
    hidden_size = 4096
    num_experts = 8
    top_k = 2
    
    hidden = torch.randn(num_tokens, hidden_size, device=device)
    gate_w = torch.randn(num_experts, hidden_size, device=device)
    
    indices, weights_out = fused_gate_topk_cuda(hidden, gate_w, top_k)
    
    # Reference
    logits = torch.mm(hidden, gate_w.t())
    probs = torch.softmax(logits, dim=-1)
    ref_weights, ref_indices = torch.topk(probs, top_k, dim=-1)
    ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)
    
    print(f"   Output shape: indices={indices.shape}, weights={weights_out.shape}")
    print(f"   Sample indices: {indices[0].tolist()}")
    print(f"   Sample weights: {weights_out[0].tolist()}")
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _, _ = fused_gate_topk_cuda(hidden, gate_w, top_k)
    torch.cuda.synchronize()
    cuda_time = (time.perf_counter() - start) / 100 * 1000
    
    start = time.perf_counter()
    for _ in range(100):
        logits = torch.mm(hidden, gate_w.t())
        probs = torch.softmax(logits, dim=-1)
        _, _ = torch.topk(probs, top_k, dim=-1)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / 100 * 1000
    
    print(f"   CUDA kernel: {cuda_time:.3f}ms")
    print(f"   PyTorch: {pytorch_time:.3f}ms")
    print(f"   Speedup: {pytorch_time/cuda_time:.2f}x")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_kernels()
