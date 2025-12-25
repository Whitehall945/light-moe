/**
 * @file fused_moe_gate.cu
 * @brief Fused MoE Gate + TopK CUDA kernel
 * 
 * Combines gate projection, softmax, and TopK selection into a single
 * kernel to minimize memory bandwidth.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include "light_moe/ops/fused_gate.h"

namespace light_moe {
namespace ops {

// ============================================================================
// Constants
// ============================================================================

constexpr int kWarpSize = 32;
constexpr int kMaxExperts = 256;  // Maximum supported experts

// ============================================================================
// Helper Functions
// ============================================================================

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// Fused Gate + TopK Kernel
// ============================================================================

/**
 * @brief Fused kernel for gate projection + softmax + topk
 * 
 * Each thread block handles one token.
 * 1. Compute gate_logits = hidden @ gate_weight.T
 * 2. Compute softmax(gate_logits)
 * 3. Select top-k experts
 */
template <typename T, int kNumExperts, int kTopK>
__global__ void fused_gate_topk_kernel(
    const T* __restrict__ hidden_states,    // [num_tokens, hidden_size]
    const T* __restrict__ gate_weight,      // [num_experts, hidden_size]
    int* __restrict__ topk_indices,          // [num_tokens, top_k]
    T* __restrict__ topk_weights,            // [num_tokens, top_k]
    int* __restrict__ expert_counts,         // [num_experts]
    int num_tokens,
    int hidden_size,
    bool normalize) {
    
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    int tid = threadIdx.x;
    
    // Shared memory for gate logits and sorting
    __shared__ float s_logits[kNumExperts];
    __shared__ int s_indices[kNumExperts];
    
    // Step 1: Compute gate logits (simplified - real impl uses vectorized loads)
    // Each thread computes one expert's logit
    float logit = 0.0f;
    if (tid < kNumExperts) {
        const T* hidden = hidden_states + token_idx * hidden_size;
        const T* weight = gate_weight + tid * hidden_size;
        
        // Dot product (simplified - should use vectorized FMA)
        for (int i = 0; i < hidden_size; i++) {
            logit += float(hidden[i]) * float(weight[i]);
        }
        s_logits[tid] = logit;
        s_indices[tid] = tid;
    }
    __syncthreads();
    
    // Step 2: Softmax
    // Find max for numerical stability
    float max_logit = -INFINITY;
    for (int i = tid; i < kNumExperts; i += blockDim.x) {
        max_logit = max(max_logit, s_logits[i]);
    }
    max_logit = warp_reduce_max(max_logit);
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < kNumExperts; i += blockDim.x) {
        s_logits[i] = expf(s_logits[i] - max_logit);
        sum += s_logits[i];
    }
    sum = warp_reduce_sum(sum);
    
    // Normalize
    for (int i = tid; i < kNumExperts; i += blockDim.x) {
        s_logits[i] /= sum;
    }
    __syncthreads();
    
    // Step 3: TopK selection (simple O(n*k) for small k)
    // TODO: Use more efficient parallel topk for larger k
    if (tid == 0) {
        for (int k = 0; k < kTopK; k++) {
            int max_idx = 0;
            float max_val = -1.0f;
            
            for (int i = 0; i < kNumExperts; i++) {
                if (s_logits[i] > max_val) {
                    max_val = s_logits[i];
                    max_idx = i;
                }
            }
            
            topk_indices[token_idx * kTopK + k] = max_idx;
            topk_weights[token_idx * kTopK + k] = T(max_val);
            s_logits[max_idx] = -1.0f;  // Mark as selected
            
            // Atomic increment expert count
            atomicAdd(&expert_counts[max_idx], 1);
        }
        
        // Normalize topk weights if requested
        if (normalize) {
            float weight_sum = 0.0f;
            for (int k = 0; k < kTopK; k++) {
                weight_sum += float(topk_weights[token_idx * kTopK + k]);
            }
            for (int k = 0; k < kTopK; k++) {
                topk_weights[token_idx * kTopK + k] = T(
                    float(topk_weights[token_idx * kTopK + k]) / weight_sum
                );
            }
        }
    }
}

// ============================================================================
// API Implementation
// ============================================================================

Status fused_moe_gate(
    const Tensor& hidden_states,
    const Tensor& gate_weight,
    Tensor& topk_indices,
    Tensor& topk_weights,
    int* expert_counts,
    const FusedGateConfig& config,
    cudaStream_t stream) {
    
    // Initialize expert counts to zero
    cudaMemsetAsync(expert_counts, 0, config.num_experts * sizeof(int), stream);
    
    // Launch kernel
    dim3 grid(config.num_tokens);
    dim3 block(256);  // 8 warps
    
    // TODO: Template instantiation based on num_experts and top_k
    // For now, just show the structure
    
    // Example: 8 experts, top-2
    // fused_gate_topk_kernel<half, 8, 2><<<grid, block, 0, stream>>>(...);
    
    return Status::kSuccess;
}

Status compute_permutation(
    const Tensor& topk_indices,
    const int* expert_counts,
    int* permute_indices,
    int* expert_offsets,
    const FusedGateConfig& config,
    cudaStream_t stream) {
    
    // TODO: Implement permutation computation
    // 1. Prefix sum on expert_counts to get expert_offsets
    // 2. Scatter token indices to permute_indices based on expert assignment
    
    return Status::kSuccess;
}

}  // namespace ops
}  // namespace light_moe
