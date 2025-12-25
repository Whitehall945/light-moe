/**
 * @file fused_moe_gate.cu
 * @brief Fused MoE Gate + TopK CUDA kernel
 * 
 * Combines gate projection, softmax, and TopK selection into a single
 * kernel to minimize memory bandwidth and kernel launch overhead.
 * 
 * Optimizations:
 * - Warp-level reductions for softmax
 * - Register-based TopK selection
 * - Vectorized memory access
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

#include "light_moe/ops/fused_gate.h"

namespace light_moe {
namespace ops {

// ============================================================================
// Constants
// ============================================================================

constexpr int kWarpSize = 32;
constexpr int kMaxExpertsPerBlock = 256;

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Warp-level maximum reduction
 */
template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

/**
 * @brief Warp-level sum reduction
 */
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * @brief Block-level maximum reduction using shared memory
 */
template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ T block_reduce_max(T val, T* shared) {
    int lane = threadIdx.x % kWarpSize;
    int wid = threadIdx.x / kWarpSize;
    int num_warps = BLOCK_SIZE / kWarpSize;
    
    // Warp-level reduction first
    val = warp_reduce_max(val);
    
    // Write warp results to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // Final reduction in first warp
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : -FLT_MAX;
    if (wid == 0) {
        val = warp_reduce_max(val);
    }
    
    return val;
}

/**
 * @brief Block-level sum reduction using shared memory
 */
template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ T block_reduce_sum(T val, T* shared) {
    int lane = threadIdx.x % kWarpSize;
    int wid = threadIdx.x / kWarpSize;
    int num_warps = BLOCK_SIZE / kWarpSize;
    
    // Warp-level reduction first
    val = warp_reduce_sum(val);
    
    // Write warp results to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // Final reduction in first warp
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    
    return val;
}

// ============================================================================
// Fused Gate + TopK Kernel
// ============================================================================

/**
 * @brief Fused gate projection + softmax + top-K selection
 * 
 * Each thread block processes one token.
 * 
 * Algorithm:
 * 1. Load hidden states (vectorized)
 * 2. Compute gate_logits = hidden @ gate_weight.T (parallel reduction)
 * 3. Softmax with numerical stability (block reduction for max/sum)
 * 4. TopK selection using insertion sort
 * 5. Atomic update of expert counts
 * 
 * @tparam T Data type (half or float)
 * @tparam NUM_EXPERTS Number of experts (compile-time constant)
 * @tparam TOP_K Number of experts to select
 * @tparam BLOCK_SIZE Threads per block
 */
template <typename T, int NUM_EXPERTS, int TOP_K, int BLOCK_SIZE = 256>
__global__ void fused_gate_topk_kernel(
    const T* __restrict__ hidden_states,    // [num_tokens, hidden_size]
    const T* __restrict__ gate_weight,      // [num_experts, hidden_size]
    int* __restrict__ topk_indices,         // [num_tokens, top_k]
    T* __restrict__ topk_weights,           // [num_tokens, top_k]
    int* __restrict__ expert_counts,        // [num_experts]
    int num_tokens,
    int hidden_size,
    bool normalize) {
    
    // Shared memory
    __shared__ float s_logits[NUM_EXPERTS];
    __shared__ float s_reduce[BLOCK_SIZE / kWarpSize];
    
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    int tid = threadIdx.x;
    
    // =========================================================================
    // Step 1: Compute gate logits via parallel dot products
    // =========================================================================
    
    // Each thread computes part of the dot product for assigned experts
    int num_experts_per_thread = (NUM_EXPERTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    const T* token_hidden = hidden_states + token_idx * hidden_size;
    
    for (int e = tid; e < NUM_EXPERTS; e += BLOCK_SIZE) {
        const T* expert_weight = gate_weight + e * hidden_size;
        
        // Compute dot product
        float sum = 0.0f;
        
        // Use vectorized loads for better memory bandwidth
        const int vec_size = 4;
        int vec_iters = hidden_size / vec_size;
        
        for (int i = 0; i < vec_iters; i++) {
            int idx = i * vec_size;
            
            // Load 4 elements at a time
            float4 h, w;
            if constexpr (sizeof(T) == 2) {  // half
                // Load as half4 and convert
                const half* h_ptr = reinterpret_cast<const half*>(token_hidden + idx);
                const half* w_ptr = reinterpret_cast<const half*>(expert_weight + idx);
                
                h.x = __half2float(h_ptr[0]);
                h.y = __half2float(h_ptr[1]);
                h.z = __half2float(h_ptr[2]);
                h.w = __half2float(h_ptr[3]);
                
                w.x = __half2float(w_ptr[0]);
                w.y = __half2float(w_ptr[1]);
                w.z = __half2float(w_ptr[2]);
                w.w = __half2float(w_ptr[3]);
            } else {  // float
                h = *reinterpret_cast<const float4*>(token_hidden + idx);
                w = *reinterpret_cast<const float4*>(expert_weight + idx);
            }
            
            sum += h.x * w.x + h.y * w.y + h.z * w.z + h.w * w.w;
        }
        
        // Handle remainder
        for (int i = vec_iters * vec_size; i < hidden_size; i++) {
            sum += static_cast<float>(token_hidden[i]) * static_cast<float>(expert_weight[i]);
        }
        
        s_logits[e] = sum;
    }
    __syncthreads();
    
    // =========================================================================
    // Step 2: Softmax with numerical stability
    // =========================================================================
    
    // Find max for numerical stability
    float local_max = -FLT_MAX;
    for (int e = tid; e < NUM_EXPERTS; e += BLOCK_SIZE) {
        local_max = max(local_max, s_logits[e]);
    }
    float global_max = block_reduce_max<float, BLOCK_SIZE>(local_max, s_reduce);
    global_max = __shfl_sync(0xffffffff, global_max, 0);
    __syncthreads();
    
    // Compute exp and local sum
    float local_sum = 0.0f;
    for (int e = tid; e < NUM_EXPERTS; e += BLOCK_SIZE) {
        float exp_val = expf(s_logits[e] - global_max);
        s_logits[e] = exp_val;
        local_sum += exp_val;
    }
    float global_sum = block_reduce_sum<float, BLOCK_SIZE>(local_sum, s_reduce);
    global_sum = __shfl_sync(0xffffffff, global_sum, 0);
    __syncthreads();
    
    // Normalize to get probabilities
    float inv_sum = 1.0f / global_sum;
    for (int e = tid; e < NUM_EXPERTS; e += BLOCK_SIZE) {
        s_logits[e] *= inv_sum;
    }
    __syncthreads();
    
    // =========================================================================
    // Step 3: TopK selection using insertion sort (single thread)
    // =========================================================================
    
    if (tid == 0) {
        // TopK arrays
        int top_indices[TOP_K];
        float top_values[TOP_K];
        
        // Initialize
        for (int k = 0; k < TOP_K; k++) {
            top_values[k] = -FLT_MAX;
            top_indices[k] = -1;
        }
        
        // Find top-K
        for (int e = 0; e < NUM_EXPERTS; e++) {
            float val = s_logits[e];
            
            // Check if this expert should be in top-K
            if (val > top_values[TOP_K - 1]) {
                // Find insertion position
                int pos = TOP_K - 1;
                while (pos > 0 && val > top_values[pos - 1]) {
                    pos--;
                }
                
                // Shift and insert
                for (int k = TOP_K - 1; k > pos; k--) {
                    top_values[k] = top_values[k - 1];
                    top_indices[k] = top_indices[k - 1];
                }
                top_values[pos] = val;
                top_indices[pos] = e;
            }
        }
        
        // Normalize weights if requested
        float weight_sum = 0.0f;
        if (normalize) {
            for (int k = 0; k < TOP_K; k++) {
                weight_sum += top_values[k];
            }
        }
        
        // Write results and update expert counts
        for (int k = 0; k < TOP_K; k++) {
            topk_indices[token_idx * TOP_K + k] = top_indices[k];
            
            float weight = normalize ? (top_values[k] / weight_sum) : top_values[k];
            if constexpr (sizeof(T) == 2) {
                topk_weights[token_idx * TOP_K + k] = __float2half(weight);
            } else {
                topk_weights[token_idx * TOP_K + k] = static_cast<T>(weight);
            }
            
            // Atomic increment expert count
            atomicAdd(&expert_counts[top_indices[k]], 1);
        }
    }
}

// ============================================================================
// Permutation Kernel
// ============================================================================

/**
 * @brief Compute token permutation for expert routing
 * 
 * Reorders tokens so that tokens going to the same expert are contiguous.
 */
__global__ void compute_permutation_kernel(
    const int* __restrict__ topk_indices,  // [num_tokens, top_k]
    const int* __restrict__ expert_counts, // [num_experts]
    int* __restrict__ permute_indices,     // [num_tokens * top_k]
    int* __restrict__ expert_offsets,      // [num_experts + 1]
    int num_tokens,
    int top_k,
    int num_experts) {
    
    // This kernel uses a single thread block for simplicity
    // For production, should use parallel prefix sum
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Compute prefix sum for expert offsets
        expert_offsets[0] = 0;
        for (int e = 0; e < num_experts; e++) {
            expert_offsets[e + 1] = expert_offsets[e] + expert_counts[e];
        }
        
        // Use temporary counters
        int* counters = permute_indices;  // Reuse as temp (will be overwritten)
        for (int e = 0; e < num_experts; e++) {
            counters[e] = expert_offsets[e];
        }
        
        // Scatter token indices
        for (int t = 0; t < num_tokens; t++) {
            for (int k = 0; k < top_k; k++) {
                int expert_idx = topk_indices[t * top_k + k];
                int dest = counters[expert_idx]++;
                permute_indices[dest] = t * top_k + k;
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
    
    // Determine kernel configuration
    dim3 grid(config.num_tokens);
    dim3 block(256);
    
    // Dispatch based on num_experts and top_k
    // Using switch for common configurations
    #define LAUNCH_KERNEL(EXPERTS, TOPK)                                    \
        fused_gate_topk_kernel<half, EXPERTS, TOPK, 256><<<grid, block, 0, stream>>>( \
            static_cast<const half*>(hidden_states.data()),                 \
            static_cast<const half*>(gate_weight.data()),                   \
            static_cast<int*>(topk_indices.data()),                         \
            static_cast<half*>(topk_weights.data()),                        \
            expert_counts,                                                  \
            config.num_tokens,                                              \
            config.hidden_size,                                             \
            config.normalize_weights                                        \
        );
    
    if (config.num_experts == 8 && config.top_k == 2) {
        LAUNCH_KERNEL(8, 2);
    } else if (config.num_experts == 16 && config.top_k == 2) {
        LAUNCH_KERNEL(16, 2);
    } else if (config.num_experts == 8 && config.top_k == 1) {
        LAUNCH_KERNEL(8, 1);
    } else {
        // Fallback for other configurations
        // In production, would use runtime compilation or more cases
        return Status::kInvalidArgument;
    }
    
    #undef LAUNCH_KERNEL
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return Status::kCudaError;
    }
    
    return Status::kSuccess;
}

Status compute_permutation(
    const Tensor& topk_indices,
    const int* expert_counts,
    int* permute_indices,
    int* expert_offsets,
    const FusedGateConfig& config,
    cudaStream_t stream) {
    
    compute_permutation_kernel<<<1, 1, 0, stream>>>(
        static_cast<const int*>(topk_indices.data()),
        expert_counts,
        permute_indices,
        expert_offsets,
        config.num_tokens,
        config.top_k,
        config.num_experts
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return Status::kCudaError;
    }
    
    return Status::kSuccess;
}

}  // namespace ops
}  // namespace light_moe
