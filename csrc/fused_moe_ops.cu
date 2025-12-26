/**
 * @file fused_moe_ops.cu
 * @brief Fused MoE Route and Permute Operations
 * 
 * Strategy:
 * - Gate projection: Use cuBLAS/ATen (already optimized)
 * - Softmax + TopK + Permute: Fuse these in CUDA kernel
 * 
 * This fuses the POST-gate operations that existing libraries don't optimize.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

// =============================================================================
// Fused Softmax + TopK Kernel
// =============================================================================

/**
 * Fused Softmax + TopK selection.
 * Input: gate_logits [num_tokens, num_experts]
 * Output: expert_ids [num_tokens, top_k], expert_weights [num_tokens, top_k]
 * 
 * Each warp handles one token.
 */
template<int NUM_EXPERTS, int TOP_K>
__global__ void fused_softmax_topk_kernel(
    const float* __restrict__ gate_logits,  // [num_tokens, num_experts]
    int32_t* __restrict__ expert_ids,       // [num_tokens, top_k]
    float* __restrict__ expert_weights,     // [num_tokens, top_k]
    int num_tokens) {
    
    const int token_id = blockIdx.x * blockDim.y + threadIdx.y;
    if (token_id >= num_tokens) return;
    
    const int lane_id = threadIdx.x;  // 0-31 within warp
    
    // Load logits into registers (each thread loads subset)
    float logits[NUM_EXPERTS / 32 + 1];
    const float* token_logits = gate_logits + token_id * NUM_EXPERTS;
    
    #pragma unroll
    for (int i = lane_id; i < NUM_EXPERTS; i += 32) {
        logits[i / 32] = token_logits[i];
    }
    
    // Warp-reduce to find max
    float max_val = -FLT_MAX;
    #pragma unroll
    for (int i = lane_id; i < NUM_EXPERTS; i += 32) {
        max_val = fmaxf(max_val, logits[i / 32]);
    }
    
    // Warp shuffle to get global max
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    #pragma unroll
    for (int i = lane_id; i < NUM_EXPERTS; i += 32) {
        float v = expf(logits[i / 32] - max_val);
        logits[i / 32] = v;
        sum += v;
    }
    
    // Warp shuffle to get global sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
    
    // Normalize and compute probabilities
    float inv_sum = 1.0f / sum;
    float probs[NUM_EXPERTS / 32 + 1];
    #pragma unroll
    for (int i = lane_id; i < NUM_EXPERTS; i += 32) {
        probs[i / 32] = logits[i / 32] * inv_sum;
    }
    
    // TopK selection using warp shuffle (only lane 0 does selection)
    // Gather all probs to lane 0
    __shared__ float s_probs[32 * NUM_EXPERTS];  // Assuming max 32 tokens per block
    
    #pragma unroll
    for (int i = lane_id; i < NUM_EXPERTS; i += 32) {
        s_probs[threadIdx.y * NUM_EXPERTS + i] = probs[i / 32];
    }
    __syncwarp();
    
    if (lane_id == 0) {
        int topk_idx[TOP_K];
        float topk_val[TOP_K];
        
        #pragma unroll
        for (int k = 0; k < TOP_K; k++) {
            topk_idx[k] = 0;
            topk_val[k] = -FLT_MAX;
        }
        
        // Find top-k
        for (int e = 0; e < NUM_EXPERTS; e++) {
            float p = s_probs[threadIdx.y * NUM_EXPERTS + e];
            
            if (p > topk_val[TOP_K - 1]) {
                int pos = TOP_K - 1;
                while (pos > 0 && p > topk_val[pos - 1]) {
                    topk_val[pos] = topk_val[pos - 1];
                    topk_idx[pos] = topk_idx[pos - 1];
                    pos--;
                }
                topk_val[pos] = p;
                topk_idx[pos] = e;
            }
        }
        
        // Renormalize
        float topk_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < TOP_K; k++) {
            topk_sum += topk_val[k];
        }
        float inv_topk_sum = 1.0f / topk_sum;
        
        // Write output
        #pragma unroll
        for (int k = 0; k < TOP_K; k++) {
            expert_ids[token_id * TOP_K + k] = topk_idx[k];
            expert_weights[token_id * TOP_K + k] = topk_val[k] * inv_topk_sum;
        }
    }
}

// =============================================================================
// Python Interface
// =============================================================================

/**
 * Standard Gate + TopK (separate ops)
 */
std::tuple<torch::Tensor, torch::Tensor> fused_gate_topk_cuda(
    torch::Tensor hidden_states,
    torch::Tensor gate_weight,
    int top_k) {
    
    TORCH_CHECK(hidden_states.is_cuda(), "hidden must be CUDA");
    TORCH_CHECK(gate_weight.is_cuda(), "gate_weight must be CUDA");
    
    const int num_tokens = hidden_states.size(0);
    const int hidden_size = hidden_states.size(1);
    const int num_experts = gate_weight.size(0);
    
    // Step 1: Gate projection using cuBLAS (this is already fast)
    auto gate_logits = torch::mm(hidden_states.to(torch::kFloat32), 
                                  gate_weight.to(torch::kFloat32).t());
    
    // Step 2: Fused Softmax + TopK using our kernel
    auto expert_ids = torch::empty({num_tokens, top_k}, 
        torch::dtype(torch::kInt32).device(hidden_states.device()));
    auto expert_weights = torch::empty({num_tokens, top_k}, 
        torch::dtype(torch::kFloat32).device(hidden_states.device()));
    
    // Launch: 1 warp per token, 8 warps per block
    constexpr int WARPS_PER_BLOCK = 8;
    dim3 grid((num_tokens + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    dim3 block(32, WARPS_PER_BLOCK);
    
    if (num_experts == 8 && top_k == 2) {
        fused_softmax_topk_kernel<8, 2><<<grid, block>>>(
            gate_logits.data_ptr<float>(),
            expert_ids.data_ptr<int32_t>(),
            expert_weights.data_ptr<float>(),
            num_tokens
        );
    } else if (num_experts == 16 && top_k == 2) {
        fused_softmax_topk_kernel<16, 2><<<grid, block>>>(
            gate_logits.data_ptr<float>(),
            expert_ids.data_ptr<int32_t>(),
            expert_weights.data_ptr<float>(),
            num_tokens
        );
    } else {
        // Fallback to PyTorch
        auto probs = torch::softmax(gate_logits, -1);
        auto topk = torch::topk(probs, top_k, -1);
        auto weights = std::get<0>(topk);
        auto indices = std::get<1>(topk);
        weights = weights / weights.sum(-1, true);
        return std::make_tuple(indices.to(torch::kInt32), weights.to(hidden_states.dtype()));
    }
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    
    return std::make_tuple(expert_ids, expert_weights.to(hidden_states.dtype()));
}

/**
 * Placeholder for full fused pipeline
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_gate_topk_permute_cuda(
    torch::Tensor hidden_states,
    torch::Tensor gate_weight,
    int top_k) {
    
    // For now, just call fused_gate_topk and do permute separately
    auto [expert_ids, expert_weights] = fused_gate_topk_cuda(hidden_states, gate_weight, top_k);
    
    // Compute expert counts and offsets
    const int num_experts = gate_weight.size(0);
    auto expert_counts = torch::zeros({num_experts}, torch::kInt64).to(hidden_states.device());
    for (int e = 0; e < num_experts; e++) {
        expert_counts[e] = (expert_ids == e).sum();
    }
    auto expert_offsets = torch::zeros({num_experts + 1}, torch::kInt64).to(hidden_states.device());
    expert_offsets.slice(0, 1) = expert_counts.cumsum(0);
    
    // Permute tokens (simple PyTorch version for now)
    auto flat_experts = expert_ids.flatten();
    auto sorted_idx = torch::argsort(flat_experts);
    
    auto hidden_expanded = hidden_states.unsqueeze(1).expand({-1, top_k, -1}).reshape({-1, hidden_states.size(1)});
    auto permuted = hidden_expanded.index_select(0, sorted_idx);
    
    return std::make_tuple(permuted, expert_ids, expert_weights, expert_offsets, sorted_idx);
}
