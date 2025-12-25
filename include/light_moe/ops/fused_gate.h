#pragma once

/**
 * @file fused_gate.h
 * @brief Fused MoE Gate + TopK operator
 * 
 * The MoE gate computes routing probabilities and selects top-K experts
 * for each token. This fused implementation reduces memory bandwidth
 * by combining gate logits computation with softmax and TopK selection.
 */

#include <cuda_runtime.h>

#include "light_moe/core/tensor.h"
#include "light_moe/core/types.h"

namespace light_moe {
namespace ops {

/**
 * @brief Configuration for fused gate operation
 */
struct FusedGateConfig {
    int num_tokens;            // Batch size (number of tokens)
    int hidden_size;           // Input hidden dimension
    int num_experts;           // Total number of experts
    int top_k;                 // Number of experts to select per token
    
    bool normalize_weights;    // Normalize routing weights to sum to 1
    float capacity_factor;     // Expert capacity factor (for load balancing)
    
    FusedGateConfig()
        : num_tokens(1)
        , hidden_size(4096)
        , num_experts(8)
        , top_k(2)
        , normalize_weights(true)
        , capacity_factor(1.25f) {}
};

/**
 * @brief Fused MoE Gate + TopK selection
 * 
 * Computes:
 * 1. gate_logits = hidden_states @ gate_weight.T
 * 2. gate_probs = softmax(gate_logits)
 * 3. topk_indices, topk_weights = topk(gate_probs, top_k)
 * 
 * @param hidden_states Input hidden states [num_tokens, hidden_size]
 * @param gate_weight Gate projection weight [num_experts, hidden_size]
 * @param topk_indices Selected expert indices [num_tokens, top_k]
 * @param topk_weights Routing weights for selected experts [num_tokens, top_k]
 * @param expert_counts Number of tokens routed to each expert [num_experts]
 * @param config Gate configuration
 * @param stream CUDA stream
 * @return Status code
 */
Status fused_moe_gate(
    const Tensor& hidden_states,
    const Tensor& gate_weight,
    Tensor& topk_indices,
    Tensor& topk_weights,
    int* expert_counts,
    const FusedGateConfig& config,
    cudaStream_t stream = nullptr);

/**
 * @brief Compute token permutation indices for expert routing
 * 
 * After gate selection, tokens need to be reordered so that tokens
 * going to the same expert are contiguous. This function computes
 * the permutation indices.
 * 
 * @param topk_indices Expert indices [num_tokens, top_k]
 * @param expert_counts Number of tokens per expert [num_experts]
 * @param permute_indices Output permutation indices [num_tokens * top_k]
 * @param expert_offsets Offset of each expert's tokens [num_experts + 1]
 * @param config Gate configuration
 * @param stream CUDA stream
 * @return Status code
 */
Status compute_permutation(
    const Tensor& topk_indices,
    const int* expert_counts,
    int* permute_indices,
    int* expert_offsets,
    const FusedGateConfig& config,
    cudaStream_t stream = nullptr);

}  // namespace ops
}  // namespace light_moe
