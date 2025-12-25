/**
 * @file fused_gate.cu
 * @brief Fused Gate + TopK using ATen operations
 * 
 * Uses torch operations for reliability and correctness.
 * The benefit is reduced Python overhead.
 */

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> fused_gate_topk_cuda(
    torch::Tensor hidden_states,
    torch::Tensor gate_weight,
    int top_k) {
    
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be a CUDA tensor");
    TORCH_CHECK(gate_weight.is_cuda(), "gate_weight must be a CUDA tensor");
    
    const int num_tokens = hidden_states.size(0);
    const int hidden_size = hidden_states.size(1);
    const int num_experts = gate_weight.size(0);
    
    // Compute gate logits: [num_tokens, num_experts]
    // logits = hidden_states @ gate_weight.T
    auto logits = torch::mm(hidden_states, gate_weight.t());
    
    // Softmax
    auto probs = torch::softmax(logits, /*dim=*/-1);
    
    // TopK
    auto topk_result = torch::topk(probs, top_k, /*dim=*/-1);
    auto topk_weights = std::get<0>(topk_result);
    auto topk_indices = std::get<1>(topk_result);
    
    // Normalize weights
    auto weight_sum = topk_weights.sum(/*dim=*/-1, /*keepdim=*/true);
    topk_weights = topk_weights / weight_sum;
    
    return std::make_tuple(topk_indices.to(torch::kInt32), topk_weights);
}
