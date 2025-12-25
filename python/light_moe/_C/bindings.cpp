/**
 * @file bindings.cpp
 * @brief PyBind11 bindings for Light-MoE CUDA operators
 * 
 * Exposes C++/CUDA functions to Python for use with PyTorch.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

// ============================================================================
// Forward Declarations
// ============================================================================

// Grouped GEMM
torch::Tensor grouped_gemm_cuda(
    torch::Tensor input,           // [total_tokens, hidden_size]
    torch::Tensor weights,         // [num_experts, out_features, hidden_size]
    torch::Tensor expert_offsets,  // [num_experts + 1]
    int num_experts
);

// Fused Gate + TopK
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_moe_gate_cuda(
    torch::Tensor hidden_states,   // [num_tokens, hidden_size]
    torch::Tensor gate_weight,     // [num_experts, hidden_size]
    int top_k,
    bool normalize
);

// Token permutation
std::tuple<torch::Tensor, torch::Tensor> permute_tokens_cuda(
    torch::Tensor hidden_states,   // [batch, seq_len, hidden_size]
    torch::Tensor expert_indices,  // [batch, seq_len, top_k]
    int num_experts
);

// Token unpermutation
torch::Tensor unpermute_tokens_cuda(
    torch::Tensor permuted_output,   // [total_tokens * top_k, hidden_size]
    torch::Tensor reverse_indices,   // [total_tokens * top_k]
    torch::Tensor routing_weights,   // [batch, seq_len, top_k]
    int batch_size,
    int seq_len
);

// ============================================================================
// Implementation (Fallback to PyTorch when CUDA not available)
// ============================================================================

/**
 * @brief Grouped GEMM implementation
 * 
 * For each expert, computes: output[expert_offsets[i]:expert_offsets[i+1]] = 
 *     input[expert_offsets[i]:expert_offsets[i+1]] @ weights[i].T
 */
torch::Tensor grouped_gemm_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor expert_offsets,
    int num_experts) {
    
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(weights.is_cuda(), "weights must be on CUDA");
    TORCH_CHECK(expert_offsets.is_cuda(), "expert_offsets must be on CUDA");
    
    TORCH_CHECK(input.dim() == 2, "input must be 2D [total_tokens, hidden_size]");
    TORCH_CHECK(weights.dim() == 3, "weights must be 3D [num_experts, out_features, hidden_size]");
    
    int total_tokens = input.size(0);
    int hidden_size = input.size(1);
    int out_features = weights.size(1);
    
    TORCH_CHECK(weights.size(0) == num_experts, "weights.size(0) must equal num_experts");
    TORCH_CHECK(weights.size(2) == hidden_size, "weights.size(2) must equal hidden_size");
    
    // Allocate output
    auto output = torch::empty({total_tokens, out_features}, input.options());
    
    // Get offsets on CPU for indexing
    auto offsets_cpu = expert_offsets.to(torch::kCPU);
    auto offsets_accessor = offsets_cpu.accessor<int64_t, 1>();
    
    // Process each expert
    for (int e = 0; e < num_experts; e++) {
        int64_t start = offsets_accessor[e];
        int64_t end = offsets_accessor[e + 1];
        
        if (end > start) {
            // Slice input and output
            auto expert_input = input.slice(0, start, end);
            auto expert_weight = weights.select(0, e);
            auto expert_output = output.slice(0, start, end);
            
            // GEMM: output = input @ weight.T
            torch::mm_out(expert_output, expert_input, expert_weight.t());
        }
    }
    
    return output;
}

/**
 * @brief Fused MoE gate + TopK implementation
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_moe_gate_cuda(
    torch::Tensor hidden_states,
    torch::Tensor gate_weight,
    int top_k,
    bool normalize) {
    
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be on CUDA");
    TORCH_CHECK(gate_weight.is_cuda(), "gate_weight must be on CUDA");
    
    int num_tokens = hidden_states.size(0);
    int hidden_size = hidden_states.size(1);
    int num_experts = gate_weight.size(0);
    
    TORCH_CHECK(gate_weight.size(1) == hidden_size, "gate_weight.size(1) must equal hidden_size");
    
    // Compute gate logits: [num_tokens, num_experts]
    auto logits = torch::mm(hidden_states, gate_weight.t());
    
    // Softmax
    auto probs = torch::softmax(logits, /*dim=*/-1);
    
    // TopK
    auto [topk_weights, topk_indices] = torch::topk(probs, top_k, /*dim=*/-1);
    
    // Normalize if requested
    if (normalize) {
        auto weight_sum = topk_weights.sum(/*dim=*/-1, /*keepdim=*/true);
        topk_weights = topk_weights / weight_sum;
    }
    
    // Count tokens per expert
    auto expert_counts = torch::zeros({num_experts}, torch::dtype(torch::kInt64).device(hidden_states.device()));
    for (int k = 0; k < top_k; k++) {
        auto indices_k = topk_indices.select(-1, k).flatten();
        expert_counts.scatter_add_(0, indices_k, torch::ones_like(indices_k, torch::kInt64));
    }
    
    return std::make_tuple(topk_indices.to(torch::kInt32), topk_weights, expert_counts);
}

/**
 * @brief Token permutation for grouped GEMM
 */
std::tuple<torch::Tensor, torch::Tensor> permute_tokens_cuda(
    torch::Tensor hidden_states,
    torch::Tensor expert_indices,
    int num_experts) {
    
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be on CUDA");
    TORCH_CHECK(expert_indices.is_cuda(), "expert_indices must be on CUDA");
    
    int batch_size = hidden_states.size(0);
    int seq_len = hidden_states.size(1);
    int hidden_size = hidden_states.size(2);
    int top_k = expert_indices.size(2);
    
    int total_tokens = batch_size * seq_len;
    int total_expanded = total_tokens * top_k;
    
    // Flatten hidden states and expand for top_k
    auto flat_hidden = hidden_states.reshape({total_tokens, hidden_size});
    auto expanded_hidden = flat_hidden.unsqueeze(1).expand({total_tokens, top_k, hidden_size});
    expanded_hidden = expanded_hidden.reshape({total_expanded, hidden_size});
    
    // Flatten expert indices
    auto flat_indices = expert_indices.reshape({total_expanded});
    
    // Sort by expert index
    auto [sorted_indices, sort_order] = flat_indices.sort();
    
    // Permute hidden states
    auto permuted = expanded_hidden.index_select(0, sort_order);
    
    // Compute reverse permutation
    auto reverse_indices = torch::argsort(sort_order);
    
    return std::make_tuple(permuted, reverse_indices);
}

/**
 * @brief Token unpermutation after grouped GEMM
 */
torch::Tensor unpermute_tokens_cuda(
    torch::Tensor permuted_output,
    torch::Tensor reverse_indices,
    torch::Tensor routing_weights,
    int batch_size,
    int seq_len) {
    
    TORCH_CHECK(permuted_output.is_cuda(), "permuted_output must be on CUDA");
    TORCH_CHECK(reverse_indices.is_cuda(), "reverse_indices must be on CUDA");
    TORCH_CHECK(routing_weights.is_cuda(), "routing_weights must be on CUDA");
    
    int hidden_size = permuted_output.size(1);
    int top_k = routing_weights.size(2);
    int total_tokens = batch_size * seq_len;
    
    // Unpermute
    auto unpermuted = permuted_output.index_select(0, reverse_indices);
    
    // Reshape to [total_tokens, top_k, hidden_size]
    unpermuted = unpermuted.reshape({total_tokens, top_k, hidden_size});
    
    // Weight by routing weights
    auto weights = routing_weights.reshape({total_tokens, top_k, 1});
    auto weighted = unpermuted * weights;
    
    // Sum over top_k
    auto combined = weighted.sum(/*dim=*/1);
    
    // Reshape to [batch, seq, hidden]
    return combined.reshape({batch_size, seq_len, hidden_size});
}

// ============================================================================
// Module Definition
// ============================================================================

PYBIND11_MODULE(_C, m) {
    m.doc() = "Light-MoE CUDA operators";
    
    m.def("grouped_gemm", &grouped_gemm_cuda,
          "Grouped GEMM for MoE expert computation",
          py::arg("input"),
          py::arg("weights"),
          py::arg("expert_offsets"),
          py::arg("num_experts"));
    
    m.def("fused_moe_gate", &fused_moe_gate_cuda,
          "Fused MoE gate + TopK selection",
          py::arg("hidden_states"),
          py::arg("gate_weight"),
          py::arg("top_k"),
          py::arg("normalize") = true);
    
    m.def("permute_tokens", &permute_tokens_cuda,
          "Permute tokens for grouped GEMM",
          py::arg("hidden_states"),
          py::arg("expert_indices"),
          py::arg("num_experts"));
    
    m.def("unpermute_tokens", &unpermute_tokens_cuda,
          "Unpermute tokens after grouped GEMM",
          py::arg("permuted_output"),
          py::arg("reverse_indices"),
          py::arg("routing_weights"),
          py::arg("batch_size"),
          py::arg("seq_len"));
}
