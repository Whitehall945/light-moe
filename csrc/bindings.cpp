/**
 * @file bindings.cpp
 * @brief PyBind11 bindings for Light-MoE CUDA kernels
 */

#include <torch/extension.h>

// Grouped GEMM
torch::Tensor grouped_gemm_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor expert_offsets,
    int num_experts);

// Fused Gate + TopK
std::tuple<torch::Tensor, torch::Tensor> fused_gate_topk_cuda(
    torch::Tensor hidden_states,
    torch::Tensor gate_weight,
    int top_k);

// Fused Gate + TopK + Permute (full pipeline)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_gate_topk_permute_cuda(
    torch::Tensor hidden_states,
    torch::Tensor gate_weight,
    int top_k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Light-MoE CUDA kernels for high-performance MoE inference";
    
    m.def("grouped_gemm", &grouped_gemm_cuda,
          "Grouped GEMM for MoE experts (uses cuBLAS)",
          py::arg("input"),
          py::arg("weights"),
          py::arg("expert_offsets"),
          py::arg("num_experts"));
    
    m.def("fused_gate_topk", &fused_gate_topk_cuda,
          "Fused Gate projection + Softmax + TopK selection",
          py::arg("hidden_states"),
          py::arg("gate_weight"),
          py::arg("top_k"));
    
    m.def("fused_gate_topk_permute", &fused_gate_topk_permute_cuda,
          "Full fused pipeline: Gate + TopK + Token Permutation",
          py::arg("hidden_states"),
          py::arg("gate_weight"),
          py::arg("top_k"));
}
