/**
 * @file bindings.cpp
 * @brief PyBind11 bindings for Light-MoE CUDA kernels
 */

#include <torch/extension.h>

// Forward declarations
torch::Tensor grouped_gemm_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor expert_offsets,
    int num_experts);

std::tuple<torch::Tensor, torch::Tensor> fused_gate_topk_cuda(
    torch::Tensor hidden_states,
    torch::Tensor gate_weight,
    int top_k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_gemm", &grouped_gemm_cuda,
          "Grouped GEMM for MoE (CUDA)",
          py::arg("input"),
          py::arg("weights"),
          py::arg("expert_offsets"),
          py::arg("num_experts"));
    
    m.def("fused_gate_topk", &fused_gate_topk_cuda,
          "Fused Gate + TopK for MoE (CUDA)",
          py::arg("hidden_states"),
          py::arg("gate_weight"),
          py::arg("top_k"));
}
