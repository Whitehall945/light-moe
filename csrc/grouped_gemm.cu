/**
 * @file grouped_gemm.cu
 * @brief Grouped GEMM using ATen (cuBLAS backend)
 * 
 * Decision: Use ATen/cuBLAS for GEMM as it's already highly optimized.
 * Focus engineering effort on Fused ops and Expert Parallelism instead.
 */

#include <torch/extension.h>

torch::Tensor grouped_gemm_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor expert_offsets,
    int num_experts) {
    
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be CUDA tensor");
    
    const int total_tokens = input.size(0);
    const int K = input.size(1);
    const int N = weights.size(1);
    
    auto output = torch::zeros({total_tokens, N}, input.options());
    if (total_tokens == 0) return output;
    
    auto offsets_cpu = expert_offsets.to(torch::kCPU);
    auto offsets_acc = offsets_cpu.accessor<int64_t, 1>();
    
    for (int e = 0; e < num_experts; e++) {
        int64_t start = offsets_acc[e];
        int64_t end = offsets_acc[e + 1];
        
        if (end > start) {
            auto expert_input = input.slice(0, start, end);
            auto expert_weight = weights.select(0, e);
            auto expert_output = output.slice(0, start, end);
            torch::mm_out(expert_output, expert_input, expert_weight.t());
        }
    }
    
    return output;
}
