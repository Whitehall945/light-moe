/**
 * @file grouped_gemm.cu
 * @brief Grouped GEMM using ATen/PyTorch operations
 * 
 * Uses torch::mm for each expert group. This is clean and works correctly.
 * Performance is the same as PyTorch since we're calling the same GEMM.
 * 
 * The advantage is reduced Python overhead when processing multiple experts.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor grouped_gemm_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor expert_offsets,
    int num_experts) {
    
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(expert_offsets.is_cuda(), "expert_offsets must be a CUDA tensor");
    
    const int total_tokens = input.size(0);
    const int K = input.size(1);  // hidden_size
    const int N = weights.size(1);  // out_features
    
    auto output = torch::zeros({total_tokens, N}, input.options());
    
    if (total_tokens == 0) return output;
    
    // Get offsets on CPU
    auto offsets_cpu = expert_offsets.to(torch::kCPU);
    auto offsets_acc = offsets_cpu.accessor<int64_t, 1>();
    
    // Process each expert using torch::mm
    for (int e = 0; e < num_experts; e++) {
        int64_t start = offsets_acc[e];
        int64_t end = offsets_acc[e + 1];
        
        if (end > start) {
            // Slice input and output
            auto expert_input = input.slice(0, start, end);
            auto expert_weight = weights.select(0, e);  // [N, K]
            auto expert_output = output.slice(0, start, end);
            
            // C[M,N] = A[M,K] @ B[N,K]^T
            torch::mm_out(expert_output, expert_input, expert_weight.t());
        }
    }
    
    return output;
}
