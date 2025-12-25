#pragma once

/**
 * @file grouped_gemm.h
 * @brief Grouped GEMM interface for MoE expert computation
 * 
 * MoE models route different tokens to different experts, resulting in
 * variable-sized batches per expert. Grouped GEMM efficiently handles
 * these dynamic shapes by batching multiple small GEMMs together.
 */

#include <cuda_runtime.h>

#include "light_moe/core/tensor.h"
#include "light_moe/core/types.h"

namespace light_moe {
namespace ops {

/**
 * @brief Configuration for grouped GEMM operation
 */
struct GroupedGemmConfig {
    int num_groups;            // Number of expert groups
    int max_m;                 // Maximum tokens per expert
    int n;                     // Output dimension
    int k;                     // Input dimension
    
    DataType input_dtype;      // Input activation dtype
    DataType weight_dtype;     // Weight dtype (may differ for quantization)
    DataType output_dtype;     // Output dtype
    
    bool use_bias;             // Add bias after GEMM
    
    GroupedGemmConfig()
        : num_groups(8)
        , max_m(256)
        , n(4096)
        , k(4096)
        , input_dtype(DataType::kFloat16)
        , weight_dtype(DataType::kFloat16)
        , output_dtype(DataType::kFloat16)
        , use_bias(false) {}
};

/**
 * @brief Grouped GEMM using CuTe for MoE computation
 * 
 * Computes: output[g] = input[g] @ weight[g].T + bias[g]
 * for each expert group g.
 * 
 * @param input Input activations [sum(m_g), k]
 * @param weights Expert weights [num_groups, n, k]
 * @param output Output activations [sum(m_g), n]
 * @param group_offsets Offset of each group in the concatenated input [num_groups + 1]
 * @param group_sizes Number of tokens per group [num_groups]
 * @param bias Optional bias [num_groups, n]
 * @param config GEMM configuration
 * @param stream CUDA stream
 * @return Status code
 */
Status grouped_gemm(
    const Tensor& input,
    const Tensor& weights,
    Tensor& output,
    const int* group_offsets,
    const int* group_sizes,
    const Tensor* bias,
    const GroupedGemmConfig& config,
    cudaStream_t stream = nullptr);

/**
 * @brief Grouped GEMM with INT4 weight-only quantization
 * 
 * Weights are stored as INT4 with per-group scales and zeros.
 * Computation: output = input @ dequant(weight)
 * 
 * @param input Input activations [sum(m_g), k] in FP16
 * @param weights_int4 Quantized weights [num_groups, n, k/2] as packed INT4
 * @param scales Per-group quantization scales [num_groups, n, k/group_size]
 * @param zeros Per-group quantization zeros [num_groups, n, k/group_size]
 * @param output Output activations [sum(m_g), n]
 * @param group_offsets Offset of each group
 * @param group_sizes Number of tokens per group
 * @param config GEMM configuration
 * @param stream CUDA stream
 * @return Status code
 */
Status grouped_gemm_int4(
    const Tensor& input,
    const Tensor& weights_int4,
    const Tensor& scales,
    const Tensor& zeros,
    Tensor& output,
    const int* group_offsets,
    const int* group_sizes,
    const GroupedGemmConfig& config,
    cudaStream_t stream = nullptr);

}  // namespace ops
}  // namespace light_moe
