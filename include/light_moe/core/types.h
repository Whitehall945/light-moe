#pragma once

/**
 * @file types.h
 * @brief Core type definitions for Light-MoE
 */

#include <cstdint>
#include <cuda_fp16.h>

namespace light_moe {

// ============================================================================
// Scalar Types
// ============================================================================

using fp32_t = float;
using fp16_t = __half;
using bf16_t = __nv_bfloat16;
using int8_t = std::int8_t;
using int32_t = std::int32_t;
using int64_t = std::int64_t;
using uint32_t = std::uint32_t;
using uint64_t = std::uint64_t;

// ============================================================================
// Data Type Enum
// ============================================================================

enum class DataType {
    kFloat32 = 0,
    kFloat16 = 1,
    kBFloat16 = 2,
    kInt8 = 3,
    kInt4 = 4,  // Packed 4-bit integers
    kUInt8 = 5,
};

/**
 * @brief Get the size in bytes for a given data type
 */
inline size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::kFloat32:
            return 4;
        case DataType::kFloat16:
        case DataType::kBFloat16:
            return 2;
        case DataType::kInt8:
        case DataType::kUInt8:
            return 1;
        case DataType::kInt4:
            return 1;  // 2 elements per byte
        default:
            return 0;
    }
}

// ============================================================================
// MoE Configuration
// ============================================================================

/**
 * @brief Configuration for MoE layer
 */
struct MoEConfig {
    int num_experts;           // Total number of experts
    int num_experts_per_token; // Top-K experts to activate
    int hidden_size;           // Hidden dimension
    int intermediate_size;     // FFN intermediate dimension
    
    // Parallelism settings
    int expert_parallel_size;  // Number of GPUs for expert parallelism
    int tensor_parallel_size;  // Number of GPUs for tensor parallelism
    
    // Quantization
    bool use_int4_weights;     // Enable W4A16 quantization
    int group_size;            // Quantization group size (default: 128)
    
    MoEConfig()
        : num_experts(8)
        , num_experts_per_token(2)
        , hidden_size(4096)
        , intermediate_size(14336)
        , expert_parallel_size(1)
        , tensor_parallel_size(1)
        , use_int4_weights(false)
        , group_size(128) {}
};

// ============================================================================
// Status and Error Handling
// ============================================================================

enum class Status {
    kSuccess = 0,
    kInvalidArgument = 1,
    kOutOfMemory = 2,
    kCudaError = 3,
    kNcclError = 4,
    kInternalError = 5,
};

inline const char* status_to_string(Status status) {
    switch (status) {
        case Status::kSuccess:
            return "Success";
        case Status::kInvalidArgument:
            return "Invalid Argument";
        case Status::kOutOfMemory:
            return "Out of Memory";
        case Status::kCudaError:
            return "CUDA Error";
        case Status::kNcclError:
            return "NCCL Error";
        case Status::kInternalError:
            return "Internal Error";
        default:
            return "Unknown Error";
    }
}

}  // namespace light_moe
