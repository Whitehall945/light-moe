#pragma once

/**
 * @file all_to_all.h
 * @brief All-to-All communication for Expert Parallelism
 * 
 * Expert Parallelism requires All-to-All communication to exchange
 * tokens between GPUs. This module provides optimized implementations
 * with support for communication-computation overlap.
 */

#include <cuda_runtime.h>

#ifdef LIGHT_MOE_USE_NCCL
#include <nccl.h>
#endif

#include "light_moe/core/tensor.h"
#include "light_moe/core/types.h"

namespace light_moe {
namespace comm {

/**
 * @brief Communication group for multi-GPU operations
 */
class CommGroup {
public:
    CommGroup() = default;
    ~CommGroup();
    
    /**
     * @brief Initialize communication group
     * @param world_size Total number of GPUs
     * @param rank Current GPU rank
     * @param device_id CUDA device ID
     * @return Status code
     */
    Status init(int world_size, int rank, int device_id);
    
    /**
     * @brief Check if initialized
     */
    bool is_initialized() const { return initialized_; }
    
    int world_size() const { return world_size_; }
    int rank() const { return rank_; }
    int device_id() const { return device_id_; }
    
#ifdef LIGHT_MOE_USE_NCCL
    ncclComm_t nccl_comm() const { return nccl_comm_; }
#endif
    
private:
    bool initialized_ = false;
    int world_size_ = 1;
    int rank_ = 0;
    int device_id_ = 0;
    
#ifdef LIGHT_MOE_USE_NCCL
    ncclComm_t nccl_comm_ = nullptr;
#endif
};

/**
 * @brief All-to-All communication for expert parallelism
 * 
 * Exchanges tokens between GPUs so each GPU processes its assigned experts.
 * Input tokens are distributed across GPUs, and each GPU sends tokens
 * to the GPU that hosts the target expert.
 * 
 * @param input Local input tokens [local_tokens, hidden_size]
 * @param output Received tokens for local experts [recv_tokens, hidden_size]
 * @param send_counts Number of tokens to send to each rank [world_size]
 * @param recv_counts Number of tokens to receive from each rank [world_size]
 * @param comm Communication group
 * @param stream CUDA stream
 * @return Status code
 */
Status all_to_all(
    const Tensor& input,
    Tensor& output,
    const int* send_counts,
    const int* recv_counts,
    const CommGroup& comm,
    cudaStream_t stream = nullptr);

/**
 * @brief Asynchronous All-to-All with double buffering
 * 
 * For overlapping communication with computation, this function
 * uses double buffering to hide communication latency.
 * 
 * @param input_buffer Input double buffer
 * @param output_buffer Output double buffer
 * @param buffer_idx Current buffer index (0 or 1)
 * @param send_counts Tokens to send per rank
 * @param recv_counts Tokens to receive per rank
 * @param comm Communication group
 * @param stream CUDA stream for communication
 * @return Status code
 */
Status all_to_all_async(
    Tensor* input_buffer,
    Tensor* output_buffer,
    int buffer_idx,
    const int* send_counts,
    const int* recv_counts,
    const CommGroup& comm,
    cudaStream_t stream = nullptr);

}  // namespace comm
}  // namespace light_moe
