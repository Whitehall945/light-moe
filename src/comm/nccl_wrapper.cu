/**
 * @file nccl_wrapper.cu
 * @brief NCCL communication wrapper for Expert Parallelism
 * 
 * Implements All-to-All communication for distributing tokens to experts
 * across multiple GPUs.
 */

#include <cuda_runtime.h>
#include <nccl.h>

#include "light_moe/comm/all_to_all.h"

namespace light_moe {
namespace comm {

// ============================================================================
// Error Checking
// ============================================================================

#define NCCL_CHECK(cmd) do {                         \
    ncclResult_t r = cmd;                            \
    if (r != ncclSuccess) {                          \
        return Status::kNcclError;                   \
    }                                                \
} while(0)

#define CUDA_CHECK(cmd) do {                         \
    cudaError_t e = cmd;                             \
    if (e != cudaSuccess) {                          \
        return Status::kCudaError;                   \
    }                                                \
} while(0)

// ============================================================================
// NCCL Context
// ============================================================================

/**
 * @brief NCCL communicator context
 */
struct NcclContext {
    ncclComm_t comm;
    int rank;
    int world_size;
    cudaStream_t stream;
    
    bool initialized = false;
};

static NcclContext g_context;

// ============================================================================
// Initialization
// ============================================================================

Status init_nccl(int rank, int world_size, const char* nccl_id) {
    if (g_context.initialized) {
        return Status::kSuccess;  // Already initialized
    }
    
    g_context.rank = rank;
    g_context.world_size = world_size;
    
    // Create CUDA stream
    CUDA_CHECK(cudaStreamCreate(&g_context.stream));
    
    // Initialize NCCL communicator
    ncclUniqueId id;
    if (nccl_id != nullptr) {
        // Use provided ID (received from rank 0)
        memcpy(&id, nccl_id, sizeof(ncclUniqueId));
    } else if (rank == 0) {
        // Generate new ID on rank 0
        NCCL_CHECK(ncclGetUniqueId(&id));
    }
    
    NCCL_CHECK(ncclCommInitRank(&g_context.comm, world_size, id, rank));
    
    g_context.initialized = true;
    return Status::kSuccess;
}

Status shutdown_nccl() {
    if (!g_context.initialized) {
        return Status::kSuccess;
    }
    
    NCCL_CHECK(ncclCommDestroy(g_context.comm));
    CUDA_CHECK(cudaStreamDestroy(g_context.stream));
    
    g_context.initialized = false;
    return Status::kSuccess;
}

// ============================================================================
// All-to-All Communication
// ============================================================================

/**
 * @brief All-to-All operation for MoE token routing
 * 
 * Each GPU sends different portions of its tokens to different GPUs,
 * and receives tokens from all other GPUs.
 * 
 * Implementation uses NCCL point-to-point operations since NCCL doesn't
 * have native All-to-All.
 */
Status all_to_all(
    const Tensor& send_buffer,
    Tensor& recv_buffer,
    const int* send_counts,
    const int* recv_counts,
    const AllToAllConfig& config,
    cudaStream_t stream) {
    
    if (!g_context.initialized) {
        return Status::kInternalError;
    }
    
    int world_size = g_context.world_size;
    int rank = g_context.rank;
    
    // Use provided stream or default
    cudaStream_t s = (stream != nullptr) ? stream : g_context.stream;
    
    // Get data type
    ncclDataType_t nccl_dtype;
    size_t elem_size;
    switch (config.dtype) {
        case DataType::kFloat16:
            nccl_dtype = ncclFloat16;
            elem_size = 2;
            break;
        case DataType::kFloat32:
            nccl_dtype = ncclFloat32;
            elem_size = 4;
            break;
        case DataType::kBFloat16:
            nccl_dtype = ncclBfloat16;
            elem_size = 2;
            break;
        default:
            return Status::kInvalidArgument;
    }
    
    // Compute send/recv offsets
    int hidden_size = config.hidden_size;
    
    int send_offsets[world_size];
    int recv_offsets[world_size];
    send_offsets[0] = recv_offsets[0] = 0;
    for (int i = 1; i < world_size; i++) {
        send_offsets[i] = send_offsets[i-1] + send_counts[i-1] * hidden_size;
        recv_offsets[i] = recv_offsets[i-1] + recv_counts[i-1] * hidden_size;
    }
    
    // Start NCCL group
    NCCL_CHECK(ncclGroupStart());
    
    // Exchange with all peers
    for (int peer = 0; peer < world_size; peer++) {
        int send_count = send_counts[peer] * hidden_size;
        int recv_count = recv_counts[peer] * hidden_size;
        
        // Send to peer
        if (send_count > 0) {
            const void* send_ptr = static_cast<const char*>(send_buffer.data()) + 
                                   send_offsets[peer] * elem_size;
            NCCL_CHECK(ncclSend(send_ptr, send_count, nccl_dtype, peer, g_context.comm, s));
        }
        
        // Receive from peer
        if (recv_count > 0) {
            void* recv_ptr = static_cast<char*>(recv_buffer.data()) + 
                             recv_offsets[peer] * elem_size;
            NCCL_CHECK(ncclRecv(recv_ptr, recv_count, nccl_dtype, peer, g_context.comm, s));
        }
    }
    
    NCCL_CHECK(ncclGroupEnd());
    
    return Status::kSuccess;
}

/**
 * @brief Asynchronous All-to-All with double buffering support
 */
Status all_to_all_async(
    const Tensor& send_buffer,
    Tensor& recv_buffer,
    const int* send_counts,
    const int* recv_counts,
    const AllToAllConfig& config,
    cudaStream_t stream,
    cudaEvent_t completion_event) {
    
    // Perform the All-to-All
    Status status = all_to_all(send_buffer, recv_buffer, send_counts, recv_counts, config, stream);
    if (status != Status::kSuccess) {
        return status;
    }
    
    // Record completion event if provided
    if (completion_event != nullptr) {
        CUDA_CHECK(cudaEventRecord(completion_event, stream));
    }
    
    return Status::kSuccess;
}

/**
 * @brief Get recommended chunk size for overlapped communication
 */
int get_optimal_chunk_size(int total_tokens, int hidden_size, int world_size) {
    // Heuristic: aim for chunks that saturate NVLink bandwidth
    // NVLink 2.0 (2080 Ti): ~50 GB/s per direction
    // Target chunk time: ~100us for good overlap
    
    constexpr int64_t target_bytes = 50 * 1024 * 1024 / 10000;  // ~5KB per chunk
    int elem_bytes = 2;  // FP16
    
    int chunk_tokens = target_bytes / (hidden_size * elem_bytes);
    chunk_tokens = std::max(chunk_tokens, 64);  // Minimum 64 tokens
    chunk_tokens = std::min(chunk_tokens, total_tokens / world_size);  // Max 1/world_size
    
    // Round to power of 2 for efficiency
    int power = 1;
    while (power < chunk_tokens) power *= 2;
    return power / 2;
}

}  // namespace comm
}  // namespace light_moe
