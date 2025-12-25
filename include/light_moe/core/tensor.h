#pragma once

/**
 * @file tensor.h
 * @brief Lightweight tensor wrapper for Light-MoE
 */

#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "light_moe/core/types.h"

namespace light_moe {

/**
 * @brief Lightweight tensor class wrapping CUDA memory
 * 
 * This is a simple tensor class for internal use. For Python interop,
 * we convert to/from PyTorch tensors via pybind11.
 */
class Tensor {
public:
    Tensor() = default;
    
    /**
     * @brief Construct a tensor with given shape and data type
     * @param shape Tensor dimensions
     * @param dtype Data type
     * @param device_id CUDA device ID (-1 for CPU)
     */
    Tensor(const std::vector<int64_t>& shape, DataType dtype, int device_id = 0);
    
    /**
     * @brief Construct a tensor from existing memory (no ownership)
     * @param data Pointer to data
     * @param shape Tensor dimensions
     * @param dtype Data type
     * @param device_id CUDA device ID
     */
    Tensor(void* data, const std::vector<int64_t>& shape, DataType dtype, int device_id);
    
    ~Tensor();
    
    // Disable copy, enable move
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Accessors
    void* data() { return data_; }
    const void* data() const { return data_; }
    
    template <typename T>
    T* data_ptr() { return static_cast<T*>(data_); }
    
    template <typename T>
    const T* data_ptr() const { return static_cast<const T*>(data_); }
    
    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t dim(int i) const { return shape_[i]; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    
    DataType dtype() const { return dtype_; }
    int device_id() const { return device_id_; }
    bool is_cuda() const { return device_id_ >= 0; }
    
    /**
     * @brief Get total number of elements
     */
    int64_t numel() const;
    
    /**
     * @brief Get size in bytes
     */
    size_t size_bytes() const;
    
    /**
     * @brief Copy data to another tensor
     */
    Status copy_to(Tensor& dst, cudaStream_t stream = nullptr) const;
    
    /**
     * @brief Fill tensor with zeros
     */
    Status zero_(cudaStream_t stream = nullptr);

private:
    void* data_ = nullptr;
    std::vector<int64_t> shape_;
    DataType dtype_ = DataType::kFloat32;
    int device_id_ = -1;
    bool owns_memory_ = false;
};

}  // namespace light_moe
