#include "gather_rows.cuh"
#include "common.cuh"

#include <cstring>

void launch_gather_rows_staged(const uint8_t* matrix_base, const int64_t* indices, int64_t k,
                                int64_t row_nbytes, uint8_t* staging, uint8_t* gpu_dst,
                                cudaStream_t stream) {
    for (int64_t i = 0; i < k; ++i) {
        std::memcpy(staging + i * row_nbytes, matrix_base + indices[i] * row_nbytes, row_nbytes);
    }
    CUDA_CHECK(cudaMemcpyAsync(gpu_dst, staging, k * row_nbytes, cudaMemcpyHostToDevice, stream));
}

void launch_gather_rows_naive(const uint8_t* matrix_base, const int64_t* indices, int64_t k,
                               int64_t row_nbytes, uint8_t* gpu_dst, cudaStream_t stream) {
    for (int64_t i = 0; i < k; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(gpu_dst + i * row_nbytes, matrix_base + indices[i] * row_nbytes,
                                    row_nbytes, cudaMemcpyHostToDevice, stream));
    }
}
