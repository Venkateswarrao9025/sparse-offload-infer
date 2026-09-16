#include "kv_cache.cuh"
#include "common.cuh"

// One block per KV head; threads copy that head's head_dim-wide K and V
// vectors into the cache at [head, pos, :]. head_dim is contiguous in both
// source and destination, so this is a fully coalesced copy.
__global__ void kv_cache_append_kernel(const half* __restrict__ k_new, const half* __restrict__ v_new,
                                        half* __restrict__ k_cache, half* __restrict__ v_cache, int max_seq_len,
                                        int head_dim, int pos) {
    const int h = blockIdx.x;
    const half* k_src = k_new + static_cast<size_t>(h) * head_dim;
    const half* v_src = v_new + static_cast<size_t>(h) * head_dim;
    half* k_dst = k_cache + (static_cast<size_t>(h) * max_seq_len + pos) * head_dim;
    half* v_dst = v_cache + (static_cast<size_t>(h) * max_seq_len + pos) * head_dim;

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        k_dst[i] = k_src[i];
        v_dst[i] = v_src[i];
    }
}

void launch_kv_cache_append(const half* k_new, const half* v_new, half* k_cache, half* v_cache, int num_kv_heads,
                             int max_seq_len, int head_dim, int pos) {
    const int threads = (head_dim < 256) ? head_dim : 256;
    kv_cache_append_kernel<<<num_kv_heads, threads>>>(k_new, v_new, k_cache, v_cache, max_seq_len, head_dim, pos);
    CUDA_CHECK(cudaGetLastError());
}
