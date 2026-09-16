#include "decode_attention.cuh"
#include "common.cuh"
#include "reduce.cuh"

// One block per query head, blockDim.x == head_dim (one thread per feature
// dimension). For each cached timestep t: every thread computes its
// dimension's product q[d]*k[t,d], block_reduce_sum combines all head_dim
// partials into one score (broadcast back to every thread via shared
// memory, since block_reduce_sum's result is only valid on thread 0), then
// every thread applies the same online-softmax rescale to its own slice of
// the running output accumulator -- the FlashAttention decode recurrence:
//   new_m = max(m, s); corr = exp(m - new_m); p = exp(s - new_m)
//   l = l*corr + p; acc[d] = acc[d]*corr + p*v[t,d]; m = new_m
// This is O(cur_len) block-wide syncs per head -- correct and simple, not
// yet fast (a real KV-cache-scan kernel would tile timesteps to cut sync
// count; that's a follow-up once this is verified against HF).
__global__ void decode_attention_kernel(const half* __restrict__ q, const half* __restrict__ k_cache,
                                         const half* __restrict__ v_cache, half* __restrict__ out, int num_q_heads,
                                         int num_kv_heads, int max_seq_len, int head_dim, int cur_len, float scale) {
    extern __shared__ float sdata[];  // (blockDim.x / 32) floats, for block_reduce_sum

    const int qh = blockIdx.x;
    const int n_rep = num_q_heads / num_kv_heads;
    const int kvh = qh / n_rep;
    const int tid = threadIdx.x;

    const float q_val = __half2float(q[static_cast<size_t>(qh) * head_dim + tid]);
    const half* k_head = k_cache + static_cast<size_t>(kvh) * max_seq_len * head_dim;
    const half* v_head = v_cache + static_cast<size_t>(kvh) * max_seq_len * head_dim;

    float m = -1e30f;  // see softmax_online.cu's kNegInfSentinel: avoids -inf - -inf = NaN
    float l = 0.0f;
    float acc = 0.0f;

    __shared__ float s_broadcast;
    for (int t = 0; t < cur_len; ++t) {
        const float partial = q_val * __half2float(k_head[static_cast<size_t>(t) * head_dim + tid]);
        const float total = block_reduce_sum(partial, sdata);
        if (tid == 0) {
            s_broadcast = total * scale;
        }
        __syncthreads();
        const float s = s_broadcast;

        const float new_m = fmaxf(m, s);
        const float correction = expf(m - new_m);
        const float p = expf(s - new_m);
        l = l * correction + p;
        acc = acc * correction + p * __half2float(v_head[static_cast<size_t>(t) * head_dim + tid]);
        m = new_m;
        __syncthreads();  // all threads must finish reading s_broadcast before the next iteration overwrites it
    }

    out[static_cast<size_t>(qh) * head_dim + tid] = __float2half(acc / l);
}

void launch_decode_attention(const half* q, const half* k_cache, const half* v_cache, half* out, int num_q_heads,
                              int num_kv_heads, int max_seq_len, int head_dim, int cur_len, float scale) {
    const size_t shmem = ceil_div(head_dim, 32) * sizeof(float);
    decode_attention_kernel<<<num_q_heads, head_dim, shmem>>>(q, k_cache, v_cache, out, num_q_heads, num_kv_heads,
                                                               max_seq_len, head_dim, cur_len, scale);
    CUDA_CHECK(cudaGetLastError());
}
