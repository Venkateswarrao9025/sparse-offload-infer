#include "softmax_online.cuh"
#include "common.cuh"
#include "reduce.cuh"

namespace {
constexpr int kThreads = 256;
}

// --- v1: naive two-pass ------------------------------------------------

__global__ void softmax_twopass_kernel(const half* __restrict__ x, half* __restrict__ out, int cols) {
    extern __shared__ float sdata[];  // (blockDim.x / 32) floats, reused across both reductions

    const int row = blockIdx.x;
    const half* x_row = x + static_cast<size_t>(row) * cols;
    half* out_row = out + static_cast<size_t>(row) * cols;

    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, __half2float(x_row[i]));
    }
    const float block_max = block_reduce_max(local_max, sdata);
    __shared__ float s_max;
    if (threadIdx.x == 0) {
        s_max = block_max;
    }
    __syncthreads();
    const float row_max = s_max;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_sum += expf(__half2float(x_row[i]) - row_max);
    }
    const float block_sum = block_reduce_sum(local_sum, sdata);
    __shared__ float s_sum;
    if (threadIdx.x == 0) {
        s_sum = block_sum;
    }
    __syncthreads();
    const float row_sum = s_sum;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float e = expf(__half2float(x_row[i]) - row_max);
        out_row[i] = __float2half(e / row_sum);
    }
}

void launch_softmax_twopass(const half* x, half* out, int rows, int cols) {
    const size_t shmem = (kThreads / 32) * sizeof(float);
    softmax_twopass_kernel<<<rows, kThreads, shmem>>>(x, out, cols);
    CUDA_CHECK(cudaGetLastError());
}

// --- v2: online single-pass statistics ----------------------------------

namespace {

struct SoftmaxState {
    float m;  // running max
    float l;  // running sum of exp(x - m)
};

// Finite stand-in for -infinity as the reduction identity's max: combining
// two identities computes (a.m - m) in the recurrence below, and
// -INFINITY - -INFINITY is NaN (poisons l even though fmaxf alone would
// resolve m correctly). A large finite negative value keeps that
// subtraction at exactly 0.
constexpr float kNegInfSentinel = -1e30f;

// Combines two (max, sum) states via the FlashAttention rescaling
// recurrence. Associative and commutative, so it works as a reduction op.
__device__ __forceinline__ SoftmaxState combine_softmax(SoftmaxState a, SoftmaxState b) {
    const float m = fmaxf(a.m, b.m);
    const float l = a.l * expf(a.m - m) + b.l * expf(b.m - m);
    return SoftmaxState{m, l};
}

__device__ __forceinline__ SoftmaxState warp_reduce_softmax(SoftmaxState val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        SoftmaxState other;
        other.m = __shfl_down_sync(0xffffffffu, val.m, offset);
        other.l = __shfl_down_sync(0xffffffffu, val.l, offset);
        val = combine_softmax(val, other);
    }
    return val;
}

__device__ __forceinline__ SoftmaxState block_reduce_softmax(SoftmaxState val, SoftmaxState* shared) {
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    val = warp_reduce_softmax(val);
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + 31) / 32;
    const SoftmaxState identity{kNegInfSentinel, 0.0f};
    val = (threadIdx.x < num_warps) ? shared[lane] : identity;
    if (warp_id == 0) {
        val = warp_reduce_softmax(val);
    }
    return val;
}

}  // namespace

__global__ void softmax_online_kernel(const half* __restrict__ x, half* __restrict__ out, int cols) {
    extern __shared__ SoftmaxState sdata_state[];  // (blockDim.x / 32) states

    const int row = blockIdx.x;
    const half* x_row = x + static_cast<size_t>(row) * cols;
    half* out_row = out + static_cast<size_t>(row) * cols;

    SoftmaxState local{kNegInfSentinel, 0.0f};
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float xi = __half2float(x_row[i]);
        local = combine_softmax(local, SoftmaxState{xi, 1.0f});
    }

    const SoftmaxState block_total = block_reduce_softmax(local, sdata_state);
    __shared__ SoftmaxState s_total;
    if (threadIdx.x == 0) {
        s_total = block_total;
    }
    __syncthreads();
    const SoftmaxState total = s_total;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float xi = __half2float(x_row[i]);
        out_row[i] = __float2half(expf(xi - total.m) / total.l);
    }
}

void launch_softmax_online(const half* x, half* out, int rows, int cols) {
    const size_t shmem = (kThreads / 32) * sizeof(SoftmaxState);
    softmax_online_kernel<<<rows, kThreads, shmem>>>(x, out, cols);
    CUDA_CHECK(cudaGetLastError());
}
