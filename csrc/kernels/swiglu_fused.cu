#include "swiglu_fused.cuh"
#include "common.cuh"
#include "reduce.cuh"

namespace {
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
}  // namespace

// One warp per intermediate-channel row i: computes gate_i = gate_W[i]·x and
// up_i = up_W[i]·x in the SAME pass over x (one float4 load of x reused for
// both dot products, matching gemv_fp16_v3's vectorization), then
// h[i] = silu(gate_i) * up_i. silu(v) = v * sigmoid(v) = v / (1 + exp(-v)),
// computed in fp32.
__global__ void swiglu_gate_up_kernel(const half* __restrict__ gate_W, const half* __restrict__ up_W,
                                       const half* __restrict__ x, half* __restrict__ h, int I, int H) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * kWarpsPerBlock + warp_id;
    if (row >= I) return;

    const float4* g4 = reinterpret_cast<const float4*>(gate_W + static_cast<size_t>(row) * H);
    const float4* u4 = reinterpret_cast<const float4*>(up_W + static_cast<size_t>(row) * H);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int H8 = H / 8;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int k = lane; k < H8; k += 32) {
        const float4 gv = g4[k];
        const float4 uv = u4[k];
        const float4 xv = x4[k];
        const half2* gh = reinterpret_cast<const half2*>(&gv);
        const half2* uh = reinterpret_cast<const half2*>(&uv);
        const half2* xh = reinterpret_cast<const half2*>(&xv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 gf = __half22float2(gh[j]);
            const float2 uf = __half22float2(uh[j]);
            const float2 xf = __half22float2(xh[j]);
            gate_acc += gf.x * xf.x + gf.y * xf.y;
            up_acc += uf.x * xf.x + uf.y * xf.y;
        }
    }
    gate_acc = warp_reduce_sum(gate_acc);
    up_acc = warp_reduce_sum(up_acc);
    if (lane == 0) {
        const float silu = gate_acc / (1.0f + expf(-gate_acc));
        h[row] = __float2half(silu * up_acc);
    }
}

void launch_swiglu_gate_up(const half* gate_W, const half* up_W, const half* x, half* h, int I, int H) {
    const int blocks = ceil_div(I, kWarpsPerBlock);
    swiglu_gate_up_kernel<<<blocks, kThreadsPerBlock>>>(gate_W, up_W, x, h, I, H);
    CUDA_CHECK(cudaGetLastError());
}
