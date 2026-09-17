#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#include <cmath>
#include <tuple>

#include "kernels/decode_attention.cuh"
#include "kernels/elementwise.cuh"
#include "kernels/gather_rows.cuh"
#include "kernels/gemv_dip_fused.cuh"
#include "kernels/gemv_fp16.cuh"
#include "kernels/gemv_sparse_accumulate.cuh"
#include "kernels/gemv_w4a16_group.cuh"
#include "kernels/gemv_w8a16.cuh"
#include "kernels/kv_cache.cuh"
#include "kernels/reduce_demo.cuh"
#include "kernels/rmsnorm.cuh"
#include "kernels/rope.cuh"
#include "kernels/softmax_online.cuh"
#include "kernels/strided_copy.cuh"
#include "kernels/swiglu_fused.cuh"
#include "kernels/topk_select.cuh"
#include "kernels/transpose.cuh"

namespace {

void check_f32_cuda_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, ": input must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, ": only float32 is supported");
    TORCH_CHECK(t.is_contiguous(), name, ": input must be contiguous");
}

void check_f16_cuda_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, ": input must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat16, name, ": only float16 is supported");
    TORCH_CHECK(t.is_contiguous(), name, ": input must be contiguous");
}

const half* half_ptr(const torch::Tensor& t) {
    return reinterpret_cast<const half*>(t.data_ptr<at::Half>());
}

half* half_ptr(torch::Tensor& t) {
    return reinterpret_cast<half*>(t.data_ptr<at::Half>());
}

}  // namespace

torch::Tensor add_one(torch::Tensor x) {
    check_f32_cuda_contiguous(x, "add_one");
    auto out = torch::empty_like(x);
    launch_add_one(x.data_ptr<float>(), out.data_ptr<float>(), x.numel());
    return out;
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    check_f32_cuda_contiguous(a, "vector_add(a)");
    check_f32_cuda_contiguous(b, "vector_add(b)");
    TORCH_CHECK(a.numel() == b.numel(), "vector_add: a and b must have the same number of elements");
    auto out = torch::empty_like(a);
    launch_vector_add(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), a.numel());
    return out;
}

torch::Tensor strided_copy(torch::Tensor in, int64_t stride) {
    check_f32_cuda_contiguous(in, "strided_copy");
    TORCH_CHECK(stride >= 1, "strided_copy: stride must be >= 1");
    const int64_t n = in.numel() / stride;
    auto out = torch::empty({n}, in.options());
    launch_strided_copy(in.data_ptr<float>(), out.data_ptr<float>(), n, stride);
    return out;
}

torch::Tensor reduce_naive_atomic(torch::Tensor in) {
    check_f32_cuda_contiguous(in, "reduce_naive_atomic");
    auto out = torch::zeros({1}, in.options());
    launch_reduce_naive_atomic(in.data_ptr<float>(), out.data_ptr<float>(), in.numel());
    return out;
}

torch::Tensor reduce_shared_tree(torch::Tensor in) {
    check_f32_cuda_contiguous(in, "reduce_shared_tree");
    auto out = torch::zeros({1}, in.options());
    launch_reduce_shared_tree(in.data_ptr<float>(), out.data_ptr<float>(), in.numel());
    return out;
}

torch::Tensor reduce_warp_shuffle(torch::Tensor in) {
    check_f32_cuda_contiguous(in, "reduce_warp_shuffle");
    auto out = torch::zeros({1}, in.options());
    launch_reduce_warp_shuffle(in.data_ptr<float>(), out.data_ptr<float>(), in.numel());
    return out;
}

torch::Tensor reduce_vectorized(torch::Tensor in) {
    check_f32_cuda_contiguous(in, "reduce_vectorized");
    TORCH_CHECK(in.numel() % 4 == 0, "reduce_vectorized: numel must be a multiple of 4");
    auto out = torch::zeros({1}, in.options());
    launch_reduce_vectorized(in.data_ptr<float>(), out.data_ptr<float>(), in.numel());
    return out;
}

namespace {

torch::Tensor transpose_impl(torch::Tensor in, void (*launcher)(const float*, float*, int)) {
    check_f32_cuda_contiguous(in, "transpose");
    TORCH_CHECK(in.dim() == 2 && in.size(0) == in.size(1), "transpose: input must be a square 2D matrix");
    auto out = torch::empty_like(in);
    launcher(in.data_ptr<float>(), out.data_ptr<float>(), static_cast<int>(in.size(0)));
    return out;
}

}  // namespace

torch::Tensor transpose_naive(torch::Tensor in) {
    return transpose_impl(in, launch_transpose_naive);
}

torch::Tensor transpose_unpadded(torch::Tensor in) {
    return transpose_impl(in, launch_transpose_unpadded);
}

torch::Tensor transpose_padded(torch::Tensor in) {
    return transpose_impl(in, launch_transpose_padded);
}

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor weight, double eps) {
    check_f16_cuda_contiguous(x, "rmsnorm(x)");
    check_f16_cuda_contiguous(weight, "rmsnorm(weight)");
    TORCH_CHECK(x.dim() == 2, "rmsnorm: x must be 2D [rows, hidden]");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1),
                "rmsnorm: weight shape must match x's hidden dim");
    TORCH_CHECK(x.size(1) % 2 == 0, "rmsnorm: hidden dim must be even (half2 vectorization)");
    auto out = torch::empty_like(x);
    launch_rmsnorm(half_ptr(x), half_ptr(weight), half_ptr(out), static_cast<int>(x.size(0)),
                   static_cast<int>(x.size(1)), static_cast<float>(eps));
    return out;
}

namespace {

torch::Tensor softmax_impl(torch::Tensor x, void (*launcher)(const half*, half*, int, int)) {
    check_f16_cuda_contiguous(x, "softmax");
    TORCH_CHECK(x.dim() == 2, "softmax: x must be 2D [rows, cols]");
    auto out = torch::empty_like(x);
    launcher(half_ptr(x), half_ptr(out), static_cast<int>(x.size(0)), static_cast<int>(x.size(1)));
    return out;
}

}  // namespace

torch::Tensor softmax_twopass(torch::Tensor x) {
    return softmax_impl(x, launch_softmax_twopass);
}

torch::Tensor softmax_online(torch::Tensor x) {
    return softmax_impl(x, launch_softmax_online);
}

namespace {

void check_gemv_shapes(const torch::Tensor& W, const torch::Tensor& x, const char* name) {
    check_f16_cuda_contiguous(W, name);
    check_f16_cuda_contiguous(x, name);
    TORCH_CHECK(W.dim() == 2, name, ": W must be 2D [N, K]");
    TORCH_CHECK(x.dim() == 1 && x.size(0) == W.size(1), name, ": x must be 1D [K] matching W's K");
}

}  // namespace

torch::Tensor gemv_fp16_v1(torch::Tensor W, torch::Tensor x) {
    check_gemv_shapes(W, x, "gemv_fp16_v1");
    auto y = torch::empty({W.size(0)}, W.options());
    launch_gemv_fp16_v1(half_ptr(W), half_ptr(x), half_ptr(y), static_cast<int>(W.size(0)),
                         static_cast<int>(W.size(1)));
    return y;
}

torch::Tensor gemv_fp16_v2(torch::Tensor W, torch::Tensor x) {
    check_gemv_shapes(W, x, "gemv_fp16_v2");
    auto y = torch::empty({W.size(0)}, W.options());
    launch_gemv_fp16_v2(half_ptr(W), half_ptr(x), half_ptr(y), static_cast<int>(W.size(0)),
                         static_cast<int>(W.size(1)));
    return y;
}

torch::Tensor gemv_fp16_v3(torch::Tensor W, torch::Tensor x) {
    check_gemv_shapes(W, x, "gemv_fp16_v3");
    TORCH_CHECK(W.size(1) % 8 == 0, "gemv_fp16_v3: K must be a multiple of 8 (float4-vectorized loads)");
    auto y = torch::empty({W.size(0)}, W.options());
    launch_gemv_fp16_v3(half_ptr(W), half_ptr(x), half_ptr(y), static_cast<int>(W.size(0)),
                         static_cast<int>(W.size(1)));
    return y;
}

torch::Tensor gemv_fp16_v4_splitk(torch::Tensor W, torch::Tensor x, int64_t split) {
    check_gemv_shapes(W, x, "gemv_fp16_v4_splitk");
    TORCH_CHECK(split >= 1, "gemv_fp16_v4_splitk: split must be >= 1");
    const int64_t N = W.size(0);
    auto y_accum = torch::zeros({N}, W.options().dtype(torch::kFloat32));
    launch_gemv_fp16_v4_splitk(half_ptr(W), half_ptr(x), y_accum.data_ptr<float>(), static_cast<int>(N),
                                static_cast<int>(W.size(1)), static_cast<int>(split));
    return y_accum.to(torch::kFloat16);
}

namespace {

// scale: [N, num_groups] fp32, or [1, num_groups] to broadcast one scale row
// to every output row (per_tensor). *num_groups/*scale_row_stride are set on
// return.
void check_quant_scale(const torch::Tensor& scale, int64_t N, const char* name, int64_t* num_groups,
                        int64_t* scale_row_stride) {
    TORCH_CHECK(scale.is_cuda(), name, ": scale must be a CUDA tensor");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, name, ": scale must be float32");
    TORCH_CHECK(scale.is_contiguous(), name, ": scale must be contiguous");
    TORCH_CHECK(scale.dim() == 2, name, ": scale must be 2D [N or 1, num_groups]");
    TORCH_CHECK(scale.size(0) == N || scale.size(0) == 1, name,
                ": scale.size(0) must be N (per-channel/group) or 1 (per-tensor broadcast)");
    *num_groups = scale.size(1);
    *scale_row_stride = (scale.size(0) == 1) ? 0 : *num_groups;
}

}  // namespace

torch::Tensor gemv_w8a16(torch::Tensor Wq, torch::Tensor scale, torch::Tensor x, int64_t group_size) {
    TORCH_CHECK(Wq.is_cuda(), "gemv_w8a16(Wq): must be a CUDA tensor");
    TORCH_CHECK(Wq.scalar_type() == torch::kInt8, "gemv_w8a16(Wq): must be int8");
    TORCH_CHECK(Wq.is_contiguous(), "gemv_w8a16(Wq): must be contiguous");
    TORCH_CHECK(Wq.dim() == 2, "gemv_w8a16(Wq): must be 2D [N, K]");
    check_f16_cuda_contiguous(x, "gemv_w8a16(x)");
    TORCH_CHECK(x.dim() == 1 && x.size(0) == Wq.size(1), "gemv_w8a16: x must be 1D [K] matching Wq's K");
    TORCH_CHECK(group_size >= 1, "gemv_w8a16: group_size must be >= 1");

    const int64_t N = Wq.size(0), K = Wq.size(1);
    int64_t num_groups = 0, scale_row_stride = 0;
    check_quant_scale(scale, N, "gemv_w8a16", &num_groups, &scale_row_stride);
    TORCH_CHECK(num_groups == 1 || group_size % 4 == 0, "gemv_w8a16: group_size must be a multiple of 4 when "
                                                          "num_groups > 1 (kernel reads 4 packed int8 per uint32 "
                                                          "word and must never span two quant groups)");
    auto y = torch::empty({N}, x.options());
    launch_gemv_w8a16(Wq.data_ptr<int8_t>(), scale.data_ptr<float>(), half_ptr(x), half_ptr(y),
                       static_cast<int>(N), static_cast<int>(K), static_cast<int>(group_size),
                       static_cast<int>(num_groups), static_cast<int>(scale_row_stride));
    return y;
}

namespace {

torch::Tensor gemv_w4a16_group_impl(torch::Tensor Wq_packed, torch::Tensor scale, torch::Tensor x, int64_t K,
                                     int64_t group_size, const char* name,
                                     void (*launcher)(const uint8_t*, const float*, const half*, half*, int, int,
                                                       int, int)) {
    TORCH_CHECK(Wq_packed.is_cuda(), name, "(Wq_packed): must be a CUDA tensor");
    TORCH_CHECK(Wq_packed.scalar_type() == torch::kUInt8, name, "(Wq_packed): must be uint8 (AWQ-packed int4)");
    TORCH_CHECK(Wq_packed.is_contiguous(), name, "(Wq_packed): must be contiguous");
    TORCH_CHECK(Wq_packed.dim() == 2, name, "(Wq_packed): must be 2D [N, ceil(K/8)*4]");
    check_f16_cuda_contiguous(x, "x");
    TORCH_CHECK(x.dim() == 1 && x.size(0) == K, name, ": x must be 1D [K]");
    TORCH_CHECK(K >= 1, name, ": K must be >= 1");
    TORCH_CHECK(group_size >= 1 && group_size % 8 == 0, name, ": group_size must be a positive multiple of 8");

    const int64_t N = Wq_packed.size(0);
    const int64_t expected_bytes = ((K + 7) / 8) * 4;
    TORCH_CHECK(Wq_packed.size(1) == expected_bytes, name, ": Wq_packed.size(1) must be ceil(K/8)*4 for the given K");
    TORCH_CHECK(scale.is_cuda(), name, ": scale must be a CUDA tensor");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, name, ": scale must be float32");
    TORCH_CHECK(scale.is_contiguous(), name, ": scale must be contiguous");
    TORCH_CHECK(scale.dim() == 2 && scale.size(0) == N, name,
                ": scale must be 2D [N, num_groups] (grouped int4 has no per-tensor broadcast)");
    const int64_t num_groups = scale.size(1);

    auto y = torch::empty({N}, x.options());
    launcher(Wq_packed.data_ptr<uint8_t>(), scale.data_ptr<float>(), half_ptr(x), half_ptr(y),
             static_cast<int>(N), static_cast<int>(K), static_cast<int>(group_size), static_cast<int>(num_groups));
    return y;
}

}  // namespace

torch::Tensor gemv_w4a16_group(torch::Tensor Wq_packed, torch::Tensor scale, torch::Tensor x, int64_t K,
                                int64_t group_size) {
    return gemv_w4a16_group_impl(Wq_packed, scale, x, K, group_size, "gemv_w4a16_group", launch_gemv_w4a16_group);
}

torch::Tensor gemv_w4a16_group_lop3(torch::Tensor Wq_packed, torch::Tensor scale, torch::Tensor x, int64_t K,
                                     int64_t group_size) {
    return gemv_w4a16_group_impl(Wq_packed, scale, x, K, group_size, "gemv_w4a16_group_lop3",
                                  launch_gemv_w4a16_group_lop3);
}

torch::Tensor gemv_w4a16_sparse_accumulate(torch::Tensor Wq_selected, torch::Tensor scale_selected,
                                            torch::Tensor h_selected, int64_t H, int64_t group_size) {
    const char* name = "gemv_w4a16_sparse_accumulate";
    TORCH_CHECK(Wq_selected.is_cuda() && Wq_selected.scalar_type() == torch::kUInt8 && Wq_selected.is_contiguous(),
                name, "(Wq_selected): must be a contiguous CUDA uint8 tensor");
    TORCH_CHECK(Wq_selected.dim() == 2, name, "(Wq_selected): must be 2D [k, ceil(H/8)*4]");
    TORCH_CHECK(H >= 1, name, ": H must be >= 1");
    TORCH_CHECK(group_size >= 1 && group_size % 8 == 0, name, ": group_size must be a positive multiple of 8");

    const int64_t k = Wq_selected.size(0);
    const int64_t expected_bytes = ((H + 7) / 8) * 4;
    TORCH_CHECK(Wq_selected.size(1) == expected_bytes, name, ": Wq_selected.size(1) must be ceil(H/8)*4 for the given H");
    TORCH_CHECK(scale_selected.is_cuda() && scale_selected.scalar_type() == torch::kFloat32 && scale_selected.is_contiguous(),
                name, "(scale_selected): must be a contiguous CUDA float32 tensor");
    TORCH_CHECK(scale_selected.dim() == 2 && scale_selected.size(0) == k, name,
                ": scale_selected must be 2D [k, num_groups]");
    check_f16_cuda_contiguous(h_selected, "gemv_w4a16_sparse_accumulate(h_selected)");
    TORCH_CHECK(h_selected.dim() == 1 && h_selected.size(0) == k, name, ": h_selected must be 1D [k]");
    const int64_t num_groups = scale_selected.size(1);

    auto y = torch::empty({H}, h_selected.options());
    launch_gemv_w4a16_sparse_accumulate(Wq_selected.data_ptr<uint8_t>(), scale_selected.data_ptr<float>(),
                                         half_ptr(h_selected), half_ptr(y), static_cast<int>(H), static_cast<int>(k),
                                         static_cast<int>(group_size), static_cast<int>(num_groups));
    return y;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> build_dip_descriptors(torch::Tensor selected_indices,
                                                                               torch::Tensor slot_of) {
    const char* name = "build_dip_descriptors";
    TORCH_CHECK(selected_indices.is_cuda() && selected_indices.scalar_type() == torch::kInt32 &&
                    selected_indices.dim() == 1 && selected_indices.is_contiguous(),
                name, "(selected_indices): must be a contiguous 1D int32 CUDA tensor");
    TORCH_CHECK(slot_of.is_cuda() && slot_of.scalar_type() == torch::kInt32 && slot_of.dim() == 1 &&
                    slot_of.is_contiguous(),
                name, "(slot_of): must be a contiguous 1D int32 CUDA tensor");

    const int64_t k = selected_indices.size(0);
    auto descriptors = torch::empty({k}, selected_indices.options());
    auto miss_channels = torch::empty({k}, selected_indices.options());
    auto miss_count = torch::empty({1}, selected_indices.options());
    launch_build_dip_descriptors(selected_indices.data_ptr<int32_t>(), static_cast<int>(k),
                                  slot_of.data_ptr<int32_t>(), descriptors.data_ptr<int32_t>(),
                                  miss_channels.data_ptr<int32_t>(), miss_count.data_ptr<int32_t>());
    return {descriptors, miss_channels, miss_count};
}

namespace {

void check_dip_fused_common(const torch::Tensor& cache_Wq, const torch::Tensor& cache_scale,
                             const torch::Tensor& staging_Wq, const torch::Tensor& staging_scale,
                             const torch::Tensor& descriptors, int64_t row_dim, const char* name) {
    const int64_t expected_bytes = ((row_dim + 7) / 8) * 4;
    TORCH_CHECK(cache_Wq.is_cuda() && cache_Wq.scalar_type() == torch::kUInt8 && cache_Wq.dim() == 2 &&
                    cache_Wq.is_contiguous() && cache_Wq.size(1) == expected_bytes,
                name, "(cache_Wq): must be a contiguous 2D uint8 CUDA tensor [C, ceil(row_dim/8)*4]");
    TORCH_CHECK(staging_Wq.is_cuda() && staging_Wq.scalar_type() == torch::kUInt8 && staging_Wq.dim() == 2 &&
                    staging_Wq.is_contiguous() && staging_Wq.size(1) == expected_bytes,
                name, "(staging_Wq): must be a contiguous 2D uint8 CUDA tensor [M, ceil(row_dim/8)*4]");
    TORCH_CHECK(cache_scale.is_cuda() && cache_scale.scalar_type() == torch::kFloat32 && cache_scale.dim() == 2 &&
                    cache_scale.is_contiguous(),
                name, "(cache_scale): must be a contiguous 2D float32 CUDA tensor [C, num_groups]");
    TORCH_CHECK(staging_scale.is_cuda() && staging_scale.scalar_type() == torch::kFloat32 &&
                    staging_scale.dim() == 2 && staging_scale.is_contiguous(),
                name, "(staging_scale): must be a contiguous 2D float32 CUDA tensor [M, num_groups]");
    TORCH_CHECK(cache_scale.size(1) == staging_scale.size(1), name,
                ": cache_scale and staging_scale must have the same num_groups");
    TORCH_CHECK(descriptors.is_cuda() && descriptors.scalar_type() == torch::kInt32 && descriptors.dim() == 1 &&
                    descriptors.is_contiguous(),
                name, "(descriptors): must be a contiguous 1D int32 CUDA tensor");
}

}  // namespace

torch::Tensor gemv_dip_fused_up(torch::Tensor cache_Wq, torch::Tensor cache_scale, torch::Tensor staging_Wq,
                                 torch::Tensor staging_scale, torch::Tensor descriptors, torch::Tensor x, int64_t K,
                                 int64_t group_size) {
    const char* name = "gemv_dip_fused_up";
    check_dip_fused_common(cache_Wq, cache_scale, staging_Wq, staging_scale, descriptors, K, name);
    check_f16_cuda_contiguous(x, "gemv_dip_fused_up(x)");
    TORCH_CHECK(x.dim() == 1 && x.size(0) == K, name, ": x must be 1D [K]");
    TORCH_CHECK(group_size >= 1 && group_size % 8 == 0, name, ": group_size must be a positive multiple of 8");

    const int64_t k = descriptors.size(0);
    const int64_t num_groups = cache_scale.size(1);
    auto y = torch::empty({k}, x.options());
    launch_gemv_dip_fused_up(cache_Wq.data_ptr<uint8_t>(), cache_scale.data_ptr<float>(),
                              staging_Wq.data_ptr<uint8_t>(), staging_scale.data_ptr<float>(),
                              descriptors.data_ptr<int32_t>(), half_ptr(x), half_ptr(y), static_cast<int>(k),
                              static_cast<int>(K), static_cast<int>(group_size), static_cast<int>(num_groups));
    return y;
}

torch::Tensor gemv_dip_fused_down(torch::Tensor cache_Wq, torch::Tensor cache_scale, torch::Tensor staging_Wq,
                                   torch::Tensor staging_scale, torch::Tensor descriptors, torch::Tensor h_selected,
                                   int64_t H, int64_t group_size) {
    const char* name = "gemv_dip_fused_down";
    check_dip_fused_common(cache_Wq, cache_scale, staging_Wq, staging_scale, descriptors, H, name);
    check_f16_cuda_contiguous(h_selected, "gemv_dip_fused_down(h_selected)");
    const int64_t k = descriptors.size(0);
    TORCH_CHECK(h_selected.dim() == 1 && h_selected.size(0) == k, name, ": h_selected must be 1D [k]");
    TORCH_CHECK(H >= 1, name, ": H must be >= 1");
    TORCH_CHECK(group_size >= 1 && group_size % 8 == 0, name, ": group_size must be a positive multiple of 8");

    const int64_t num_groups = cache_scale.size(1);
    auto y = torch::empty({H}, h_selected.options());
    launch_gemv_dip_fused_down(cache_Wq.data_ptr<uint8_t>(), cache_scale.data_ptr<float>(),
                                staging_Wq.data_ptr<uint8_t>(), staging_scale.data_ptr<float>(),
                                descriptors.data_ptr<int32_t>(), half_ptr(h_selected), half_ptr(y),
                                static_cast<int>(H), static_cast<int>(k), static_cast<int>(group_size),
                                static_cast<int>(num_groups));
    return y;
}

torch::Tensor swiglu_gate_up(torch::Tensor gate_W, torch::Tensor up_W, torch::Tensor x) {
    check_f16_cuda_contiguous(gate_W, "swiglu_gate_up(gate_W)");
    check_f16_cuda_contiguous(up_W, "swiglu_gate_up(up_W)");
    check_f16_cuda_contiguous(x, "swiglu_gate_up(x)");
    TORCH_CHECK(gate_W.dim() == 2 && up_W.dim() == 2, "swiglu_gate_up: gate_W and up_W must be 2D [I, H]");
    TORCH_CHECK(gate_W.sizes() == up_W.sizes(), "swiglu_gate_up: gate_W and up_W must have the same shape");
    TORCH_CHECK(x.dim() == 1 && x.size(0) == gate_W.size(1), "swiglu_gate_up: x must be 1D [H] matching the weights' H");
    TORCH_CHECK(gate_W.size(1) % 8 == 0, "swiglu_gate_up: H must be a multiple of 8 (float4-vectorized loads)");

    const int64_t I = gate_W.size(0), H = gate_W.size(1);
    auto h = torch::empty({I}, x.options());
    launch_swiglu_gate_up(half_ptr(gate_W), half_ptr(up_W), half_ptr(x), half_ptr(h), static_cast<int>(I),
                           static_cast<int>(H));
    return h;
}

void kv_cache_append(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_new, torch::Tensor v_new,
                      int64_t pos) {
    check_f16_cuda_contiguous(k_cache, "kv_cache_append(k_cache)");
    check_f16_cuda_contiguous(v_cache, "kv_cache_append(v_cache)");
    check_f16_cuda_contiguous(k_new, "kv_cache_append(k_new)");
    check_f16_cuda_contiguous(v_new, "kv_cache_append(v_new)");
    TORCH_CHECK(k_cache.dim() == 3 && v_cache.sizes() == k_cache.sizes(),
                "kv_cache_append: k_cache/v_cache must be 3D [num_kv_heads, max_seq_len, head_dim] and same shape");
    TORCH_CHECK(k_new.sizes() == v_new.sizes(), "kv_cache_append: k_new/v_new must have the same shape");
    TORCH_CHECK(k_new.dim() == 2 && k_new.size(0) == k_cache.size(0) && k_new.size(1) == k_cache.size(2),
                "kv_cache_append: k_new/v_new must be 2D [num_kv_heads, head_dim] matching the cache");
    const int64_t num_kv_heads = k_cache.size(0), max_seq_len = k_cache.size(1), head_dim = k_cache.size(2);
    TORCH_CHECK(pos >= 0 && pos < max_seq_len, "kv_cache_append: pos out of range for max_seq_len");
    launch_kv_cache_append(half_ptr(k_new), half_ptr(v_new), half_ptr(k_cache), half_ptr(v_cache),
                            static_cast<int>(num_kv_heads), static_cast<int>(max_seq_len),
                            static_cast<int>(head_dim), static_cast<int>(pos));
}

torch::Tensor decode_attention(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, int64_t cur_len) {
    check_f16_cuda_contiguous(q, "decode_attention(q)");
    check_f16_cuda_contiguous(k_cache, "decode_attention(k_cache)");
    check_f16_cuda_contiguous(v_cache, "decode_attention(v_cache)");
    TORCH_CHECK(q.dim() == 2, "decode_attention: q must be 2D [num_q_heads, head_dim]");
    TORCH_CHECK(k_cache.dim() == 3 && v_cache.sizes() == k_cache.sizes(),
                "decode_attention: k_cache/v_cache must be 3D [num_kv_heads, max_seq_len, head_dim] and same shape");
    TORCH_CHECK(q.size(1) == k_cache.size(2), "decode_attention: q's head_dim must match the cache's head_dim");
    const int64_t num_q_heads = q.size(0), num_kv_heads = k_cache.size(0), max_seq_len = k_cache.size(1),
                  head_dim = k_cache.size(2);
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "decode_attention: num_q_heads must be a multiple of num_kv_heads");
    TORCH_CHECK(cur_len >= 1 && cur_len <= max_seq_len, "decode_attention: cur_len out of range for max_seq_len");

    auto out = torch::empty({num_q_heads, head_dim}, q.options());
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    launch_decode_attention(half_ptr(q), half_ptr(k_cache), half_ptr(v_cache), half_ptr(out),
                             static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads),
                             static_cast<int>(max_seq_len), static_cast<int>(head_dim), static_cast<int>(cur_len),
                             scale);
    return out;
}

void rope_apply(torch::Tensor x, torch::Tensor cos_vals, torch::Tensor sin_vals) {
    check_f16_cuda_contiguous(x, "rope_apply(x)");
    TORCH_CHECK(x.dim() == 2, "rope_apply: x must be 2D [num_heads, head_dim]");
    TORCH_CHECK(x.size(1) % 2 == 0, "rope_apply: head_dim must be even");
    TORCH_CHECK(cos_vals.is_cuda() && cos_vals.scalar_type() == torch::kFloat32 && cos_vals.is_contiguous(),
                "rope_apply: cos_vals must be a contiguous float32 CUDA tensor");
    TORCH_CHECK(sin_vals.sizes() == cos_vals.sizes() && sin_vals.is_cuda() &&
                    sin_vals.scalar_type() == torch::kFloat32 && sin_vals.is_contiguous(),
                "rope_apply: sin_vals must match cos_vals (contiguous float32 CUDA, same shape)");
    TORCH_CHECK(cos_vals.dim() == 1 && cos_vals.size(0) == x.size(1) / 2,
                "rope_apply: cos_vals/sin_vals must be 1D [head_dim / 2]");

    const int64_t num_heads = x.size(0), head_dim = x.size(1);
    launch_rope_apply(half_ptr(x), cos_vals.data_ptr<float>(), sin_vals.data_ptr<float>(), static_cast<int>(num_heads),
                       static_cast<int>(head_dim));
}

std::tuple<torch::Tensor, torch::Tensor> topk_threshold_select(torch::Tensor abs_g, int64_t k) {
    check_f32_cuda_contiguous(abs_g, "topk_threshold_select(abs_g)");
    TORCH_CHECK(abs_g.dim() == 1, "topk_threshold_select: abs_g must be 1D [I]");
    TORCH_CHECK(k >= 1 && k <= abs_g.size(0), "topk_threshold_select: k must be in [1, I]");

    const int64_t n = abs_g.size(0);
    auto out_indices = torch::empty({k}, abs_g.options().dtype(torch::kInt32));
    auto out_count = torch::empty({1}, abs_g.options().dtype(torch::kInt32));
    launch_topk_threshold_select(abs_g.data_ptr<float>(), static_cast<int>(n), static_cast<int>(k),
                                  out_indices.data_ptr<int32_t>(), out_count.data_ptr<int32_t>());
    return {out_indices, out_count};
}

namespace {

void check_gather_rows_common(const torch::Tensor& matrix, const torch::Tensor& indices,
                               const torch::Tensor& gpu_dst, const char* name) {
    TORCH_CHECK(matrix.is_cpu() && matrix.is_pinned(), name, "(matrix): must be pinned host memory");
    TORCH_CHECK(matrix.scalar_type() == torch::kUInt8 && matrix.dim() == 2 && matrix.is_contiguous(),
                name, "(matrix): must be a contiguous 2D uint8 tensor [num_rows, row_nbytes]");
    TORCH_CHECK(indices.is_cpu() && indices.scalar_type() == torch::kInt64 && indices.dim() == 1,
                name, "(indices): must be a 1D int64 CPU tensor");
    TORCH_CHECK(gpu_dst.is_cuda() && gpu_dst.scalar_type() == torch::kUInt8 && gpu_dst.is_contiguous(),
                name, "(gpu_dst): must be a contiguous CUDA uint8 tensor");
    const int64_t k = indices.size(0);
    const int64_t row_nbytes = matrix.size(1);
    TORCH_CHECK(gpu_dst.numel() >= k * row_nbytes, name, "(gpu_dst): too small for k * row_nbytes bytes");
}

}  // namespace

void gather_rows_staged(torch::Tensor matrix, torch::Tensor indices, torch::Tensor staging, torch::Tensor gpu_dst) {
    check_gather_rows_common(matrix, indices, gpu_dst, "gather_rows_staged");
    TORCH_CHECK(staging.is_cpu() && staging.is_pinned() && staging.scalar_type() == torch::kUInt8,
                "gather_rows_staged(staging): must be pinned host uint8 memory");
    const int64_t k = indices.size(0);
    const int64_t row_nbytes = matrix.size(1);
    TORCH_CHECK(staging.numel() >= k * row_nbytes, "gather_rows_staged(staging): too small for k * row_nbytes bytes");

    auto stream = at::cuda::getCurrentCUDAStream();
    launch_gather_rows_staged(matrix.data_ptr<uint8_t>(), indices.data_ptr<int64_t>(), k, row_nbytes,
                               staging.data_ptr<uint8_t>(), gpu_dst.data_ptr<uint8_t>(), stream.stream());
}

void gather_rows_naive(torch::Tensor matrix, torch::Tensor indices, torch::Tensor gpu_dst) {
    check_gather_rows_common(matrix, indices, gpu_dst, "gather_rows_naive");
    const int64_t k = indices.size(0);
    const int64_t row_nbytes = matrix.size(1);

    auto stream = at::cuda::getCurrentCUDAStream();
    launch_gather_rows_naive(matrix.data_ptr<uint8_t>(), indices.data_ptr<int64_t>(), k, row_nbytes,
                              gpu_dst.data_ptr<uint8_t>(), stream.stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_one", &add_one, "Add 1.0 to every element of a float32 CUDA tensor (M0 toolchain smoke test)");
    m.def("vector_add", &vector_add, "out = a + b, elementwise (M1)");
    m.def("strided_copy", &strided_copy, "out[i] = in[i*stride] -- demonstrates coalescing collapse (M1)");
    m.def("reduce_naive_atomic", &reduce_naive_atomic, "Sum reduction v1: naive atomic (M1)");
    m.def("reduce_shared_tree", &reduce_shared_tree, "Sum reduction v2: shared-memory tree (M1)");
    m.def("reduce_warp_shuffle", &reduce_warp_shuffle, "Sum reduction v3: warp-shuffle (M1)");
    m.def("reduce_vectorized", &reduce_vectorized, "Sum reduction v4: vectorized float4 + warp-shuffle (M1)");
    m.def("transpose_naive", &transpose_naive, "Square matrix transpose v1: naive (M1)");
    m.def("transpose_unpadded", &transpose_unpadded, "Square matrix transpose v2: tiled, bank-conflicted (M1)");
    m.def("transpose_padded", &transpose_padded, "Square matrix transpose v3: tiled, padded (M1)");
    m.def("rmsnorm", &rmsnorm, "RMSNorm, fp32-accumulated (M2)");
    m.def("softmax_twopass", &softmax_twopass, "Row-wise softmax v1: naive two-pass statistics (M2)");
    m.def("softmax_online", &softmax_online, "Row-wise softmax v2: online single-pass statistics (M2)");
    m.def("gemv_fp16_v1", &gemv_fp16_v1, "FP16 GEMV v1: one thread per row (M2)");
    m.def("gemv_fp16_v2", &gemv_fp16_v2, "FP16 GEMV v2: one warp per row, shuffle reduce (M2)");
    m.def("gemv_fp16_v3", &gemv_fp16_v3, "FP16 GEMV v3: v2 + float4-vectorized loads (M2)");
    m.def("gemv_fp16_v4_splitk", &gemv_fp16_v4_splitk, "FP16 GEMV v4: split-K with atomics (M2)");
    m.def("gemv_w8a16", &gemv_w8a16, "W8A16 GEMV: symmetric INT8 weights, FP16 activations (M4)");
    m.def("gemv_w4a16_group", &gemv_w4a16_group,
          "W4A16 grouped GEMV: AWQ-packed INT4 weights, scalar dequant, register-cached group scale (M4)");
    m.def("gemv_w4a16_group_lop3", &gemv_w4a16_group_lop3,
          "W4A16 grouped GEMV: same as gemv_w4a16_group but with bit-pattern-construction INT4->FP16 dequant (M4)");
    m.def("gemv_w4a16_sparse_accumulate", &gemv_w4a16_sparse_accumulate,
          "Sparse 'down' GEMV: y[H] = sum_i h[i] * W_T[i,:] over k selected (already row-gathered) rows of a "
          "transposed-stored down_proj (M7)");
    m.def("build_dip_descriptors", &build_dip_descriptors,
          "Resolves each selected channel to a (cache slot | staging miss-position) descriptor, compacting misses "
          "for a targeted gather (M8)");
    m.def("gemv_dip_fused_up", &gemv_dip_fused_up,
          "Descriptor-driven 'up' GEMV: reads each selected row from either the resident cache or the staging "
          "buffer in one pass, no host branching (M8)");
    m.def("gemv_dip_fused_down", &gemv_dip_fused_down,
          "Descriptor-driven 'down' GEMV: same cache-or-stream fusion as gemv_dip_fused_up, for the transposed-"
          "storage accumulate direction (M8)");
    m.def("swiglu_gate_up", &swiglu_gate_up,
          "Fused SwiGLU gate+up: h = silu(gate_W @ x) * (up_W @ x), one pass over x for both projections (M5)");
    m.def("kv_cache_append", &kv_cache_append,
          "Append one token's K/V into a contiguous [num_kv_heads, max_seq_len, head_dim] cache at `pos` (M5)");
    m.def("decode_attention", &decode_attention,
          "Single-query GQA decode attention over a KV cache, online-softmax, fp32 accumulation (M5)");
    m.def("rope_apply", &rope_apply,
          "Applies RoPE (rotate-half convention, matching HF exactly) to x [num_heads, head_dim] in place (M5)");
    m.def("topk_threshold_select", &topk_threshold_select,
          "Selects the k largest-magnitude indices via single-launch binary-search threshold + compact (M7)");
    m.def("gather_rows_staged", &gather_rows_staged,
          "Row gather v1: host memcpy into a contiguous pinned staging buffer, then one H2D cudaMemcpyAsync (M7)");
    m.def("gather_rows_naive", &gather_rows_naive,
          "Row gather baseline: one cudaMemcpyAsync per selected row, straight from the pinned arena (M7)");
}
