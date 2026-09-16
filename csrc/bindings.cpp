#include <torch/extension.h>
#include <cuda_fp16.h>

#include "kernels/elementwise.cuh"
#include "kernels/gemv_fp16.cuh"
#include "kernels/gemv_w4a16_group.cuh"
#include "kernels/gemv_w8a16.cuh"
#include "kernels/reduce_demo.cuh"
#include "kernels/rmsnorm.cuh"
#include "kernels/softmax_online.cuh"
#include "kernels/strided_copy.cuh"
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
}
