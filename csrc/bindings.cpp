#include <torch/extension.h>

#include "kernels/elementwise.cuh"
#include "kernels/reduce_demo.cuh"
#include "kernels/strided_copy.cuh"
#include "kernels/transpose.cuh"

namespace {

void check_f32_cuda_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, ": input must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, ": only float32 is supported");
    TORCH_CHECK(t.is_contiguous(), name, ": input must be contiguous");
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
}
