#include <torch/extension.h>

#include "kernels/elementwise.cuh"

torch::Tensor add_one(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "add_one: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "add_one: only float32 is supported (M0 smoke test)");
    TORCH_CHECK(x.is_contiguous(), "add_one: input must be contiguous");

    auto out = torch::empty_like(x);
    launch_add_one(x.data_ptr<float>(), out.data_ptr<float>(), x.numel());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_one", &add_one, "Add 1.0 to every element of a float32 CUDA tensor (M0 toolchain smoke test)");
}
