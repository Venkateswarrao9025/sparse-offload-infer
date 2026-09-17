"""Build the soinfer CUDA extension.

Target GPU is NVIDIA T4 (Turing, sm_75) by default -- PROJECT_SPEC.md sec 3.
Override with SOINFER_CUDA_ARCHS="75;80;86" (semicolon-separated) to build for
other/multiple architectures. Do not silently assume Ampere+ features (bf16,
cp.async) are available -- they are gated in csrc/include/common.cuh.
"""
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CSRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")


def _arch_flags() -> list[str]:
    archs = os.environ.get("SOINFER_CUDA_ARCHS", "75").split(";")
    return [f"-gencode=arch=compute_{a.strip()},code=sm_{a.strip()}" for a in archs if a.strip()]


ext_modules = [
    CUDAExtension(
        name="soinfer._C",
        sources=[
            os.path.join(CSRC_DIR, "bindings.cpp"),
            os.path.join(CSRC_DIR, "kernels", "elementwise.cu"),
            os.path.join(CSRC_DIR, "kernels", "strided_copy.cu"),
            os.path.join(CSRC_DIR, "kernels", "reduce_demo.cu"),
            os.path.join(CSRC_DIR, "kernels", "transpose.cu"),
            os.path.join(CSRC_DIR, "kernels", "rmsnorm.cu"),
            os.path.join(CSRC_DIR, "kernels", "softmax_online.cu"),
            os.path.join(CSRC_DIR, "kernels", "gemv_fp16.cu"),
            os.path.join(CSRC_DIR, "kernels", "gemv_w8a16.cu"),
            os.path.join(CSRC_DIR, "kernels", "gemv_w4a16_group.cu"),
            os.path.join(CSRC_DIR, "kernels", "swiglu_fused.cu"),
            os.path.join(CSRC_DIR, "kernels", "kv_cache.cu"),
            os.path.join(CSRC_DIR, "kernels", "decode_attention.cu"),
            os.path.join(CSRC_DIR, "kernels", "rope.cu"),
        ],
        include_dirs=[
            os.path.join(CSRC_DIR, "include"),
            os.path.join(CSRC_DIR, "kernels"),
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "-lineinfo"] + _arch_flags(),
        },
    )
]

setup(
    name="soinfer",
    version="0.0.1",
    description="Sparse-Offload Inference Engine: INT4/INT8 CUDA decode runtime with Dynamic Input Pruning",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=2.1",
        "transformers>=4.40",
        "accelerate>=0.30",
        "numpy",
        "matplotlib",
    ],
    python_requires=">=3.10",
)
