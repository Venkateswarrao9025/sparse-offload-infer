# Design notes and deviations from PROJECT_SPEC.md

This file records decisions made while building the project, and any place
where the spec turned out to be wrong or infeasible on the actual hardware
(per the working protocol in PROJECT_SPEC.md sec 10).

## Deviation: no local NVIDIA GPU

**Status:** active, as of 2026-09-14.

The primary dev machine (Windows, this repo's working directory) has an Intel
Arc integrated GPU only -- no NVIDIA hardware, no `nvcc`, no `nvidia-smi`.
PROJECT_SPEC.md assumes a T4 or similar is available locally or via
Colab/Kaggle/Lambda; here it is Colab/Kaggle exclusively, and *only* via
Colab/Kaggle, not locally at all.

**Consequence for workflow:**
- All CUDA compilation, kernel testing, and benchmarking happens in a Colab or
  Kaggle notebook session against a T4 (see `notebooks/colab_bootstrap.ipynb`).
- Code is authored and reviewed locally, pushed to GitHub, then pulled and
  built inside the notebook. There is no local `make build`/`make test` loop.
- `tests/` uses `pytest.importorskip("soinfer")` plus a CUDA-availability skip
  so the suite is inert (not red) on the local machine and only meaningful
  when actually run on the GPU session.
- Iteration speed is bounded by notebook round-trips (push -> pull -> rebuild).
  Batch multiple kernel changes per Colab session rather than round-tripping
  per line changed.

## M0 decisions

- **Binding style:** plain `pybind11` (`PYBIND11_MODULE`) rather than
  `TORCH_LIBRARY`, for the M0 smoke-test op. Simpler for a single free
  function; revisit for `torch.compile`-graph-capturable ops if that becomes
  necessary once soinfer ops are called from inside `nn.Module` forward paths.
- **Build isolation:** `pip install -e . --no-build-isolation`. Build-isolated
  installs would let pip resolve its own torch in a throwaway venv, which then
  mismatches the CUDA build of torch already present in the target environment
  (Colab ships a specific torch+cu12x pairing). `--no-build-isolation` links
  against whatever torch is already importable.
- **Arch flags:** `SOINFER_CUDA_ARCHS` env var (default `"75"`) drives
  `-gencode` flags in `setup.py`, so a Colab T4 build and any future
  Ampere+ build use the same setup.py without editing it.

## M5 deviation: task 5 ("RMSNorm + quantize fusion") not implemented

**Status:** active, as of 2026-09-16.

PROJECT_SPEC.md M5 task 5 asks to "emit quantized activations directly from
the norm." Section 2's explicit scope, though, is weight-only quantization:
"Activation quantization below INT8 (W4A16 and W8A16 only; activations stay
FP16)" is out of scope, and both this project's real quant formats (W4A16,
W8A16) keep activations in FP16 -- there is no activation-quantized format
in scope for an RMSNorm-fusion to emit into. Implementing this task as
literally stated would mean inventing an activation quantization path the
rest of the spec explicitly excludes.

**Consequence:** M5's other four tasks (fused SwiGLU MLP, fused QKV
projection, KV cache append, decode attention) are implemented; task 5 is
skipped rather than built against a scope contradiction. If a future
milestone reintroduces activation quantization (e.g. a W8A8 experiment as
a stretch/study), revisit this and fuse the quantize step into `rmsnorm.cu`
at that point -- the RMSNorm kernel itself needs no change to support it,
since it already computes in fp32 before the final per-element write.
