"""Session-wide determinism.

test_m8_calibration.py's determinism tests assert bit-exact reproducibility
of top-k-derived channel counts across repeated calls. cuBLAS's default
(non-deterministic) algorithm selection can flip near-tied top-k boundaries
depending on allocator fragmentation left behind by whichever tests happened
to run earlier in the same process -- passed reliably in isolation, was
flaky as part of the full suite. `CUBLAS_WORKSPACE_CONFIG` must be set
before any CUDA/cuBLAS call, so this has to live in conftest.py (loaded
before any test module is imported), not in the test file itself. See
docs/LEARNING_NOTES.md's M8 entry.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

torch.use_deterministic_algorithms(True)
