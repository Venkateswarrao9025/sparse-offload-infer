from . import offload, quant

try:
    from . import ops
except ImportError:
    # soinfer._C (the compiled CUDA extension) isn't built -- expected on a
    # machine with no NVIDIA GPU (see docs/DESIGN.md). quant/ and offload/
    # are pure Python/PyTorch at import time (they only touch CUDA when
    # their classes are actually instantiated) and work fine without it;
    # anything under ops needs the extension and will only fail if
    # actually called.
    ops = None

try:
    from . import runtime
except ImportError:
    # runtime.generate needs both soinfer.ops and soinfer.offload.load_hf_checkpoint
    # (safetensors) -- see their own None-guards for why.
    runtime = None

__all__ = ["ops", "quant", "offload", "runtime"]
