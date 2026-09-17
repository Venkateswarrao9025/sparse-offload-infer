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

__all__ = ["ops", "quant", "offload"]
