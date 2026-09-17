try:
    from . import generate, loader
except ImportError:
    # generate.py/loader.py need soinfer.offload.load_hf_checkpoint
    # (safetensors) and soinfer.ops (the compiled extension) -- neither is
    # available on this GPU-less local machine. See soinfer/__init__.py's
    # ops=None and soinfer/offload/__init__.py's load_hf_checkpoint=None
    # for the same pattern.
    generate = None
    loader = None

__all__ = ["generate", "loader"]
