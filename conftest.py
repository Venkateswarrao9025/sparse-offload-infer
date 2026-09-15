"""Make `import soinfer` resolve to python/soinfer without an editable pip
install. Colab already has soinfer pip-installed (needed to load the CUDA
extension), so this is a harmless no-op there; locally (no NVIDIA GPU, so
`pip install -e .` can't build the CUDA extension -- see docs/DESIGN.md)
this is what lets pure-Python modules like soinfer.quant be imported and
tested at all.
"""
import sys
from pathlib import Path

_python_dir = Path(__file__).resolve().parent / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))
