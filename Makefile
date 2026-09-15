.PHONY: build test bench bench-pcie bench-baseline bench-m1 bench-m2 profile clean

# --no-build-isolation: link against the torch already installed in this
# environment (e.g. Colab's preinstalled torch+cu121) instead of pip building
# an isolated env that may resolve a mismatched torch/CUDA pair.
build:
	pip install -e . --no-build-isolation -v

test:
	pytest tests/ -v

bench: bench-pcie bench-baseline bench-m1 bench-m2

bench-pcie:
	python bench/bench_pcie.py

bench-baseline:
	python bench/bench_baseline.py

bench-m1:
	python bench/bench_m1.py

bench-m2:
	python bench/bench_m2.py

profile:
	nsys profile -o reports/profile python bench/bench_pcie.py

clean:
	rm -rf build/ *.egg-info python/soinfer.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
