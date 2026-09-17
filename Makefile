.PHONY: build test bench bench-pcie bench-baseline bench-m1 bench-m2 bench-m4 bench-m6 bench-m6-headline profile profile-m6-overlap clean

# --no-build-isolation: link against the torch already installed in this
# environment (e.g. Colab's preinstalled torch+cu121) instead of pip building
# an isolated env that may resolve a mismatched torch/CUDA pair.
build:
	pip install -e . --no-build-isolation -v

test:
	pytest tests/ -v

bench: bench-pcie bench-baseline bench-m1 bench-m2 bench-m4 bench-m6

bench-pcie:
	python bench/bench_pcie.py

bench-baseline:
	python bench/bench_baseline.py

bench-m1:
	python bench/bench_m1.py

bench-m2:
	python bench/bench_m2.py

bench-m4:
	python bench/bench_m4.py

bench-m6:
	python bench/bench_m6_roofline.py

# Not part of `make bench` -- downloads a 14B model (~28GB) and takes
# several minutes. Run explicitly: `make bench-m6-headline`.
bench-m6-headline:
	python bench/bench_m6_headline_model.py

profile:
	nsys profile -o reports/profile python bench/bench_pcie.py

# M6 acceptance: Nsight Systems overlap evidence. See bench/profile_m6_overlap.py's
# docstring if `nsys` isn't on PATH (common on Colab -- it ships under
# /opt/nvidia/nsight-compute/<version>/host/target-linux-x64/nsys).
profile-m6-overlap:
	nsys profile --trace=cuda --capture-range=cudaProfilerApi -o reports/m6_overlap_profile bench/profile_m6_overlap.py
	nsys stats --report cuda_gpu_trace --format csv --output reports/m6_overlap_profile reports/m6_overlap_profile.nsys-rep
	python bench/analyze_nsys_overlap.py reports/m6_overlap_profile_cuda_gpu_trace.csv

clean:
	rm -rf build/ *.egg-info python/soinfer.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
