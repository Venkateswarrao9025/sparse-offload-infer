# Sparse-Offload Inference Engine

**Status: work in progress -- M0 (harness and baseline).**

This README is a placeholder. The real one -- roofline plot, accuracy-vs-
speedup Pareto curve, ablation table -- gets written last, from the numbers in
`reports/`, per the project's own rule against unbacked claims. Until then:

- Full spec and milestone plan: [PROJECT_SPEC.md](PROJECT_SPEC.md)
- Design decisions and deviations from spec: [docs/DESIGN.md](docs/DESIGN.md)
- Running learning log: [docs/LEARNING_NOTES.md](docs/LEARNING_NOTES.md)
- Dev workflow: this repo has no local NVIDIA GPU; all CUDA build/test/bench
  happens on a Colab/Kaggle T4 via [notebooks/colab_bootstrap.ipynb](notebooks/colab_bootstrap.ipynb).
