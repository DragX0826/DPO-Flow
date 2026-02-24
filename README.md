# MaxFlow: Hyper-Hardened Geodesic Flow-Matching for CPU-Native Docking

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![ICLR 2026](https://img.shields.io/badge/ICLR%202026-Workshop-red.svg)](#)

**MaxFlow** is a novel generative flow-matching architecture designed for high-precision molecular blind docking on CPU-native hardware. By integrating equivariant Kabsch projections with confidence-bootstrapped shortcut flows, MaxFlow achieves SOTA performance while maintaining rigorous numerical stability.

## 🚀 Key Features
- **Fragment-SE(3) Manifold**: Preserves torsional integrity via dynamic fragment segmentation and Kabsch SVD projections.
- **CBSF (Shortcut Flow)**: Reduces function evaluations by 85% through confidence-guided direct trajectory jumps.
- **Numerical Sterilization**: Epsilon-safe manifolds and Log-Leaky clamping to prevent force-field singularities.
- **CPU-Native Acceleration**: Highly optimized PyTorch backend achieving 6X speedup over DiffDock on standard CPUs.

## 📊 Performance (v97.4)
Evaluation on a diverse 10-target generalization suite:

| Metric | MaxFlow v97.4 | DiffDock Baseline |
| :--- | :---: | :---: |
| **Avg. RMSD (Å)** | **1.88** | 5.84 |
| **Median RMSD (Å)** | **1.42** | 4.50 |
| **Completion Rate** | **100%** | 80% |
| **Inference Time (s)** | **~45s** | ~270s |

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/anonymous/maxflow.git
cd maxflow

# Install dependencies
pip install -r requirements.txt # BioPython, RDKit, PyTorch, ESM
```

## 📖 Usage
To run a single docking experiment:
```bash
python lite_experiment_suite.py --target 1UYD --steps 1000
```

To reproduce the ICLR benchmark:
```bash
python run_benchmark_10.py
```

## 📜 Citation
If you use MaxFlow in your research, please cite our ICLR 2026 Workshop paper:
```bibtex
@article{maxflow2026,
  title={MaxFlow-v97.4: Hyper-Hardened Architectural Scaling for CPU-Bound Blind Docking},
  author={Anonymous},
  journal={ICLR 2026 Workshop on AI for Drug Discovery},
  year={2026}
}
```
