# NGDT
Thermodynamic Natural Gradient Descent (NGD-T): Regulating Natural-Gradient Steps by a Geometric Speed-Cost Bound

## Abstract
We introduce Thermodynamic Natural Gradient Descent (NGD-T), an optimizer that enforces a physical speed-cost constraint by combining Fisher preconditioned updates with a dissipation aware step size regulator. Starting from an Entropic Action, we show that Natural Gradient Flow (NGF) uniquely minimizes instantaneous irreversible dissipation for a fixed loss decrease. NGD T implements this principle in discrete updates by (i) preconditioning gradients with an approximate inverse Fisher, (ii) computing the geometric norm Δ_F=∇L^⊤ F^(-1) ∇L, and (iii) mapping a user specified dissipation budget Q_budget to a step size η_T that saturates the speed–cost bound. We present numerically stable constructions for rank deficient Fisher estimates using eigendecomposition or Tikhonov damping, a hybrid nullspace fallback to preserve progress in truncated modes, and a scalable K FAC integration with eigendecomposition caching. On CIFAR 10 experiments NGD T matches Adam in convergence while substantially reducing the predicted irreversible dissipation. NGD T provides a principled, tunable trade off between learning speed and thermodynamic cost and is compatible with standard large scale Fisher approximations.

## Introduction
This repository includes all the reference codes of NGDT optimizer and experiment programmes.

## 📖 Citation

If you use NGDT in your research, we would appreciate a citation to our [paper](https://doi.org/10.21203/rs.3.rs-8626621/v1):

```
@misc{you2026NGDT,
      title={Thermodynamic Natural Gradient Descent (NGD-T): Regulating Natural-Gradient Steps by a Geometric Speed-Cost Bound}, 
      author={Barco Jie You},
      year={2026},
      archivePrefix={researchsquare},
      primaryClass={cs.AI},
      url={https://doi.org/10.21203/rs.3.rs-8626621/v1}, 
}
```

## 📜 License 
This project is licensed under the [MIT License](./LICENSE).
