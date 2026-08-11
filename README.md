# Calorimeter Energy Regression - ML Architecture Benchmark

Machine learning benchmark for electron energy calibration in a sampling
calorimeter, comparing five learning-machine architectures plus a
hyperparameter-optimized variant, evaluated with a full ML-methodology
workflow (cross-validation, overfitting diagnostics, feature selection, and a
physics-specific energy-resolution metric). Built for a computational
challenge at the [IDPASC](https://www.idpasc.pt/) doctoral programme, using
shower-shape data simulated with the
[Lorenzetti Showers Framework](https://doi.org/10.1016/j.cpc.2023.108671).

## Overview

High Energy Physics experiments commonly use fast, fixed-size sliding-window
algorithms to reconstruct electron energy from calorimeter data. This is
computationally cheap but doesn't capture the full lateral extent of the
electromagnetic shower, so a secondary calibration step is needed. This
notebook explores machine learning as that calibration step: given
one-dimensional longitudinal and lateral shower-shape variables (energy
ratios between calorimeter layers, shower width, hadronic leakage, etc.),
predict the electron's true energy directly.

The notebook works through nine design questions end to end:

1. **Five-plus architectures**: shallow MLP, deep MLP (BatchNorm + Dropout),
   1D CNN, ResNet-style MLP, and Gradient Boosting Regression, plus an
   Optuna-tuned variant of the best-performing model.
2. **Unified evaluation**: a single `evaluate()` function (RMSE, MAE, R^2,
   and a Gaussian fit of the relative residual for bias/resolution) applied
   identically to every model.
3. **Hyperparameter optimization** with Optuna.
4. **Isolated sensitivity analysis**: batch size, network width/depth, and
   optimizer choice, varied one at a time.
5. **Overfitting diagnostics**: early-stopping patience, L2/Dropout
   regularization, and a training-set-size learning curve.
6. **5-fold cross-validation**, reporting mean +/- std per architecture to
   check for statistical fluctuations in the (comparatively small) dataset.
7. **Energy resolution**: quantified per architecture and fit with the
   standard calorimetry stochastic/constant/noise-term formula for the
   best model.
8. **Preprocessing and feature selection**: scaler comparison, a
   log-target check, a correlation matrix, and model-agnostic permutation
   importance.
9. **Auto-generated starter slide deck** summarizing the comparison table
   and key figures.

## Data

The notebook expects a ROOT file `zee.root` produced by the Lorenzetti 
Showers Framework, containing an `events` tree with per-cluster shower-shape
branches: `cluster_e`, `cluster_et`, `cluster_eta`, `cluster_phi`,
`cluster_reta`, `cluster_rphi`, `cluster_weta2`, `cluster_ehad1/2/3`,
`cluster_rhad`, `cluster_rhad1`, `cluster_eratio`, `cluster_e237`, and `cluster_e277`.
See the [Lorenzetti Showers Framework](https://github.com/lorenzetti-hep/lorenzetti)
repository for how to generate this data with the simulator.

## Installation

TensorFlow and PyTorch both install CPU-only versions by default via this
`requirements.txt`; if you have a CUDA-capable GPU, install the appropriate
GPU builds of `tensorflow` and `torch` for a substantial training-time
speedup, since the notebook trains roughly ten neural network
configurations across the sensitivity, regularization, and cross-validation
studies.

## Usage

```bash
jupyter notebook calorimeter_energy_regression.ipynb
```

Place `zee.root` in the same directory before running. The notebook caches
trained models and histories under `trained_models/`, so re-running it
after an interruption loads existing models instead of retraining from
scratch. Delete that folder to force a clean re-run.

## Results

Running the notebook produces:
- A comparison table (`results_df`) ranking all architectures by RMSE, MAE,
  R^2, and resolution.
- Loss curves, true-vs-predicted scatter plots, and per-energy-bin
  resolution curves for every architecture.
- A permutation-importance ranking of the shower-shape features.
- A starter PowerPoint deck (`calorimeter_energy_regression_results.pptx`)
  with the comparison table and key figures embedded.

## References

- Lorenzetti Simulator collaboration (2024). *A fast simulation framework
  for electromagnetic calorimeters.* Computer Physics Communications, 108671.
  https://doi.org/10.1016/j.cpc.2023.108671
- Akiba, T., et al. (2019). *Optuna: A Next-generation Hyperparameter
  Optimization Framework.* KDD 2019.
