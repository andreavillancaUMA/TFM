# Benchmark of Explainability Techniques for Neural Networks in Banking Fraud Detection

Master's thesis (TFM) repository — Master in Big Data, Artificial Intelligence and Data Engineering, Universidad de Málaga.

This repository trains a TabNet classifier on the [Bank Account Fraud (BAF)](https://github.com/feedzai/bank-account-fraud) dataset, benchmarks five explainability methods (SHAP KernelExplainer, SHAP GradientExplainer, LIME, Integrated Gradients, and TabNet's native attention) against each other using fidelity, robustness, stability and calibration metrics, and explores loss-level alternatives to SMOTE for handling class imbalance as future work.

## Repository Structure

### Notebooks

| File | Description |
|---|---|
| `1_TFM_EDA_Training_Evaluation.ipynb` | Exploratory data analysis, temporal-consistent preprocessing pipeline, class-imbalance handling (SMOTE), TabNet training, and predictive-performance evaluation (ROC, confusion matrix, calibration). Exports the fitted preprocessor, trained model weights, and processed data arrays to `baf_xai_export/`. |
| `2_TFM_XAI_Benchmark.ipynb` | The core explainability benchmark: SHAP (Kernel and Gradient explainers), LIME, Integrated Gradients, and TabNet native attention, compared via one-hot-aware fidelity, raw/scale-normalized robustness, stability, bootstrap confidence intervals, grouped permutation importance, and post-hoc probability calibration (Platt scaling, isotonic regression, ECE). Documents two methodological corrections (an invalid SHAP background distribution, and a one-hot-aware perturbation protocol) that were necessary to obtain trustworthy results. Exports its results to `final_outputs/`. |
| `3_TFM_FutureWork_ImbalancedLosses.ipynb` | Exploratory future-work notebook, outside the main thesis pipeline. Trains TabNet directly on the original, un-resampled data using three imbalance-aware loss functions (Focal Loss, Influence-Balanced Loss, and a boundary-vicinity-weighted loss inspired by BVSLoss) as an alternative to SMOTE, following Kanász et al. (2024). Compares the resulting models against the thesis' SMOTE + cross-entropy baseline on recall, precision, F1, AUC-ROC and geometric mean (GM) of sensitivity/specificity. |

### Document

| File | Description |
|---|---|
| `4_TFM_Thesis_Memory.docx` | The thesis write-up. Documents the full methodology, the two methodological corrections found during the benchmark, quantitative results (fidelity, robustness, stability, calibration), a local case study, proxy-risk/fairness discussion, and limitations and future work. |

### Data and Artifacts

| Path | Description |
|---|---|
| `baf_xai_export/` | Artifacts exported by `1_TFM_EDA_Training_Evaluation.ipynb`: the fitted preprocessing pipeline, trained TabNet weights, and processed train/validation/test arrays (`processed_arrays.npz`, including both the original and SMOTE-resampled training data). |
| `final_outputs/` | Artifacts exported by `2_TFM_XAI_Benchmark.ipynb`: raw attribution matrices per explanation method, the corrected SHAP background and Integrated Gradients baseline, the fidelity/robustness/stability/calibration results, and the figures and tables consumed by the thesis document. |
| `Base.csv` | Synthetic bank account fraud tabular dataset, used as input for `1_TFM_EDA_Training_Evaluation.ipynb`. |

### Other

| File | Description |
|---|---|
| `.gitignore` | Excludes environment files, caches, and large/regeneratable artifacts from version control. |

## Reproducing the Results

1. Run `1_TFM_EDA_Training_Evaluation.ipynb` first. It generates the preprocessing pipeline, the trained TabNet model, and the processed data arrays that every other notebook depends on.
2. Run `2_TFM_XAI_Benchmark.ipynb` to reproduce the explainability benchmark and calibration results reported in the thesis.
3. (Optional, future work) Run `3_TFM_FutureWork_ImbalancedLosses.ipynb` to reproduce the imbalance-aware loss function comparison. This notebook trains three additional TabNet models and is not required to reproduce the thesis' main results.

All random seeds are fixed at 42. A `QUICK_MODE` flag in the first two notebooks allows a fast, reduced-fidelity debug run before committing to the full, thesis-grade configuration.

## Key Findings

- **SHAP KernelExplainer** is the most faithful post-hoc explanation method for this model.
- **Integrated Gradients** is the best-balanced, most scalable alternative.
- **LIME** is the most robust and stable method but the least faithful.
- **TabNet's native attention** should be treated as an architectural diagnostic rather than a faithful local explanation of individual predictions.
- The model relies predominantly on categorical, profile-like variables (device operating system, housing status, employment status, address tenure), a finding with direct relevance to fairness governance and GDPR Article 22 compliance.

## Author

Andrea Claudia Villanca Rosales — andrea.villanca@gmail.com
Advisors: Javier del Ser, José Manuel Nieto (Enseñanzas Propias, Universidad de Málaga)
