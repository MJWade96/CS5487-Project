# Section 5 — Discussion

## 5. Discussion

The main transferable finding is that preprocessing is not universally beneficial; it is classifier-family dependent. Results in `final_selected_models.csv` and `preprocessing_tradeoff_summary.csv` show a consistent split: raw pixels are repeatedly selected for 1-NN, RBF SVM, and Random Forest, while min-max scaling is selected for logistic regression and MLP, and PCA is selected for linear SVM. This pattern suggests that the value of preprocessing depends on the model's inductive bias rather than on a single global "best" representation. In practical terms, preprocessing should be treated as part of model design, not as a fixed default step.

The RBF SVM advantage comes from boundary shape, not only from better average accuracy. According to `summary_by_model.csv`, RBF SVM reaches 0.9473 mean official-test accuracy, while linear SVM reaches 0.8673. The confusion analysis in Section 4 shows that difficult pairs (such as 4/9, 5/3, and 2/7) differ by local stroke attachment and loop closure patterns that are hard to separate with one-vs-all linear hyperplanes. RBF kernels can model these local nonlinear transitions directly in pixel space, so they preserve decision detail where linear separators compress too aggressively, even when PCA improves conditioning.

Beyond SVM variants, Random Forest is the strongest non-kernel alternative, while MLP is competitive but less stable. `summary_by_model.csv` reports Random Forest at 0.9268 mean official-test accuracy and 0.7033 challenge accuracy, both clearly above 1-NN (0.9160 and 0.6833). MLP reaches 0.9120 on the official test, but with larger cross-trial variation (std 0.0113) and lower challenge robustness (0.6633). Combined with runtime evidence in `model_tradeoff_summary.csv`, this indicates two useful interpretations: tree ensembles provide robust second-best performance with minimal preprocessing sensitivity, whereas shallow fully connected MLPs can match classical baselines on in-domain data but are more sensitive to trial shift and optimization dynamics.

Under resource constraints, the recommended deployment order is driven by accuracy-runtime-stability tradeoffs rather than accuracy alone. If maximum accuracy is the priority and moderate training cost is acceptable, RBF SVM is the best default (mean 0.9473; runtime about 67 s in `model_tradeoff_summary.csv`). If compute budget is very tight, 1-NN provides a fast and reproducible baseline (about 4 s selection runtime) but sacrifices around 3.1 percentage points of official-test accuracy. If a backup model is needed for operational robustness, Random Forest is the preferred fallback because it retains high accuracy with stable raw-feature behavior across both trials. A key limitation is that all conclusions are based on only two official writer-disjoint trials, so variance estimates are coarse and may understate uncertainty across broader handwriting domains. A direct actionable implication is to add additional writer-disjoint resampling or nested repeated CV in future work before final model lock-in, while keeping challenge data strictly post-selection to avoid protocol leakage.

## Evidence Sources
- final_selected_models.csv
- preprocessing_tradeoff_summary.csv
- summary_by_model.csv
- model_tradeoff_summary.csv

## Quality Checklist
- [x] Goes beyond restating Section 4 tables.
- [x] Includes at least one limitation and one actionable implication.
