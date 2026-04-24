# Section 3 — Experimental Setup

All experiments follow the same reproducible selection protocol. For each official trial, we run five-fold stratified cross-validation on the trial training split only, with accuracy as the model-selection objective. After selecting the best pipeline by CV score, we refit that pipeline on the full trial training set and evaluate once on the untouched official test split. Challenge evaluation is strictly post-selection: we apply the already selected official-trial pipeline to challenge data without any retuning.

To make comparisons auditable, we keep one consolidated search design across model families. Preprocessing candidates are drawn from the shared pipeline pool (raw, minmax, zscore, zscore+pca_50, zscore+pca_100, zscore+pca_150), and each model uses a fixed hyperparameter grid.

| Model | Preprocessing candidates | Hyperparameter grid |
| --- | --- | --- |
| 1-NN | raw, minmax, zscore, zscore+pca_50, zscore+pca_100, zscore+pca_150 | n_neighbors fixed at 1 |
| Logistic Regression OvA | raw, minmax, zscore, zscore+pca_50, zscore+pca_100, zscore+pca_150 | C in {1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 1e4}; penalty=l2 |
| Linear SVM OvA | raw, minmax, zscore, zscore+pca_50, zscore+pca_100, zscore+pca_150 | C in {1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 1e4}; class_weight in {None, balanced} |
| RBF SVM OvA | raw, minmax, zscore, zscore+pca_50, zscore+pca_100, zscore+pca_150 | C in {0.1, 1, 10, 100}; gamma in {scale, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2} |
| Random Forest | raw, minmax, zscore, zscore+pca_50, zscore+pca_100, zscore+pca_150 | n_estimators in {300, 600}; max_depth in {None, 20, 40}; max_features in {sqrt, 0.5}; min_samples_leaf in {1, 2} |
| MLP | raw, minmax, zscore, zscore+pca_50, zscore+pca_100, zscore+pca_150 | hidden_layer_sizes in {(256), (256,128)}; alpha in {1e-4, 5e-4, 1e-3}; learning_rate_init in {5e-4, 1e-3, 5e-3} |

Accuracy is the primary metric because course deliverables center on final classification performance under the official split. Macro-F1 and macro-recall are reported as auxiliary diagnostics to check whether gains are class-balanced instead of concentrated in easy classes. Confusion matrices, per-class tables, and representative case examples are exported to support qualitative error interpretation beyond aggregate scores.

The runtime pipeline is split into light and heavy batches for practical execution, then merged into canonical artifacts. Light batch includes 1-NN, logistic regression, and linear SVM. Heavy batch includes RBF SVM, Random Forest, and MLP. The merged outputs in `cv_results_detailed.csv`, `cv_leaderboard.csv`, and `model_tradeoff_summary.csv` are treated as the single source of truth for selection scores, final test performance, and runtime trade-offs.

## Evidence Hooks
- cv_results_detailed.csv
- cv_leaderboard.csv
- model_tradeoff_summary.csv

## Quality Checklist
- [x] No leakage ambiguity.
- [x] Reader can reproduce without external assumptions.
- [x] Terms "official test" and "challenge" clearly separated.
