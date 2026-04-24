# Section 4 — Experimental Results

## Writing Objective
- Present results from macro to micro: overall ranking -> per-trial stability -> preprocessing effects -> confusion/cases.

---

## Subsection 4.1 Overall Official-Test Results
Table 4.1 reports the official-test ranking by mean accuracy, standard deviation, and macro-F1. The strongest model is RBF SVM OvA with 94.73% mean accuracy and macro-F1 0.9471, followed by Random Forest at 92.67% and 0.9266. The 1-NN baseline reaches 91.60% and 0.9155. The best model therefore improves absolute accuracy by 3.12 percentage points over 1-NN (94.73% - 91.60%). Source: artifacts/results/summary_by_model.csv.

| Model | Mean official-test accuracy | Std | Mean macro-F1 |
|---|---:|---:|---:|
| RBF SVM OvA | 94.73% | 0.11% | 0.9471 |
| Random Forest | 92.67% | 0.39% | 0.9266 |
| 1-NN | 91.60% | 0.35% | 0.9155 |
| MLP | 91.20% | 1.13% | 0.9120 |
| Logistic Regression OvA | 88.45% | 0.85% | 0.8836 |
| Linear SVM OvA | 86.72% | 1.31% | 0.8661 |

Insert figure: artifacts/figures/mnist_accuracy_by_model.png.

---

## Subsection 4.2 Per-Trial Stability
Per-trial selected preprocessing and test accuracy are shown in Table 4.2. RBF SVM keeps raw pixels in both trials and changes only from 94.80% to 94.65%, showing strong cross-trial stability. In contrast, MLP drops from 92.00% to 90.40%, and linear SVM changes from pca_50 to pca_100 with a larger accuracy gap (87.65% to 85.80%). Source: artifacts/results/final_selected_models.csv.

| Model | Trial 1 preprocessing | Trial 1 acc. | Trial 2 preprocessing | Trial 2 acc. | Selection runtime (s) |
|---|---|---:|---|---:|---:|
| RBF SVM OvA | raw | 94.80% | raw | 94.65% | 64.17 / 70.28 |
| Random Forest | raw | 92.95% | raw | 92.40% | 194.42 / 190.21 |
| 1-NN | raw | 91.35% | raw | 91.85% | 5.67 / 2.39 |
| MLP | minmax | 92.00% | minmax | 90.40% | 33.08 / 38.38 |
| Logistic Regression OvA | minmax | 89.05% | minmax | 87.85% | 39.57 / 40.21 |
| Linear SVM OvA | pca_50 | 87.65% | pca_100 | 85.80% | 188.40 / 180.78 |

These gaps support a stability ordering in this study: RBF SVM and Random Forest are stable at top ranks, while MLP and linear SVM show larger trial sensitivity.

---

## Subsection 4.3 Preprocessing Effects
Model families prefer different representations. Raw pixels are selected for 1-NN, RBF SVM, and Random Forest in both trials; min-max scaling is selected for logistic regression and MLP in both trials; linear SVM prefers PCA (pca_50 and pca_100 across the two trials). Source: artifacts/results/final_selected_models.csv.

Cross-validation ranking in cv_leaderboard.csv is consistent with that pattern: for RBF SVM, raw (0.9485 and 0.9410) outperforms PCA variants; for logistic regression, minmax (0.8885 and 0.8920) beats raw; for linear SVM, PCA settings rank above raw in both trials. Source: artifacts/results/cv_leaderboard.csv.

Insert figure: artifacts/figures/mnist_accuracy_vs_runtime.png.

Interpretation: scaling helps optimization-sensitive objectives (logistic regression and MLP), PCA helps linear margins by suppressing noisy directions, while raw pixels preserve local stroke cues exploited by nonlinear boundaries and tree splits.

---

## Subsection 4.4 Hyperparameter Sensitivity
Hyperparameter sensitivity is modest but structured. For logistic regression, both trials select C = 0.1, indicating that moderate regularization is consistently preferred over very weak or very strong regularization. For linear SVM, both trials also select C = 0.1 under OvA, while the winning preprocessing changes across trials (pca_50 vs pca_100), suggesting representation has stronger effect than small C variations. Source: artifacts/results/final_selected_models.csv.

For RBF SVM, both trials select C = 10 and gamma = scale, and this setting is also associated with top official-test performance. This supports the view that nonlinear capacity is beneficial, but only when kernel width and margin penalty are jointly balanced. Source: artifacts/results/final_selected_models.csv and artifacts/results/cv_leaderboard.csv.

For Random Forest and MLP, selected configurations vary in depth/trees and hidden-layer layout across trials, yet both remain in the upper-middle or top ranks, indicating that their generalization is less tied to a single fragile setting and more tied to model-family inductive bias.

---

## Subsection 4.5 Confusion and Per-Class Analysis
Error structure is not random. For RBF SVM trial 1 case examples, top confusion groups include 4->9, 5->3, and 7->2; trial 2 includes 5->6, 5->3, and 9->4. This confirms persistent shape-level ambiguity around loop closure and stroke connection, especially for the 4/9 and 5/3 families. Source: artifacts/results/case_examples/trial_1_rbf_svm_ova_mnist_test_cases.csv and artifacts/results/case_examples/trial_2_rbf_svm_ova_mnist_test_cases.csv.

Compared with linear SVM, RBF SVM maintains higher per-class recall on difficult digits. In trial 1, digit-5 recall is 0.920 for RBF versus 0.755 for linear SVM; in trial 2, digit-8 recall is 0.930 for RBF versus 0.775 for linear SVM. Source: artifacts/results/per_class/trial_1_rbf_svm_ova_mnist_test.csv, artifacts/results/per_class/trial_1_linear_svm_ova_mnist_test.csv, artifacts/results/per_class/trial_2_rbf_svm_ova_mnist_test.csv, artifacts/results/per_class/trial_2_linear_svm_ova_mnist_test.csv.

Technical note for correctness: these OvA confusion outcomes are generated under sklearn OneVsRestClassifier default decision behavior, where prediction is selected by the maximum decision_function score among ten binary classifiers.

---

## Subsection 4.6 Representative Success/Failure Cases
Representative failures in RBF SVM trial 1 include sample 2826 and 2872 (4->9), plus 3045 and 3123 (5->3). In trial 2, representative failures include 1065 and 1112 (5->6), plus 1860 and 1938 (9->4). Source: artifacts/results/case_examples/trial_1_rbf_svm_ova_mnist_test_cases.csv and artifacts/results/case_examples/trial_2_rbf_svm_ova_mnist_test_cases.csv.

These errors are frequent enough to form the top-ranked confusion groups but still limited in count relative to class support (200 per class), which indicates that the model captures most shape variations but fails at boundary cases with near-identical local geometry.

Linear SVM examples show broader high-rank failure groups (for example, 5->3 and 9->7 in trial 1, and 8->1, 8->5, 9->7 in trial 2), consistent with an insufficiently flexible linear separator for curved handwriting manifolds. Source: artifacts/results/case_examples/trial_1_linear_svm_ova_mnist_test_cases.csv and artifacts/results/case_examples/trial_2_linear_svm_ova_mnist_test_cases.csv.

---

## Quality Checklist
- [x] Every claim references a file-backed number.
- [x] No contradiction with selected preprocessing in final_selected_models.csv.
- [x] Confusion analysis emphasizes patterns, not isolated anecdotes.
