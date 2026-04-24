# CS5487 Project - Digit Classification

**Authors:** Sun Baozheng (59433383); Zhang Yuxuan (59948249)

## Abstract

This project studies ten-class handwritten digit classification on the default CS5487 digits4000 dataset. We compare six supervised classifiers under the two official writer-disjoint train/test trials: 1-NN, one-vs-all (OvA) logistic regression, one-vs-all linear SVM, one-vs-all RBF SVM, Random Forest, and MLP (multi-layer perceptron). All model selection is done with five-fold cross-validation inside the training split only. The best official-test result comes from the raw-pixel RBF SVM, with mean accuracy 94.73% across the two trials, which is 3.12 percentage points above the 1-NN baseline. Preprocessing effects are clearly model-dependent: scaling helps optimization-based models, PCA helps the linear SVM, and the strongest nonlinear models perform best on raw pixels where local stroke detail is preserved. Error analysis also shows that the hardest classes are shared across models rather than unique to RBF SVM: digits 5, 8, 2, 3, and 9 are consistently harder, while digits 1, 0, and 6 are consistently easier.

## 1. Introduction

The task is to map each 28 x 28 grayscale digit image to one of ten classes. Even though this is a standard supervised problem, it is still a useful model-selection test because the input is a fixed 784-dimensional pixel vector rather than engineered features. A strong classifier must separate visually similar digits while remaining robust to writer-specific variation such as stroke thickness, slant, incomplete loops, and small shifts.

This dataset works well for the course project because it brings several supervised-learning ideas into one controlled setup. The labels are simple, but the feature space is high-dimensional and the class boundaries are not clearly linear. Methods perform well only when their inductive bias matches digit geometry: some classes differ by global shape, while others hinge on small local cues like loop closure or short stroke attachment.

The original proposal set three goals: beat the course 1-NN baseline, test whether preprocessing improves generalization, and identify failure patterns on visually similar classes. The final experiment keeps those goals and adds Random Forest and MLP as nonlinear baselines. This yields a controlled comparison across local-neighbor, linear, kernel, tree-ensemble, and neural-network models under the same official protocol.

Accordingly, this report treats accuracy as the first layer of evidence, not the final answer. The more important questions are why one model wins, whether the result is stable across both trials, and whether the same failure patterns appear in confusion matrices, per-class recall, and representative examples. This avoids over-interpreting a single aggregate number.

## 2. Methodology

The data are the provided digits4000 files. The feature matrix contains 4000 samples, each with 784 grayscale pixel values in [0, 255], and the labels cover digits 0 through 9 with 400 samples per class. The official index files define two trials, each with 2000 training samples and 2000 test samples. The same writer does not appear in both train and test within a trial, so the test accuracy measures writer-level generalization rather than memorization of a writer's style.

The writer-disjoint split is stricter than a random image split. If the same writer appeared in both train and test, a model could partially rely on writer-specific style cues (stroke style, pen pressure, centering habits). Under the official split, it must learn class structure that transfers across writers.

All preprocessing is implemented in scikit-learn Pipelines, so scalers and PCA are fit only on training folds during cross-validation. Candidate transformations include raw pixels, min-max scaling to [0, 1], z-score standardization, and z-score plus PCA (50, 100, or 150 dimensions). Some model families use a subset when a transformation is not meaningful or not included in that model's grid, but selection is always based on training-fold-only cross-validation.

The classifier set includes the originally proposed methods plus two extra baselines. 1-NN is a strong local baseline with minimal training cost, but it is sensitive to irrelevant pixel-space variation. Logistic regression and linear SVM are interpretable one-vs-all linear baselines, but both are limited to global linear boundaries. RBF SVM keeps the one-vs-all setup and adds a nonlinear kernel, enabling curved decision boundaries. Random Forest provides a tree-ensemble alternative that is largely insensitive to monotonic scaling. MLP tests whether a compact neural network can outperform classical methods on the same fixed pixel vectors.

Each family has a different trade-off. 1-NN is transparent and has no real training phase, but it stores all samples and can be distorted by irrelevant pixel directions. Logistic regression is fast and stable, and linear SVM benefits from a margin objective, yet both are constrained by linear boundaries. RBF SVM, Random Forest, and MLP are more expressive, but they require heavier tuning and higher selection cost.

## 3. Experimental Setup

For each trial, each model/preprocessing candidate is selected using GridSearchCV with five stratified folds and accuracy as the validation metric. The selected pipeline is then refit on the full training split and evaluated once on the official test split. Challenge digits are evaluated only after this selection step by reusing the same fitted pipeline, without retraining or parameter changes.

This separation is the main technical control in the study. The training split is the only source used for model selection, including scaler fitting, PCA fitting, and hyperparameter comparison. The official test split is never used to choose C, gamma, PCA dimension, tree count, learning rate, or any other setting.

Table 0 summarizes the hyperparameter grids used in model selection.

| Model | Hyperparameter grid |
| --- | --- |
| Logistic Regression OvA | C in {1e^{-4}, 1e^{-3}, 1e^{-2}, 0.1, 1, 10, 100, 1000, 1e^{4}}; penalty = L2 |
| Linear SVM OvA | C in {1e^{-4}, 1e^{-3}, 1e^{-2}, 0.1, 1, 10, 100, 1000, 1e^{4}}; class_weight in {None, balanced} |
| RBF SVM OvA | C in {0.1, 1, 10, 100}; gamma in {scale, 1e^{-4}, 3e^{-4}, 1e^{-3}, 3e^{-3}, 1e^{-2}} |
| Random Forest | n_estimators in {300, 600}; max_depth in {None, 20, 40}; max_features in {sqrt, 0.5}; min_samples_leaf in {1, 2} |
| MLP | hidden_layer_sizes in {(256,), (256, 128)}; alpha in {1e^{-4}, 5e^{-4}, 1e^{-3}}; learning_rate_init in {5e^{-4}, 1e^{-3}, 5e^{-3}} |

The main metric is official-test accuracy, reported separately for trial 1 and trial 2 and as mean plus standard deviation across trials. Macro-F1 and per-class recall are used to check whether a model improves broadly across classes. Confusion matrices and representative case examples are used for failure analysis.

Because the full grid is time-consuming, the run was split into light and heavy batches on Tencent Cloud and then merged into the canonical artifacts folder. The light batch contains 1-NN, logistic regression, and linear SVM; the heavy batch contains RBF SVM, Random Forest, and MLP. The merge step only consolidates artifacts and does not alter selected hyperparameters.

The latest artifact export also includes a post-hoc analysis layer generated after batch merging. It stores per-candidate confusion matrices, per-class metrics, per-sample predictions, enriched case examples, and cross-model sample comparisons. These files are not used for selection; they are used only to interpret already selected pipelines.

## 4. Experimental Results

Table 1 reports the official-test results after combining the light and heavy batches.

| Model | Selected preprocessing | Trial 1 acc. | Trial 2 acc. | Mean acc. | Std. | Mean macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| RBF SVM OvA | raw / raw | 94.80% | 94.65% | 94.73% | 0.11% | 0.9471 |
| Random Forest | raw / raw | 92.95% | 92.40% | 92.67% | 0.39% | 0.9266 |
| 1-NN | raw / raw | 91.35% | 91.85% | 91.60% | 0.35% | 0.9155 |
| MLP | minmax / minmax | 92.00% | 90.40% | 91.20% | 1.13% | 0.9120 |
| Logistic Regression OvA | minmax / minmax | 89.05% | 87.85% | 88.45% | 0.85% | 0.8836 |
| Linear SVM OvA | PCA 50 / PCA 50 | 87.65% | 85.80% | 86.72% | 1.31% | 0.8661 |

RBF SVM is the top model, with mean official-test accuracy 94.73%. It improves over the 1-NN baseline (91.60%) by 3.12 percentage points. Random Forest ranks second and remains clearly above 1-NN. MLP is close to the nearest-neighbor baseline but less stable across trials. Logistic regression and linear SVM trail behind, which is consistent with the limited expressiveness of a single linear boundary on the hardest shapes.

This ranking is meaningful because every model follows the same trial protocol and training-only selection rule. The RBF SVM vs 1-NN gap reflects the value of nonlinear neighborhood structure beyond raw Euclidean nearest-neighbor distance.

![Figure 1. Mean official-test accuracy by model.](../artifacts/figures/mnist_accuracy_by_model.png)

Table 2 adds the per-trial selections and average model-selection runtime.

| Model | Trial 1 prep. | Trial 1 acc. | Trial 2 prep. | Trial 2 acc. | Mean runtime (s) |
| --- | --- | --- | --- | --- | --- |
| RBF SVM OvA | raw | 94.80% | raw | 94.65% | 66.7 |
| Random Forest | raw | 92.95% | raw | 92.40% | 192.0 |
| 1-NN | raw | 91.35% | raw | 91.85% | 4.9 |
| MLP | minmax | 92.00% | minmax | 90.40% | 35.3 |
| Logistic Regression OvA | minmax | 89.05% | minmax | 87.85% | 44.0 |
| Linear SVM OvA | PCA 50 | 87.65% | PCA 50 | 85.80% | 184.6 |

RBF SVM is also stable: trial-level accuracies differ by only 0.15 percentage points. MLP varies more (1.60 points). Linear SVM is one of the weakest trade-offs in this setup: it has the lowest accuracy while taking about 184.6 seconds per trial, versus 66.7 seconds for the stronger RBF SVM. Random Forest is slightly slower (about 192.0 seconds per trial) but still much more accurate than linear SVM.

Runtime changes the practical picture. 1-NN is the cheapest baseline, but RBF SVM delivers much higher accuracy at a still-manageable search cost. In this grid, linear SVM is the least attractive option because it is both slow and weak.

![Figure 2. Accuracy-runtime trade-off on the official test set.](../artifacts_newest/figures/mnist_accuracy_vs_runtime.png)

### 4.1 Preprocessing Effects

Table 3 shows representative cross-validation averages for preprocessing choices.

| Model | Preprocessing | Mean best CV acc. | Selected trials |
| --- | --- | --- | --- |
| 1-NN | raw | 92.10% | 2/2 |
| 1-NN | minmax | 92.10% | 0/2 |
| Logistic Regression OvA | minmax | 89.03% | 2/2 |
| Logistic Regression OvA | raw | 86.75% | 0/2 |
| Linear SVM OvA | PCA 50 | 87.83% | 1/2 |
| Linear SVM OvA | raw | 84.15% | 0/2 |
| RBF SVM OvA | raw | 94.47% | 2/2 |
| RBF SVM OvA | PCA 100 | 92.53% | 0/2 |
| Random Forest | raw | 92.27% | 2/2 |
| Random Forest | PCA 100 | 90.15% | 0/2 |
| MLP | minmax | 91.90% | 2/2 |
| MLP | zscore | 90.48% | 0/2 |

The preprocessing pattern is consistent across trials. Raw pixels are selected by 1-NN, RBF SVM, and Random Forest. Min-max scaling is selected by logistic regression and MLP. PCA 50 is selected for the linear SVM setup in this report. This supports the central representation result: scaling helps optimization-sensitive models, PCA helps linear margins through denoising, and raw pixels keep local stroke detail that benefits strong nonlinear models.

The direction of the preprocessing effect depends on classifier family. Logistic regression rises from 86.75% mean best CV accuracy on raw pixels to 89.03% with min-max scaling. Linear SVM rises from 84.15% on raw pixels to 87.83% with PCA 50. By contrast, RBF SVM and Random Forest both lose accuracy with PCA, suggesting that dimensionality reduction removes useful local stroke cues.

Table 4 adds per-class recall deltas to show that preprocessing changes are not uniformly beneficial.

| Model | Trial | Preprocessing comparison | Digit | Recall change | Interpretation |
| --- | --- | --- | --- | --- | --- |
| Linear SVM OvA | trial 1 | PCA 50 vs raw | 4 | +12.5 pp | PCA helps the linear boundary recover digit 4 |
| Linear SVM OvA | trial 2 | PCA 50 vs raw | 8 | +10.5 pp | PCA improves a hard high-variance digit |
| Logistic Regression OvA | trial 1 | minmax vs raw | 8 | +9.0 pp | scaling helps optimization on ambiguous 8s |
| 1-NN | trial 1 | zscore vs raw | 2 | -10.5 pp | z-score can distort nearest-neighbor distances |
| RBF SVM OvA | trial 2 | zscore vs raw | 7 | -10.0 pp | raw pixels preserve local stroke geometry |
| Random Forest | trial 2 | PCA 100 vs raw | 5 | -8.5 pp | PCA can remove local shape detail needed by trees |

The per-class deltas show that preprocessing changes are not uniformly beneficial. PCA gives the linear SVM its largest gains on difficult digits such as 4 and 8, where the raw linear boundary underfits. In contrast, z-score scaling hurts 1-NN on digit 2 by about 10 percentage points, and it also hurts the RBF SVM on digit 7 in trial 2. This supports the interpretation that distance-based and kernel methods benefit from preserving the original local pixel geometry, while linear optimization benefits from denoised or rescaled features.

### 4.2 Confusion, Per-Class, and Cross-Model Case Analysis

The best model's remaining errors are concentrated in visually plausible pairs. In trial 1, the largest RBF SVM confusions are 4->9 (9 cases), 7->2 (5), 5->3 (5), 9->4 (4), and 2->7 (4). In trial 2, the largest are 5->3 (5), 9->4 (5), 5->6 (5), 8->9 (4), and 4->9 (4). These pairs are not random: they correspond to loop closure, stroke attachment, and local curvature differences.

![Figure 3. Trial 1 confusion matrix for the selected RBF SVM.](../artifacts/figures/trial_1_rbf_svm_ova_mnist_test_confusion.png)

Per-class recall reinforces the same interpretation. For RBF SVM, digits 0, 1, and 6 are among the easiest classes in both trials, with recalls near or above 0.97 in most cases. The weakest recall in trial 2 is digit 5 at 0.895, followed by digit 2 at 0.915 and digits 8 and 9 at 0.930. These are exactly the classes involved in the largest confusion pairs.

Table 5 summarizes mean selected-model accuracy by digit and confirms that the hardest classes are not specific to one model.

| Digit | Mean selected-model accuracy |
| --- | --- |
| 5 | 83.42% |
| 8 | 85.46% |
| 2 | 88.04% |
| 3 | 89.33% |
| 9 | 89.58% |
| 4 | 91.04% |
| 7 | 92.79% |
| 6 | 94.29% |
| 0 | 96.75% |
| 1 | 98.25% |

Averaging over the selected models confirms that the hardest classes are not specific to the RBF SVM. Digit 5 has the lowest mean selected-model accuracy at 83.42%, followed by 8, 2, 3, and 9. Digits 1, 0, and 6 are consistently easier. This cross-model pattern matches the pairwise confusion analysis: the hard classes are those where small loop, curvature, or stroke-attachment changes can alter the label.

No digit has average recall below 0.90 under the selected RBF SVM. Digit 5 is the hardest because it can look like 3 when the lower curve is rounded and like 6 when the lower loop closes. The linear SVM makes the same types of mistakes in larger numbers, which indicates that the issue is model flexibility rather than random label noise.

Table 6 shows stable high-confusion pairs across selected models.

| Model | Selected preprocessing | Most stable/highest confusion pairs |
| --- | --- | --- |
| RBF SVM OvA | raw | 4->9 mean 6.5, 5->3 mean 5.0, 7->2 mean 5.0, 9->4 mean 4.5, 5->6 mean 4.0 |
| Random Forest | raw | 4->9 mean 10.0, 8->1 mean 10.0, 5->3 mean 7.5, 7->2 mean 6.0, 8->3 mean 5.5 |
| 1-NN | raw | 4->9 mean 17.0, 8->1 mean 11.0, 8->3 mean 10.0, 8->5 mean 8.5, 9->7 mean 8.5 |
| MLP | minmax | 5->3 mean 12.0, 4->9 mean 11.5, 2->7 mean 6.0, 5->8 mean 6.0, 8->1 mean 5.5 |
| Logistic Regression OvA | minmax | 5->8 mean 14.0, 5->3 mean 13.5, 4->9 mean 11.5, 8->1 mean 9.0, 9->4 mean 7.5 |
| Linear SVM OvA | PCA 50 summary mode | 5->3 mean 10.5, 8->5 mean 10.0, 9->7 mean 10.0, 8->1 mean 9.0, 6->5 mean 8.5 |

The same visually plausible pairs recur across models. The RBF SVM reduces the counts, but the error identities barely change: 4/9, 5/3, 5/6, 7/2, 8/1, and 9/4 keep showing up. So the remaining mistakes look more like genuine shape ambiguity in the 784-pixel representation, not a model-specific quirk.

The representative-case figures make this easier to see at a glance. In trial 1, the difficult samples cluster around 4->9, 5->3, and 7->2; in trial 2, the main groups are 5->6, 5->3, and 9->4. To keep the display clean, each panel uses normalized labels with sample index, true label, and predicted label.

![Figure 4. Trial 1 representative cases built from images_png (success and failure examples).](figures/rbf_trial_1_cases_from_images_png.png)

![Figure 5. Trial 2 representative cases built from images_png (success and failure examples).](figures/rbf_trial_2_cases_from_images_png.png)

These two panels also explain why linear models lag behind. The same confusing shapes appear there too, but with higher counts. In trial 1, linear SVM produces 16 cases of 5->3 and 13 cases of 9->7, while RBF SVM keeps those categories much smaller.

Table 7 separates easy, model-specific, and intrinsically hard samples by counting how many selected models are correct on each sample.

| Number of selected models correct | Trial 1 samples | Trial 2 samples |
| --- | --- | --- |
| 6 | 1607 | 1586 |
| 5 | 142 | 153 |
| 4 | 83 | 61 |
| 3 | 57 | 60 |
| 2 | 34 | 55 |
| 1 | 33 | 44 |
| 0 | 44 | 41 |

In trial 1, all six selected models classify 1607 of 2000 official-test samples correctly, while all six miss 44 samples. In trial 2, the corresponding counts are 1586 and 41. The middle bands are useful diagnostically: samples solved by RBF SVM and Random Forest but missed by linear models show the value of nonlinear boundaries, while samples missed by all selected models are likely ambiguous under the fixed 784-pixel representation.

Table 8 gives the RBF per-class recall values directly as a text table rather than an embedded image.

| Digit | Trial 1 recall | Trial 2 recall | Mean recall |
| --- | --- | --- | --- |
| 0 | 0.985 | 0.975 | 0.980 |
| 1 | 0.985 | 0.995 | 0.990 |
| 2 | 0.950 | 0.915 | 0.932 |
| 3 | 0.935 | 0.935 | 0.935 |
| 4 | 0.935 | 0.945 | 0.940 |
| 5 | 0.920 | 0.895 | 0.908 |
| 6 | 0.965 | 0.980 | 0.972 |
| 7 | 0.945 | 0.965 | 0.955 |
| 8 | 0.930 | 0.930 | 0.930 |
| 9 | 0.930 | 0.930 | 0.930 |

## 5. Private Challenge Evaluation

The challenge results are included here for the instructor-facing written report, but they should not be copied into public presentation or poster material. The challenge set contains 150 handwritten digits and is evaluated only by applying each trial's already selected pipeline.

| Model | Mean challenge acc. | Std. | Delta vs. 0.683 reference |
| --- | --- | --- | --- |
| RBF SVM OvA | 75.67% | 4.24% | +7.37 pp |
| Random Forest | 70.33% | 2.36% | +2.03 pp |
| 1-NN | 68.33% | 3.30% | +0.03 pp |
| MLP | 66.33% | 0.47% | -1.97 pp |
| Logistic Regression OvA | 62.67% | 0.00% | -5.63 pp |
| Linear SVM OvA | 54.33% | 0.47% | -13.97 pp |

The challenge ranking is broadly consistent with the official-test ranking: RBF SVM remains best, Random Forest remains second, and linear SVM remains last. Every method drops relative to the official test split, indicating a handwriting-style domain shift. Even so, RBF SVM is still 7.37 percentage points above the 0.683 reference. MLP, despite being competitive on the official test set, falls below the nearest-neighbor reference on challenge data.

Macro-F1 tells the same story as accuracy: RBF SVM is highest (0.7455), followed by Random Forest (0.6874) and 1-NN (0.6757). This suggests the challenge conclusion is not driven by only one or two classes.

This challenge drop is expected because challenge digits come from a different handwriting source. Official-test accuracy measures generalization to held-out writers within the course dataset, while challenge accuracy adds an extra distribution shift. That is why absolute numbers are lower even though model ordering is similar.

## 6. Conclusion

Under the CS5487 protocol, the one-vs-all RBF SVM on raw pixels is the most reliable overall choice. It gives the best accuracy, stays stable across both writer-disjoint trials, and holds up better than the other candidates on the private challenge set. Random Forest is consistently second. The linear baselines are still useful because they make the preprocessing trade-offs visible, even though they cannot model the hardest nonlinear boundaries as well.

The key takeaway is the interaction between preprocessing and model family. Min-max scaling helps logistic regression and MLP, PCA helps linear SVM, and raw pixels work best for 1-NN, RBF SVM, and Random Forest. Error analysis tells the same story from another angle: most remaining mistakes stay in a small set of shape-confusable pairs, including 4/9, 5/3, 5/6, 7/2, and 8/9. Stronger nonlinear models mostly reduce the frequency of these errors rather than changing which pairs are difficult.

The main limitation is representation. We used fixed 784-dimensional pixel vectors, not richer spatial features. Better normalization (for example deskewing or centering correction) or convolution-based models would likely improve the hardest classes. Within the scope of this course project, however, the current setup cleanly isolates how classical models and preprocessing choices behave under one common protocol.

## Third-Party Acknowledgements

The implementation uses standard Python libraries: NumPy and pandas for data handling, scikit-learn for modeling and cross-validation, matplotlib and seaborn for experiment figures, joblib for saved pipelines, Pillow for case visualizations, and python-docx for report export. No external training code or pretrained models were copied into the experiment pipeline.

## Contributions

Sun Baozheng (59433383) was responsible for the primary code development and report writing. Zhang Yuxuan (59948249) was responsible for code optimization and presentation video production.
