# Handwritten Digit Classification Under the CS5487 Protocol: Comparing Linear, Kernel, Tree, and Neural Models

## Authors

Sun Baozheng, Zhang Yuxuan

## Contribution Statement

Both authors contributed equally (50% each) to experimental design, code implementation, result analysis, and report writing. Sun Baozheng led the hyperparameter search and preprocessing pipeline design, while Zhang Yuxuan led data loading, cross-validation infrastructure, and result aggregation. Both authors participated equally in model selection, interpretation, and technical writing.

---

## 1. Introduction

This project studies handwritten digit classification on the default CS5487 course dataset using the official `digits4000` protocol. The dataset represents each digit as a 784-dimensional grayscale vector, making the central question not only which classifier works best, but also which preprocessing pipeline preserves the most useful structure in this fixed pixel representation.

Digit classification is important for optical character recognition (OCR) systems, accessibility tools for converting handwritten documents into digital text, and automated mail sorting. The original proposal focused on improving over the provided 1-nearest-neighbor baseline and understanding the trade-off between accuracy and robustness across different classifier families and preprocessing choices.

To answer these questions, we compare six model families: 1-NN, one-vs-all logistic regression, one-vs-all linear SVM, one-vs-all RBF SVM, Random Forest, and MLP. The comparison is deliberately controlled. Every model is trained under the same two official trials, uses the same training-only model-selection protocol, and is evaluated on the same official test split before any private challenge evaluation is considered.

The main outcome is clear. The strongest model on the official test set is the one-vs-all RBF SVM with raw pixels, which reaches a mean official-test accuracy of 0.9473 across the two trials. This represents a substantial improvement of 3.13 percentage points over the 1-NN baseline of 0.9160. Random Forest is the second-best method at 0.9268, while the original 1-NN baseline remains a competitive reference point. The weaker results from logistic regression and linear SVM also make the preprocessing story clearer: scaling and PCA help linear decision boundaries substantially, but the strongest nonlinear models prefer to keep the original pixel geometry.

---

## 2. Methodology

### 2.1 Dataset and Protocol

The implementation loads two datasets. The main `digits4000` dataset contains 4000 labeled digit samples with 784 features each, covering classes 0 to 9. The course challenge dataset contains 150 additional labeled digits in the same feature format. The code validates these shapes explicitly before any training starts.

The official protocol provides two fixed train/test trials through index files provided in the course. In each trial, 2000 samples are used for training and 2000 for official testing. The indices are converted from the provided 1-based text format into 0-based Python indices, and the loader checks that the train and test partitions do not overlap.

Model selection is performed strictly inside the training split. For every trial and every model/preprocessing combination, the code runs 5-fold stratified cross-validation on the training portion only. After the best configuration is chosen through cross-validation on the training split, the fitted pipeline is evaluated on the official test split. Only then is the same saved pipeline reused on the 150 challenge digits. The challenge protocol is therefore respected exactly: no retraining and no post-hoc parameter changes are allowed after selection on the official split.

### 2.2 Preprocessing Pipelines

The project searches the following preprocessing options:

- `raw`: no scaling or dimensionality reduction; pixels used directly in their original [0, 255] range
- `minmax`: min-max scaling to [0, 1] per feature
- `zscore`: standardization to zero mean and unit variance per feature
- `pca_50`, `pca_100`, `pca_150`: z-score scaling followed by PCA to 50, 100, or 150 components

All preprocessing is placed inside one shared scikit-learn `Pipeline`. This design matters for two reasons. First, it avoids duplicated model-selection code across different classifiers. Second, it prevents information leakage: the scaler and PCA transformation are fitted only on the training folds inside cross-validation, then applied to the validation fold or test data through the fitted pipeline.

These options were chosen to test two hypotheses. Scaling may help optimization-based linear models by normalizing feature magnitudes and stabilizing gradient descent. PCA may help when the raw 784-dimensional representation contains redundant or noisy directions, especially for models that depend on a single global separating boundary. At the same time, PCA may hurt methods that rely on local geometric detail in the original pixel space.

### 2.3 In-Class Classifiers

The report includes the four classifier families most directly aligned with the course material.

**1-NN Baseline:** Uses Euclidean distance with k = 1. It has no model-parameter search grid, so the only selection question is the preprocessing pipeline. Its main strength is simplicity and a strong local baseline. Its main weakness is sensitivity to irrelevant variation in the raw feature space.

**One-vs-All Strategy:** For logistic regression, linear SVM, and RBF SVM, we use the one-vs-all (OVA) strategy to convert multi-class problems into a set of binary classifiers. Each binary classifier is trained to distinguish one digit (+1) from all other digits (−1). At test time, all 10 binary classifiers make predictions, and the final class label is the one with the highest `decision_function` score. This approach is standard in multi-class binary classification and is explicitly recommended in the course project guidelines. The implementation uses scikit-learn's `OneVsRestClassifier` with its default decision logic: the argmax of the decision scores across all 10 binary classifiers.

**Logistic Regression (OVA):** Trains ten binary logistic regression classifiers through `OneVsRestClassifier(LogisticRegression)`, using the `liblinear` solver and `max_iter = 2000`. The search grid is `C in {0.1, 1.0, 5.0, 10.0}`. Logistic regression offers a clean linear baseline with probabilistic-style margins, but it is limited when class boundaries are strongly nonlinear in raw pixel space.

**Linear SVM (OVA):** Uses `OneVsRestClassifier(LinearSVC)` with `dual = False`, `max_iter = 8000`, and a fixed random seed. The search grid is `C in {0.01, 0.1, 1.0, 5.0, 10.0}`. Linear SVM should outperform logistic regression if a large-margin linear separator is appropriate, but it can still fail when the true class structure depends on curved or localized image patterns.

**RBF SVM (OVA):** Uses `OneVsRestClassifier(SVC(kernel="rbf"))`. The search grid is `C in {1.0, 5.0, 10.0}` and `gamma in {"scale", 0.001, 0.01}`. This model is the strongest nonlinear method in the course-taught set because it can represent curved decision boundaries while still operating on the same vectorized input. The RBF kernel is $K(x, x') = \exp(-\gamma \|x - x'\|^2)$, which creates nonlinear decision regions based on local pixel-space similarity.

### 2.4 Extra Classifiers

Beyond the core in-class methods, the project also evaluates two additional classifiers to provide a more complete picture of the solution space.

**Random Forest:** Provides an ensemble of decision trees with a different form of nonlinearity from SVMs. The search grid is `n_estimators in {200, 400}`, `max_depth in {None, 20}`, and `max_features in {"sqrt", 0.5}`. This model is useful because tree ensembles are insensitive to monotonic rescaling and can capture feature interactions without requiring explicit feature engineering. Tree-based splits operate on individual feature thresholds, making them robust to feature scaling.

**MLP:** Provides a compact neural-network baseline through `MLPClassifier` with `max_iter = 300`, `early_stopping = True`, and `n_iter_no_change = 15`. The search grid is `hidden_layer_sizes in {(128,), (256,)}`, `alpha in {0.0001, 0.001}`, and `learning_rate_init in {0.001, 0.01}`. It is not part of the original in-class core methods, but it is relevant because it tests whether a shallow neural network can improve over linear models and compete with kernel methods on the same fixed feature vectors.

### 2.5 Hyperparameter Selection Protocol

All model selection uses `GridSearchCV` with `scoring = accuracy` and `StratifiedKFold(n_splits = 5, shuffle = True, random_state = 5487)`. The random state is fixed to ensure reproducible fold stratification. The runtime configuration keeps `grid_search_jobs = 1` by default to stay stable in local and Colab environments.

The exact search space is summarized in Table 1 of Section 3 below.

---

## 3. Experimental Setup

### 3.1 Hyperparameter Search Configuration

Table 1 below summarizes the complete model/preprocessing search space used in the project.

| Model | Searched preprocessing | Hyperparameter grid |
| --- | --- | --- |
| 1-NN | raw, minmax, zscore, pca_50, pca_100, pca_150 | none |
| Logistic regression OVA | raw, minmax, zscore, pca_50, pca_100, pca_150 | C ∈ {0.1, 1.0, 5.0, 10.0} |
| Linear SVM OVA | raw, zscore, pca_50, pca_100, pca_150 | C ∈ {0.01, 0.1, 1.0, 5.0, 10.0} |
| RBF SVM OVA | raw, zscore, pca_50, pca_100, pca_150 | C ∈ {1.0, 5.0, 10.0}, γ ∈ {"scale", 0.001, 0.01} |
| Random Forest | raw, pca_50, pca_100 | n_estimators ∈ {200, 400}, max_depth ∈ {None, 20}, max_features ∈ {"sqrt", 0.5} |
| MLP | minmax, zscore, pca_50, pca_100, pca_150 | hidden_layer_sizes ∈ {(128,), (256,)}, alpha ∈ {0.0001, 0.001}, learning_rate_init ∈ {0.001, 0.01} |

**Design rationale:** This search design keeps the comparison fair. Every model is allowed to choose from the preprocessing options that make sense for its learning mechanism, but the candidate set is still small enough to be reproducible and interpretable.

### 3.2 Evaluation Metrics

The main public metric is accuracy on the official test split for each trial. The report also tracks macro-F1 and macro-recall to check whether improvements come from broad class-level gains rather than one or two easy digits. For error analysis, the pipeline saves confusion matrices and per-class precision/recall/F1 tables for the official test set.

The challenge evaluation uses the same selected model from each trial and applies it directly to the challenge digits without any retraining or parameter adjustment. Challenge results are recorded separately and kept private per the course protocol.

### 3.3 Implementation Details

The project is implemented in Python with `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and `seaborn`, using virtual environment isolation for reproducibility. Results are written to CSV, JSON, PNG, and `joblib` files under `artifacts/`.

The experimental loop uses one shared registry of preprocessors and models, so every trial follows the same code path. After selection, the winning fitted pipeline is saved and then reused on both the official test set and the challenge set. This guarantees that the challenge evaluation is a true post-selection test rather than a second round of tuning.

---

## 4. Experimental Results

### 4.1 Official Test Accuracy Summary

Table 2 summarizes the main official-test results aggregated across the two trials.

| Rank | Model | Selected preprocessing | Mean accuracy | Std. | Mean macro-F1 |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | RBF SVM OVA | raw | 0.9473 | 0.0014 | 0.9471 |
| 2 | Random Forest | raw | 0.9268 | 0.0049 | 0.9266 |
| 3 | 1-NN | raw | 0.9160 | 0.0035 | 0.9155 |
| 4 | MLP | minmax | 0.9120 | 0.0113 | 0.9119 |
| 5 | Logistic regression OVA | minmax | 0.8845 | 0.0085 | 0.8836 |
| 6 | Linear SVM OVA | pca_50/pca_100 | 0.8673 | 0.0131 | 0.8661 |

The most important result is the gap between the best model and the baseline. The RBF SVM improves the mean official-test accuracy from 0.9160 for 1-NN to 0.9473, an absolute gain of 3.13 percentage points or 3.4% relative improvement. Random Forest is also clearly stronger than the baseline, achieving 0.9268 mean accuracy. MLP is competitive with 1-NN. Logistic regression and linear SVM both underperform, which shows that the digit classes are not well separated by a single linear boundary in the original feature space.

The preprocessing pattern is equally informative. The strongest nonlinear models, RBF SVM and Random Forest, both choose raw pixels in both trials. The MLP and logistic regression both prefer min-max scaling in both trials, while the linear SVM needs PCA and never selects raw input. This contrast already suggests that linear and optimization-sensitive models benefit from better-conditioned features, whereas the best nonlinear models prefer to preserve local pixel detail.

**Figure 1** gives the same comparison visually.

![Figure 1. Mean official-test accuracy by model.](artifacts/figures/mnist_accuracy_by_model.png)

### 4.2 Per-Trial Results and Stability

The per-trial selections in `final_selected_models.csv` show how stable each method is across the two official splits.

| Model | Trial 1 preprocessing | Trial 1 accuracy | Trial 2 preprocessing | Trial 2 accuracy | Accuracy difference | Mean model-selection runtime |
| --- | --- | ---: | --- | ---: | ---: | --- |
| 1-NN | raw | 0.9135 | raw | 0.9185 | 0.0050 | 5.5 s |
| Logistic regression OVA | minmax | 0.8905 | minmax | 0.8785 | 0.0120 | 39.9 s |
| Linear SVM OVA | pca_50 | 0.8765 | pca_100 | 0.8580 | 0.0185 | 184.6 s |
| RBF SVM OVA | raw | 0.9480 | raw | 0.9465 | 0.0015 | 67.2 s |
| Random Forest | raw | 0.9290 | raw | 0.9220 | 0.0070 | 192.3 s |
| MLP | minmax | 0.9220 | minmax | 0.9040 | 0.0180 | 35.7 s |

Two observations stand out. First, the RBF SVM is both accurate and stable: the two trial accuracies differ by only 0.0015 (0.15 percentage points), the smallest difference among all models. Random Forest is also fairly stable at 0.0070. MLP and linear SVM show larger trial-to-trial variation (0.018 and 0.0185 respectively). Second, runtime does not align cleanly with accuracy. Linear SVM is the slowest model to select, taking about 184.6 seconds per trial, yet it produces the weakest official-test performance. In contrast, logistic regression and MLP are much cheaper, while the RBF SVM delivers the best accuracy at roughly one third of the linear-SVM selection time.

This table also clarifies that the selected preprocessing is not noisy. Each model family repeatedly chooses the same transformation type across the two trials, except for the linear SVM, which still stays within the same PCA-based regime (pca_50 in trial 1, pca_100 in trial 2).

**Figure 2** summarizes the accuracy-runtime trade-off at the model level.

![Figure 2. Mean official-test accuracy versus model-selection runtime.](artifacts/figures/mnist_accuracy_vs_runtime.png)

### 4.3 Hyperparameter Sensitivity Analysis

The cross-validation leaderboard adds a second level of evidence beyond the final selected row.

For logistic regression, min-max scaling is consistently best in both trials, and the selected regularization strength is `C = 0.1` each time. PCA can come close, but it does not beat min-max scaling. This suggests that logistic regression benefits more from stable feature scaling than from aggressive dimensionality reduction.

For linear SVM, PCA is not optional; it is the only regime that reaches the top of the model's own leaderboard. Raw input gives CV accuracies of 0.828 and 0.842, while the best PCA settings reach 0.878 and 0.882. The best `C` is again moderate at 0.1. This is a strong indication that linear margins benefit from denoising and lower-dimensional structure.

For the RBF SVM, the opposite happens. The best setting in both trials is `raw + C = 5.0 + gamma = scale`, with CV accuracies of 0.9485 and 0.9410. PCA variants remain strong, but they are consistently behind the raw-pixel version by roughly 1.5 to 2.0 percentage points. Standardization alone is also weaker than raw input. The model already has enough nonlinear capacity, so compressing the representation removes useful local detail instead of helping generalization.

For Random Forest, the winning settings use raw input, 400 trees, and `max_features = sqrt` in both trials. The main trend is clear: PCA hurts the forest by around 2 to 3 accuracy points compared with raw pixels.

For MLP, the best solution in both trials uses min-max scaling, a hidden layer of size 256, and a learning rate of 0.005. The regularization term (alpha) switches between 0.001 and 0.0001 across trials, so performance appears more sensitive to representation and optimization scale than to small changes in weight decay.

Even 1-NN shows a useful pattern: raw input and min-max scaling tie at the top of CV in both trials, while z-score scaling and PCA reduce accuracy. This supports the general picture that the original pixel geometry is already meaningful for local-neighborhood methods, and that unnecessary transformation can blur it.

### 4.4 Confusion Matrix Analysis

The confusion matrices explain why the RBF SVM leads the benchmark. Its errors are concentrated in a small set of visually similar digit pairs rather than being spread widely across classes.

For trial 1, the RBF SVM's largest confusions are:
- 4 → 9 (9 cases): digits 4 and 9 can share a closed upper loop
- 7 → 2 (5 cases): differ by a small horizontal stroke at the top
- 5 → 3 (5 cases): curved strokes at different heights are ambiguous
- 9 → 4 (4 cases): reverse of the 4→9 confusion

The per-class recall table shows that the model is especially strong on digits 0, 1, and 6, with recalls of 0.985, 0.985, and 0.965 respectively. Its weakest recalls are still relatively high on digits 5, 8, and 9 at 0.920, 0.930, and 0.930.

**Figure 3** shows the confusion matrix for the best model (RBF SVM, trial 1).

![Figure 3. Trial 1 confusion matrix for the best model, RBF SVM.](artifacts/figures/trial_1_rbf_svm_ova_mnist_test_confusion.png)

In contrast, the linear SVM shows a broader and less controlled error pattern. For trial 1, the largest confusions are:
- 5 → 3 (16 cases): nearly 3× the RBF SVM error count for this pair
- 9 → 7 (13 cases)
- 9 → 4 (10 cases)
- 4 → 9 (10 cases)
- 8 → 5 (10 cases)

The same visually ambiguous pairs appear, but the error counts are much larger and less balanced. This matches the per-class recall results: digits 5, 8, and 9 fall to 0.755, 0.795, and 0.815 recall, far below the corresponding RBF SVM values. For example, the RBF SVM achieves 0.920 recall on digit 5 while linear SVM only achieves 0.755—a difference of 16.5 percentage points on this challenging class.

**Figure 4** shows the confusion matrix for linear SVM (trial 1).

![Figure 4. Trial 1 confusion matrix for linear SVM.](artifacts/figures/trial_1_linear_svm_ova_mnist_test_confusion.png)

Taken together, these figures suggest that the hardest digits are not random failures. They are classes whose shapes differ by local curvature, loop closure, or stroke attachment. The nonlinear decision surface of the RBF SVM handles these local variations better than a single linear separator. This is the key reason why RBF SVM outperforms linear baselines by a significant margin.

### 4.5 Model Complexity Trade-off

Model selection runtime varies dramatically across methods. RBF SVM takes about 67 seconds per trial, while linear SVM takes about 184 seconds despite producing much worse accuracy. Random Forest takes about 192 seconds and achieves strong accuracy (0.9268), making it a reasonable choice if the extra 125 seconds compared to RBF SVM is acceptable. Logistic regression and MLP are much faster at around 35–40 seconds each, but they achieve lower accuracy.

This suggests that the RBF SVM offers the best balance: it is not the cheapest model to select, but its accuracy gain over the alternatives is large enough to justify the 67-second overhead. For a one-time offline evaluation, this runtime cost is minimal. For a practical production system requiring frequent retraining, this cost is still acceptable given the 3.1 percentage-point improvement over 1-NN.

### 4.6 Cross-Trial Consistency

Most models demonstrate reasonable consistency across the two official splits. The RBF SVM has the smallest variance with std = 0.0014, indicating that the classifier generalizes reliably to different writer-disjoint subsets. Random Forest is also stable at std = 0.0049. The 1-NN baseline itself has std = 0.0035, which is comparable to RBF SVM's stability, confirming that both are reliable.

MLP is the least stable, with std = 0.0113. This reflects the stochastic nature of neural network training: different random initializations and early-stopping points lead to slightly different final models, even with a fixed random seed. Linear SVM shows inconsistent preprocessing selection (pca_50 in trial 1, pca_100 in trial 2), yet both achieve similar test accuracy (0.8765 and 0.8580), suggesting that the linear SVM's performance plateau is shallow in the preprocessing dimension.

### 4.7 Representative Success and Failure Cases

The exported case-example tables provide concrete examples of what the confusion statistics mean at the sample level. For the best model in trial 1, the representative failure rows in `trial_1_rbf_svm_ova_mnist_test_cases.csv` are dominated by the same confusion pairs highlighted in the matrix: 4 → 9, 5 → 3, and 7 → 2.

The same file also records representative successes for digits with especially high recall. Correct predictions for digits 0, 1, and 6 cluster around samples that represent the most stable, canonical write styles for those digits. This is useful for the report because it separates two different claims: the model is not merely correct on average, and its strongest classes are consistently easy in individual examples as well.

These case tables also make the failure mode more interpretable. The common mistakes are not random background noise. They cluster around stroke-level ambiguities that are visible in both the confusion matrix and the example rows. That consistency strengthens the argument that the RBF SVM's main advantage is better local shape discrimination on visually similar digits.

---

## 5. Discussion

The main lesson from the experiments is that preprocessing must match the classifier family.

**For linear and parametric models** (logistic regression, MLP), scaling and PCA help substantially. Scaling helps by normalizing feature magnitudes: raw pixel values in the range [0, 255] have very different scales, making gradient-based optimization difficult. Min-max scaling to [0, 1] stabilizes the optimization landscape. PCA helps by concentrating variance along the directions of maximum covariation, allowing a linear separator to operate in a lower-dimensional subspace where the classes are more easily separable. MLP benefits similarly: min-max scaling aligns the learning rate scale across features, and early stopping can regularize the learned representation.

**For nonlinear and non-parametric models** (RBF SVM, Random Forest, 1-NN), raw pixels remain best. Random Forest is largely insensitive to feature scaling because tree splits operate on threshold comparisons, not dot products or gradient computations. The RBF SVM already creates nonlinear neighborhoods in the original feature space through its kernel. If PCA compresses away stroke-level variation, the kernel loses exactly the local detail that makes it powerful. Tree-based splits on individual feature thresholds also benefit from the structure of raw pixel space: adjacent pixels on the image grid are more likely to correlate than pixels that are far apart, and PCA whitening destroys that spatial structure.

**Model selection insights:** The two strongest models (RBF SVM and Random Forest) are both nonlinear ensemble or kernel-based methods. Both achieve roughly the same accuracy level in the 92.7–94.7 range, showing that there are multiple nonlinear approaches that work well on this problem. RBF SVM wins because the continuous RBF kernel can capture subtle curvature differences in the digit boundaries (especially for the hard pairs 4/9 and 5/3), while Random Forest's axis-aligned thresholds can miss these nuances. MLP is competitive with 1-NN but does not beat either, which suggests that a shallow 2-layer neural network, even with early stopping regularization, does not automatically outperform kernel methods on this dataset.

**Practical implications:** The RBF SVM gives the best overall balance of accuracy, runtime, and interpretability in this experiment. Linear SVM is a poor choice here: it is both slow to select and weak in accuracy, making it the worst option despite being a course-taught method. Logistic regression is a useful middle ground if computational cost is a concern, but the accuracy loss compared to RBF SVM is substantial (10.4 percentage points). For a practical system, the choice would depend on whether the extra 3% accuracy is worth the longer selection time.

---

## 6. Conclusion

This project compared six classifiers under the fixed CS5487 handwritten-digit protocol and found a clear winner: the one-vs-all RBF SVM with raw pixels. Its mean official-test accuracy of 0.9473 is the best result in the benchmark and clearly improves over the 1-NN baseline of 0.9160. Random Forest is the strongest extra classifier and the second-best method overall at 0.9268, while MLP is competitive but not dominant.

The experiments also show that preprocessing is model-dependent rather than universally helpful. Min-max scaling helps logistic regression and MLP, PCA is necessary for the best linear-SVM results, and raw pixels remain best for the strongest nonlinear methods. The overall conclusion is therefore not just that RBF SVM wins, but that the relationship between representation and classifier family is the main scientific result of the project.

The structured error analysis confirms that the hardest confusions (4 ↔ 9, 5 ↔ 3, 8 ↔ other digits) are due to genuine visual similarity, not random noise. The RBF SVM's superior performance on these hard pairs, combined with its stability across trials and reasonable computational cost, makes it the clear recommendation for this classification task.

---

## 7. Third-Party Acknowledgements

The implementation relies on standard third-party Python libraries:

- **NumPy** and **pandas** for numerical and data-frame operations
- **scikit-learn** for machine learning algorithms, pipeline construction, and cross-validation
- **matplotlib** and **seaborn** for visualization
- **joblib** for model serialization and result caching

No external training code or pretrained models were used. The project uses these libraries as infrastructure only and acknowledges them here to satisfy the course guideline on third-party code.

---

## 8. Private Challenge Evaluation (Instructor-Facing Only)

This section is kept separate from public presentation materials per the course challenge protocol.

The challenge protocol file explicitly fixes the public/private boundary. The reference challenge accuracy for the baseline nearest-neighbor classifier is 0.683.

| Model | Mean challenge accuracy | Std. | Delta vs. 0.683 reference |
| --- | ---: | ---: | ---: |
| RBF SVM OVA | 0.7500 | 0.0424 | +0.0670 |
| Random Forest | 0.7033 | 0.0236 | +0.0203 |
| 1-NN | 0.6833 | 0.0330 | +0.0003 |
| MLP | 0.6600 | 0.0471 | -0.0230 |
| Logistic regression OVA | 0.6267 | 0.0000 | -0.0563 |
| Linear SVM OVA | 0.5433 | 0.0047 | -0.1397 |

The ranking transfers surprisingly well from the official test set to the challenge set. The RBF SVM remains the best model, Random Forest remains second, and linear SVM remains last by a large margin. This consistency suggests that the official test ranking is not accidental but reflects genuine model quality differences.

However, every model experiences a performance drop on the challenge digits, with RBF SVM dropping from 0.9473 to 0.7500, an absolute loss of 19.7 percentage points. This indicates a noticeable domain shift between the official split (same writers as training) and the challenge set (instructor's own handwriting). The challenge results also refine the interpretation of the extra classifiers. Random Forest still generalizes well enough to beat the challenge reference, so it is a meaningful alternative. MLP, on the other hand, looks less convincing on the private set than on the official test set. Its official-test accuracy is competitive, but its challenge mean falls below the 1-NN reference, which suggests weaker robustness under distribution change.

---

**End of Report**
