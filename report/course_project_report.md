# CS5487 Course Project Report

## Title

Default Project: Handwritten Digit Classification with PCA, One-vs-All Classifiers, and Extra Classifiers

## Authors

- Sun Baozheng
- Zhang Yuxuan

## Contribution Statement

Fill in the contribution split required by the course guidelines.

## 1. Introduction

Describe the handwritten digit classification problem, the official digits4000
protocol, and why the project studies both discriminative models and
preprocessing choices. Keep the motivation aligned with the original proposal:
improving over the 1-NN baseline and understanding accuracy-versus-robustness
trade-offs.

## 2. Methodology

### 2.1 Dataset and Protocol

- digits4000 contains 4000 grayscale digit images, 10 classes, 784 features.
- Two official trials are provided via fixed train/test index sets.
- Model selection uses 5-fold cross-validation inside the training split only.

### 2.2 Preprocessing Pipelines

- Raw pixels
- Min-max scaling to [0, 1]
- Z-score standardization
- PCA with 50, 100, and 150 components

Explain why the preprocessing is implemented inside a scikit-learn pipeline:
the same design avoids duplicated code and prevents information leakage because
scaling and PCA are fitted only on training folds.

### 2.3 In-Class Classifiers

- 1-NN with Euclidean distance
- One-vs-all logistic regression
- One-vs-all linear SVM
- One-vs-all RBF-kernel SVM

For each classifier, state the decision rule, key hyperparameters, and expected
strengths or weaknesses on vectorized digit images.

### 2.4 Extra Classifiers Beyond Class Coverage

- Random Forest
- Multi-Layer Perceptron (MLP)

This section should explicitly answer the teacher feedback. Explain the core
idea of each extra classifier, why it is not part of the main in-class method
set used in the proposal, and why it is still relevant to the same feature
representation.

## 3. Experimental Setup

### 3.1 Hyperparameter Search

Document the exact search grids used in the implementation.

### 3.2 Evaluation Metrics

- accuracy for each official trial
- mean and standard deviation across trials
- confusion matrix
- per-class precision, recall, and F1
- private challenge accuracy for each trained trial model

### 3.3 Implementation Details

State the software stack:

- Python in `.venv`
- numpy, pandas, scikit-learn, matplotlib, seaborn

Mention that the same saved trial pipeline is reused for the challenge set with
no retraining and no parameter changes.

## 4. Experimental Results

### 4.1 Official Test Results

Insert the main table from `artifacts/results/summary_by_model.csv` and discuss:

- whether proposal models beat the 1-NN baseline
- whether PCA helps or hurts each classifier
- how the extra classifiers compare with in-class models

### 4.2 Per-Trial Results

Insert or summarize `artifacts/results/final_selected_models.csv`.

### 4.3 Confusion Analysis

Use the saved confusion matrix figures and per-class tables to discuss common
confusions such as similar-looking digits.

### 4.4 Hyperparameter Sensitivity

Use the CV leaderboard to comment on how `C`, `gamma`, PCA dimension, tree
depth, and MLP regularization affect performance.

## 5. Discussion

Discuss:

- why some preprocessing pipelines help linear models more than others
- whether the extra classifiers justify the added complexity
- why MLP or Random Forest may underperform if that happens
- trade-offs among accuracy, stability, runtime, and interpretability

## 6. Private Challenge Evaluation

This section is for the written report or private notes only. Do not copy these
numbers into presentation or poster material.

- report challenge accuracy for each trained trial model
- compare with the reference 1-NN challenge average of 0.683
- explain whether the ranking on the official test set transfers to challenge digits

## 7. Conclusion

Summarize the best-performing classifier, the effect of preprocessing, and the
main conclusion from adding Random Forest and MLP to the original proposal.

## Appendix

### A. Email Summary to Instructor

Use `artifacts/results/email_summary.csv` as the compact summary table.

### B. Reproducibility Notes

Include the exact command used to run the experiments and the directory layout of
the generated artifacts.
