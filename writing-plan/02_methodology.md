# Section 2 — Methodology

This section defines the compared methods and selection protocol, with emphasis on technical validity and fair model comparison under the official CS5487 constraints.

## 2.1 Dataset and Protocol

We use the provided digits4000 benchmark: 4000 grayscale handwritten digits, each represented as a 784-dimensional vector (28 x 28), with 10 balanced classes. The official evaluation protocol provides two fixed trials, each with 2000 training samples and 2000 test samples. The split is writer-disjoint within each trial, so test accuracy measures style generalization to unseen writers rather than writer memorization.

All trial index files are used as provided, and train/test overlap checks are applied when loading data to confirm protocol integrity. Cross-validation is performed only on each trial's training split; official test sets are never used for model selection.

## 2.2 Preprocessing Pipelines

We compare six preprocessing variants: raw pixels, min-max scaling (to [0, 1]), z-score standardization, z-score plus PCA to 50 dimensions, z-score plus PCA to 100 dimensions, and z-score plus PCA to 150 dimensions. Each preprocessing step is implemented inside a unified scikit-learn Pipeline together with the classifier.

Pipeline wrapping is critical for leakage safety: in each cross-validation fold, scaling and PCA are fit only on the fold-specific training partition and then applied to validation data. This avoids accidental use of validation statistics during representation learning.

Before observing results, the working hypothesis is that scaling should benefit optimization-sensitive linear or neural models, while PCA may help linear margins by suppressing noisy dimensions but may also remove local shape details useful to nonlinear models.

## 2.3 In-Class Classifiers

The core in-class methods are 1-NN, OvA logistic regression, OvA linear SVM, and OvA RBF SVM. 1-NN predicts from local neighborhood structure with minimal training assumptions; it is strong as a geometry-preserving baseline but sensitive to irrelevant pixel-space distortions. Logistic regression and linear SVM provide linear decision boundaries with efficient optimization and interpretability, but their expressiveness is limited for strongly nonlinear class overlap. RBF SVM extends linear margins with a kernel-induced nonlinear feature space, typically improving separation for curved class boundaries at higher computational cost.

All OvA classifiers in this project use sklearn OneVsRestClassifier default decision behavior: prediction is the class with maximum decision_function score across the 10 binary classifiers.

## 2.4 Extra Classifiers

Two additional families are included to broaden the comparison scope. Random Forest provides a tree-ensemble baseline that can model nonlinear interactions and is usually less sensitive to monotonic feature scaling. MLP provides a compact feed-forward neural baseline to test whether learned hidden representations improve over classical methods under the same protocol and feature inputs.

## 2.5 Hyperparameter Selection

Hyperparameters are selected independently per trial using GridSearchCV with five-fold StratifiedKFold on the training split and accuracy as the primary selection metric. The searched grids are model-specific but consistent across trials, enabling direct trial-level comparability. Final selected pipelines are retrained on each trial's full training data and then evaluated once on the untouched official test split.

This methodology section focuses on mechanism and protocol design; empirical rankings, preprocessing wins, and error patterns are reported in Section 4.
