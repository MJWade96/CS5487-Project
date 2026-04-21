# CS5487 Course Project Proposal

1) Problem Introduction This project studies multi-class handwritten digit classification on the default course dataset, a 10-class subset of MNIST. We aim to measure how much discriminative classifiers can improve over a simple nearest-neighbor baseline under the provided writer-disjoint train/test splits, and to understand whether preprocessing improves robustness. 

# 2) Component Proposal

Goal. Classify each 28 $\times$ 28 grayscale image into one of ten digits (0–9), and identify which classifier/preprocessing pipeline performs best under the official protocol. 

Features and data. We will use the provided 784-dimensional pixel features directly from digits4000_digits_vec.txt, with labels from digits4000_digits_labels.txt. We will use the official split indices from digits4000_trainset.txt and digits4000_testset.txt. Preprocessing pipelines will include raw pixels, min-max scaling to [0,1], z-score normalization, and PCA-reduced features (e.g., 50/100/150 dimensions). 

Algorithms. We will compare: 

1. 1-NN (Euclidean) baseline; 

2. one-vs-all logistic regression; 

3. one-vs-all linear SVM; 

4. one-vs-all RBF-kernel SVM. 

Model selection. Hyperparameters (e.g., $C$ , $\gamma$ , PCA dimension) will be selected by cross-validation on the training split only. Test labels are reserved strictly for final evaluation. 

3) Evaluation Plan We will evaluate all methods on both official trials (each trial: 2000 training and 2000 test samples). The primary metric is classification accuracy on the held-out test set for each trial. We will report trial-1 accuracy, trial-2 accuracy, and mean $\pm$ standard deviation across trials. In addition, we will analyze confusion matrices and per-class recall/F1 to inspect common confusion patterns among visually similar digits. We will also examine sensitivity to regularization strength, kernel width, and PCA dimension. 

# 4) Definition of Good Outcome

• reproducible and technically correct implementations with proper train/validation/test separation; 

• consistent improvement over the given 1-NN mean baseline (0.9160); 

• target performance around ∼0.95 mean accuracy if feasible, or a clear explanation when this is not achieved; 

• insightful analysis of accuracy–complexity–robustness trade-offs across preprocessing/model choices; 

• representative success/failure case discussion. 

5) Expected Deliverables Reproducible code, a complete report with experimental analysis, and concise poster-ready figures/tables. 