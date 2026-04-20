# CS5487 Course Project Presentation Draft

Target length: about 5 minutes total across 6 slides.

## Slide 1 - What Works Best on the Official MNIST Test?

**On-slide bullets**
- Goal: compare six classifiers under one reproducible pipeline.
- All results in this talk use the two official MNIST test trials only.
- Main takeaway: RBF SVM is the strongest and most consistent model.

**Visual asset/reference**
- Use `artifacts/figures/mnist_accuracy_by_model.png` as a small right-side teaser.

**Speaker notes (0:00-0:30, 30s)**
- This project asks a simple question: under the official course protocol, which classifier works best for handwritten digit recognition?
- I implemented one shared pipeline so every model uses the same data loading, preprocessing search, and evaluation logic.
- The headline result is that the one-vs-all RBF SVM is the clear winner on the official MNIST test set.

## Slide 2 - Dataset and Evaluation Protocol

**On-slide bullets**
- Main dataset: 4000 digit images, 784 pixel features, labels 0 to 9.
- Two official trials, each with 2000 training and 2000 test samples.
- Model selection uses 5-fold CV on the training split only.
- Reported metrics are official MNIST test accuracy and error patterns.

**Visual asset/reference**
- Build a simple protocol strip directly on the slide: `4000 samples -> trial 1 / trial 2 -> 2000 train + 2000 test -> 5-fold CV on train -> official test accuracy`.
- Reference: `src/digits_project/data.py` and the provided train/test index files.

**Speaker notes (0:30-1:15, 45s)**
- The key point here is fairness: both official trials come from the provided split files, and the code validates that each trial has exactly 2000 training and 2000 test samples with no overlap.
- For model selection, I never tune on the test split. Hyperparameters are chosen by 5-fold cross-validation on the training portion only, then evaluated once on the official test set.
- That makes the final comparison much cleaner and keeps the talk focused on public, official-test evidence.

## Slide 3 - Models and Preprocessing Search

**On-slide bullets**
- Compared 6 classifiers: 1-NN, logistic OvA, linear SVM OvA, RBF SVM OvA, random forest, and MLP.
- Searched raw pixels, scaling, and PCA variants in one shared pipeline registry.
- Best preprocessing was model-specific, not one-size-fits-all.
- Selected pairings: kNN raw, logistic min-max, linear SVM PCA, RBF raw, random forest raw, MLP min-max.

**Visual asset/reference**
- Build a compact 2-column table from `artifacts/results/final_selected_models.csv` showing each model and its selected preprocessing.
- Reference: `src/digits_project/models.py` for the search space.

**Speaker notes (1:15-2:00, 45s)**
- I kept the search space centralized in one registry so the experiment logic was not duplicated across models.
- The interesting pattern is that the best preprocessing depends on the classifier: raw pixels stayed best for kNN, RBF SVM, and random forest, while min-max scaling helped logistic regression and MLP.
- Linear SVM was the only model that consistently preferred PCA, which already hints that its decision boundary is less expressive than the RBF model.

## Slide 4 - Main Official-Test Results

**On-slide bullets**
- RBF SVM OvA: **94.70%** mean official-test accuracy, best on both trials.
- Next group: random forest **92.55%**, 1-NN **91.60%**, MLP **91.45%**.
- Linear baselines trail: logistic regression **88.45%**, linear SVM **86.73%**.
- RBF SVM beats the strongest simple baseline, 1-NN, by **3.10 points**.

**Visual asset/reference**
- Use `artifacts/figures/mnist_accuracy_by_model.png` as the main figure on this slide.

**Speaker notes (2:00-3:00, 60s)**
- This is the central result of the project. The RBF SVM leads the full comparison with a mean official-test accuracy of 94.7%.
- It is also stable across both official trials: 94.8% on trial 1 and 94.6% on trial 2.
- The next tier is random forest, 1-NN, and MLP, all around the low 92 to low 91 range, while the linear models are clearly lower.
- So the main result is not just that RBF SVM wins, but that a nonlinear boundary gives a meaningful gain over simpler baselines under the same protocol.

## Slide 5 - Where the Nonlinear Model Helps

**On-slide bullets**
- Trial 1 confusion matrix is visibly cleaner for RBF SVM than for linear SVM.
- Linear SVM struggles most on visually similar curved digits, especially 5, 8, and 9.
- Per-class F1 improves strongly with RBF: digit 5 from **0.805** to **0.946**.
- It also lifts digit 8 from **0.820** to **0.930** and digit 9 from **0.834** to **0.923**.

**Visual asset/reference**
- Show `artifacts/figures/trial_1_rbf_svm_ova_mnist_test_confusion.png` next to `artifacts/figures/trial_1_linear_svm_ova_mnist_test_confusion.png`.
- Optional small callout from the per-class CSVs: `artifacts/results/per_class/trial_1_rbf_svm_ova_mnist_test.csv` and `artifacts/results/per_class/trial_1_linear_svm_ova_mnist_test.csv`.

**Speaker notes (3:00-4:15, 75s)**
- This slide explains why the best model wins. In the confusion matrices, the RBF SVM has a noticeably tighter diagonal and fewer dense off-diagonal mistakes.
- The linear SVM has much heavier confusion among curved look-alike digits. One concrete example is digit 5 being predicted as 3 much more often, with 16 such errors for the linear SVM versus 5 for the RBF model.
- The per-class F1 scores tell the same story: digit 5 improves from 0.805 to 0.946, digit 8 from 0.820 to 0.930, and digit 9 from 0.834 to 0.923.
- So the gain is not just a small average boost. The nonlinear model is materially better on the hard classes that create most of the visible confusion.

## Slide 6 - Conclusion

**On-slide bullets**
- Best overall model: one-vs-all RBF SVM on raw pixels.
- Accuracy lead over simpler baselines: +3.10 points vs 1-NN, +6.25 vs logistic, +7.98 vs linear SVM.
- It also stays ahead of the stronger extra baselines: +2.15 vs random forest and +3.25 vs MLP.
- Final takeaway: on this task, nonlinear separation mattered more than heavier preprocessing.

**Visual asset/reference**
- Reuse `artifacts/figures/mnist_accuracy_by_model.png` with the RBF SVM bar highlighted.

**Speaker notes (4:15-5:00, 45s)**
- To conclude, the strongest result is a simple one: the best-performing system is the raw-pixel one-vs-all RBF SVM.
- It beats every simpler baseline, and it still stays ahead of the extra nonlinear baselines like random forest and MLP.
- The practical lesson is that for this digits task, choosing a model with a flexible nonlinear boundary mattered more than adding heavier preprocessing.
- If I had one sentence to leave on the final slide, it would be: the RBF SVM gave the best accuracy and the cleanest error profile under the official protocol.
