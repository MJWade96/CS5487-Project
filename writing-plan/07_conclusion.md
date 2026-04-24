# Section 7 — Conclusion

Under the fixed CS5487 protocol, the final answer to "which classifier does better" is clear: OvA RBF SVM is the best overall model. In `summary_by_model.csv`, it achieves the highest mean official-test accuracy (0.9473), ahead of the course baseline 1-NN (0.9160) by 0.0313 and ahead of the second-best Random Forest (0.9268). This meets the proposal objective of improving over the baseline under writer-disjoint official evaluation, while preserving stability across both official trials.

The key scientific takeaway is that preprocessing effectiveness is model-dependent rather than universal. Evidence in `final_selected_models.csv` and `preprocessing_tradeoff_summary.csv` shows a consistent pattern: min-max scaling benefits optimization-sensitive logistic regression and MLP, PCA benefits linear SVM by improving linear separability, and raw pixels remain strongest for 1-NN, RBF SVM, and Random Forest. Therefore, representation choice should be treated as part of model-family design, not as a one-size-fits-all preprocessing default.

Parameter and error analyses further explain why nonlinear methods lead. The selected RBF SVM settings are stable across trials (same best-kernel regime in `final_selected_models.csv`), and the main residual errors are concentrated in structured, visually similar pairs such as 4/9, 5/3, 5/6, and 2/7 (from confusion and case outputs summarized in Section 4). This indicates that remaining mistakes are driven by local stroke ambiguity rather than random failure, and it also explains why linear margins remain limited even when PCA improves conditioning.

This study still has limits. The analysis is based on fixed vectorized pixels and two official trials, so uncertainty under broader handwriting shifts is only partially observed. Future work should therefore focus on four directions: data augmentation for stroke and geometric variability, convolutional architectures that learn local shape hierarchies directly from images, probability calibration for more reliable confidence estimates, and stronger robustness checks under additional writer-disjoint or shifted-domain protocols. These extensions are natural next steps while keeping the same no-leakage selection discipline used in this project.

## Quality Checklist
- [x] No new data introduced.
- [x] Directly maps back to proposal success criteria.
