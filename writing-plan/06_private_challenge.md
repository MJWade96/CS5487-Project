# Section 6 — Private Challenge Evaluation

> **Private only:** This section is for the written course report and must not be included in any public poster or presentation material.

The challenge set contains 150 private samples and is evaluated strictly after model selection. No challenge outcomes are used for preprocessing choice, hyperparameter tuning, or model-family selection; all selections are fixed from the official trial protocol before challenge inference (challenge_protocol.json).

| Model | Mean challenge accuracy | Std. | Delta vs. 0.683 reference |
| --- | --- | --- | --- |
| RBF SVM OvA | 0.7567 | 0.0424 | +0.0737 |
| Random Forest | 0.7033 | 0.0236 | +0.0203 |
| 1-NN | 0.6833 | 0.0330 | +0.0003 |
| MLP | 0.6633 | 0.0047 | -0.0197 |
| Logistic Regression OvA | 0.6267 | 0.0000 | -0.0563 |
| Linear SVM OvA | 0.5433 | 0.0047 | -0.1397 |

Challenge ranking remains broadly aligned with official-test ranking: RBF SVM is still strongest, Random Forest remains second, and linear SVM remains weakest (summary_by_model.csv). This consistency suggests that the relative ordering of model families is not random, even under domain shift. At the same time, all models drop compared with official-test performance, indicating that the private set introduces handwriting styles that are not fully covered by the two official writer-disjoint splits.

The most practical interpretation is robustness separation: models that preserve nonlinear local-shape discrimination on official tests (especially RBF SVM) also degrade more gracefully on challenge data, while linear separators lose more heavily. This section therefore supports a deployment view where challenge results are used only for post-hoc robustness assessment, not for iterative tuning.

## Evidence Hooks
- summary_by_model.csv
- challenge_protocol.json

## Quality Checklist
- [x] Public/private boundary is explicit at the top of the section.
- [x] No wording that suggests challenge results were used for model tuning.
