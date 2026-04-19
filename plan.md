## Plan: CS5487 Digits Project

在默认 digits4000 手写数字分类项目上，保留 proposal 中的 1-NN、one-vs-all logistic regression、one-vs-all linear SVM、one-vs-all RBF-kernel SVM 与 PCA 预处理比较，再新增 Random Forest 和 MLP 两个课外分类器来回应老师反馈。实现上用 scikit-learn pipeline 严格把缩放和 PCA 放进训练内交叉验证，按官方两次 trial 产出代码、结果表、分析图和英文报告；本轮默认不制作正式 poster 文件，但会产出可直接放进 poster 的图表。

**Steps**
1. Phase 1 - 锁定协议与工程骨架
- 以课程要求和 proposal 为唯一实验协议来源，固定数据、trial、评价指标和“课内/课外分类器”边界。
- 在工作区根目录新增最小结构：`src/`、`artifacts/results/`、`artifacts/figures/`、`report/`、依赖文件和统一运行入口。
- 明确当前无现成代码，避免为兼容历史实现而增加复杂度。
2. Phase 2 - 数据读取与官方 split 复原
- 依据 digits4000_txt/README.txt 的说明处理文本矩阵转置问题，重建特征矩阵 `X` 为 4000 x 784，标签向量 `y` 为 4000。
- 将 train/test 索引从 MATLAB 风格的 1-based 转成 Python 0-based，并明确两次官方 trial：trial 1 用前 2000 训练、后 2000 测试；trial 2 反向交换。
- 加入 shape、索引范围、train/test 是否重叠的检查，先把协议跑通再做模型。
3. Phase 3 - 预处理与模型工厂
- 预处理管线包含 raw、min-max scaling、z-score normalization、PCA 50/100/150 维。
- 保留 proposal 模型：1-NN 基线、OVA logistic regression、OVA linear SVM、OVA RBF-kernel SVM。
- 新增课外模型：Random Forest 和 MLP；其中 MLP 采用 scikit-learn 的 MLPClassifier，而不是 CNN/深度学习框架，以控制实现规模并保持和向量化输入一致。
- 缩放与 PCA 一律放进 sklearn Pipeline 中，避免交叉验证或测试阶段数据泄漏。
4. Phase 4 - 训练、调参与测试执行
- 主执行入口改为 `digits_project_colab.ipynb`，优先在 Colab 上分批运行，而不是继续在本地直接跑 `run_experiments.py`；本地 `.py` 仅保留为回退入口。
- 每个 trial 只在训练集内部做 5-fold stratified cross-validation 选参数，然后用最佳参数在该 trial 全训练集上重训，再到对应测试集评估。
- 参数网格保持小而可辩护：logistic/linear SVM 搜 C，RBF SVM 搜 C 与 gamma，Random Forest 搜树数/深度/特征子采样，MLP 搜隐藏层宽度、alpha 与学习率方案。
- 默认采用 batch 执行：`batch_light` 跑 `knn_1`、`logistic_regression_ova`、`linear_svm_ova`；`batch_heavy` 跑 `rbf_svm_ova`、`random_forest`、`mlp`。若重模型仍超时，再拆成单模型 batch。
- notebook 需要先盘点当前已完成的 trial/model 组合，再只补跑缺失批次；完整批次完成后，再通过合并步骤重建 `artifacts/results/`，避免把半成品和新结果混写在一起。
- 输出每个模型在每个 trial 的预测结果、accuracy、macro/weighted precision/recall/F1、confusion matrix、最佳参数和运行时间。
- 在每个 trial 的最佳模型于对应 MNIST test set 完成评估后，立刻复用同一个已训练好的 pipeline 和参数对 `challenge/cdigits_digits_vec.txt` 做推理；不得重训、不得重新选参、不得单独为 challenge 重新拟合缩放或 PCA。
- challenge 结果单独记录为每个 trial 的 accuracy，并与对应 MNIST test accuracy 成对保存，满足 challenge/README.txt 的邮件汇报要求。

5. Phase 5 - 结果汇总与分析图表
- 生成 trial-1、trial-2、mean±std 的总表，覆盖所有模型与主要预处理组合。
- 生成 confusion matrix 图、参数敏感性图，以及“proposal 模型 vs 新增模型”的对比图。
- 单独分析老师反馈对应的问题：课外分类器是否优于课内方案；若没有显著更好，也要说明原因，例如样本量、过拟合、输入表示与模型偏好不匹配。
6. Phase 6 - 项目报告撰写
- 报告按课程要求写成英文，结构固定为 introduction、methodology、experimental setup、experimental results、discussion/conclusion。
- methodology 中明确区分课内模型与课外模型，并给出 Random Forest 与 MLP 的原理、关键超参数和采用理由。
- experimental results 中保留 proposal 承诺的 confusion matrix、per-class recall/F1、参数敏感性分析，并补上 accuracy-complexity-robustness 的横向讨论。
- discussion 中加入成功/失败样例解释，说明哪些数字对最易混淆，以及 PCA/标准化对各模型的影响。
7. Phase 7 - 交付物收口
- 补一个简短 README，写清依赖安装、Colab notebook 批处理运行方式、结果输出位置和合并流程。
- 整理最终提交物为：源代码、结果图表、英文报告；正式 poster 文件默认不做，但保留 poster-ready 表格与图片。
- 如果后续你要冲 A，再基于现有图表单独扩一版 poster 计划，而不是把 poster 设计混进当前实现。

**Relevant files**
- `f:\课程\Semester B\CS5487 Machine Learning：Principle & Practice\作业\CS5487 Course Project\Code\课程资料\Project Proposal.pdf` — 原 proposal，确定原始方法集、评价指标与“good outcome”。
- `f:\课程\Semester B\CS5487 Machine Learning：Principle & Practice\作业\CS5487 Course Project\Code\课程资料\Guidelines.md` — 官方协议、默认项目要求、允许使用课外分类器的规则。
- `f:\课程\Semester B\CS5487 Machine Learning：Principle & Practice\作业\CS5487 Course Project\Code\课程资料\Lecture 0 - Info 2026 SemB.pdf` — 课程安排证据，用于判断哪些分类器明确属于课堂内容。
- `f:\课程\Semester B\CS5487 Machine Learning：Principle & Practice\作业\CS5487 Course Project\Code\课程资料\PS-8.pdf` — logistic regression、perceptron、linear classifiers 的课堂证据。
- `f:\课程\Semester B\CS5487 Machine Learning：Principle & Practice\作业\CS5487 Course Project\Code\课程资料\PS-9.pdf` — linear SVM 的课堂证据。
- `f:\课程\Semester B\CS5487 Machine Learning：Principle & Practice\作业\CS5487 Course Project\Code\课程资料\PS-10.pdf` — kernel functions、kernel SVM 的课堂证据。
- `f:\课程\Semester B\CS5487 Machine Learning：Principle & Practice\作业\CS5487 Course Project\Code\digits4000_txt\README.txt` — 文本矩阵被转置存储的关键读取说明。
- `f:\课程\Semester B\CS5487 Machine Learning：Principle & Practice\作业\CS5487 Course Project\Code\digits4000_txt\digits4000_digits_vec.txt` — 784 维像素特征源数据。
- `f:\课程\Semester B\CS5487 Machine Learning：Principle & Practice\作业\CS5487 Course Project\Code\digits4000_txt\digits4000_digits_labels.txt` — 标签源数据。
- `f:\课程\Semester B\CS5487 Machine Learning：Principle & Practice\作业\CS5487 Course Project\Code\challenge\README.txt` — challenge 协议，明确不得重训、需按每个 trial 记录 challenge accuracy、且 presentation 不得展示该结果。
- `f:\课程\Semester B\CS5487 Machine Learning：Principle & Practice\作业\CS5487 Course Project\Code\challenge\cdigits_digits_vec.txt` — 150 个 challenge digits 的 784 维向量输入。
- `f:\课程\Semester B\CS5487 Machine Learning：Principle & Practice\作业\CS5487 Course Project\Code\challenge\cdigits_digits_labels.txt` — challenge ground-truth labels，用于最终 accuracy 计算。
- `f:\课程\Semester B\CS5487 Machine Learning：Principle & Practice\作业\CS5487 Course Project\Code\challenge\cdigits_eg.png` — challenge digit montage，可用于内部质检，不用于正式汇报 challenge accuracy。

- `f:\课程\Semester B\CS5487 Machine Learning：Principle & Practice\作业\CS5487 Course Project\Code\digits4000_txt\digits4000_trainset.txt` — 两次官方训练 split。
- `f:\课程\Semester B\CS5487 Machine Learning：Principle & Practice\作业\CS5487 Course Project\Code\digits4000_txt\digits4000_testset.txt` — 两次官方测试 split。

**Verification**
1. 数据读取校验：确认 `X.shape == (4000, 784)`、`y.shape == (4000,)`，且每个 trial 都是 2000 train / 2000 test，train 与 test 无重叠。
2. 协议校验：确认所有缩放、PCA 和参数选择都只在对应 trial 的训练集内完成，没有任何 test leakage。
3. 基线校验：1-NN 的两次 trial 均值应接近官方基线 0.9160；如果明显偏离，优先排查数据转置、索引与距离实现。
4. 结果完整性校验：每个模型都必须产出两次 trial 的 accuracy、F1、confusion matrix、最佳参数和可复现实验记录。
5. 报告校验：报告中必须明确说明 Random Forest 与 MLP 是课外分类器，并解释它们与课内模型的差异、优缺点和实验表现。
6. challenge 协议校验：challenge 特征应读取为 `X_challenge.shape == (150, 784)`，且每个 trial 的 challenge 预测必须直接复用该 trial 在 digits4000 训练集上学到的完整 pipeline。
7. challenge 结果校验：保存 trial-1、trial-2 的 challenge accuracy，并可额外核对 1-NN 在 challenge 上的平均准确率是否大致接近 challenge/README.txt 给出的 0.683 参考值。
8. 展示边界校验：challenge accuracy 不进入 presentation/poster；如果书面报告要提及，也必须与公开展示材料隔离。

**Decisions**
- 采用方案 2：保留 proposal 原模型集，并新增 Random Forest + MLP 作为课外分类器。
- 默认使用 scikit-learn，不引入 PyTorch/TensorFlow；这样更贴合当前数据规模和课程项目周期。
- 对 logistic regression 与 SVM 保持 one-vs-all 叙述，和 proposal 保持一致；即使底层库支持其他多分类策略，也不改变报告口径。
- MLP 被视为本轮的课外分类器之一，理由是课程周计划明确覆盖到 kernel PCA，未把多层神经网络列入正式周次展开内容；报告中仍需主动说明这一边界。
- challenge 评估必须复用每个 trial 在 digits4000 训练集上得到的最终模型与预处理，不允许针对 challenge 重训或改参。
- challenge accuracy 可进入内部结果文件与给老师的邮件摘要；是否写入正式书面报告可在执行阶段保留一个受控出口，但 presentation/poster 必须排除这项结果，因为 [challenge/README.txt](challenge/README.txt) 明确禁止展示。
- 当前范围包含代码、结果和英文报告，不包含正式 poster 文件；但会保留可上 poster 的非 challenge 图表。

**Further Considerations**
1. 如果实现阶段发现 MLP 在当前样本量上极不稳定，可保留其为课外模型但收缩搜索空间，把重点放在 Random Forest 的稳定对比和误差分析上，而不是强行追求最高分。
3. 本地资源压力已被验证为真实瓶颈，因此主执行面固定为 Colab notebook，并通过 notebook 顶部变量控制 batch、合并与恢复运行；不要再把本地直跑 `.py` 当作默认路径。

2. 如果你后续决定要冲 A，可以在现有 figures 基础上追加一页 poster 计划；那会是单独增补，不影响当前主线。