# CS5487 Digits Course Project

This workspace contains a reproducible experiment pipeline for the default
CS5487 handwritten digit classification project. The code keeps the official
digits4000 protocol, the challenge-set protocol, and the proposal methods in a
single implementation so model selection logic is not duplicated across scripts.

## What is implemented

- Official data loading for `digits4000_txt/` and `challenge/`
- Two official trials using the provided train/test index files
- Proposal models: `1-NN`, `one-vs-all logistic regression`, `one-vs-all linear SVM`, `one-vs-all RBF SVM`
- Extra classifiers for the teacher feedback: `Random Forest`, `MLP`
- Preprocessing variants: raw pixels, min-max scaling, z-score scaling, PCA 50/100/150
- Trial-specific challenge evaluation without retraining or parameter changes
- CSV/JSON/PNG outputs for reporting and private challenge tracking

## Environment

The workspace already contains a local virtual environment in `.venv/`.

Install or refresh dependencies with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you want to open the notebook locally in VS Code or Jupyter, register the
kernel once:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name cs5487-digits --display-name "CS5487 Digits"
```

## Run In Google Colab

The primary workflow is now the notebook [digits_project_colab.ipynb](digits_project_colab.ipynb).

Use the notebook when local CPU and memory are limited.

The checked-in notebook may keep work-in-progress values from the last batch setup, so treat the first code cell as configuration that must be reviewed before every run.

### Recommended starting configuration

For a clean Colab run that syncs the repository from GitHub and stores artifacts outside the checkout, start from these values in code cell 1:

```python
USE_GOOGLE_DRIVE = False
SYNC_PROJECT_FROM_GITHUB = True
GITHUB_REF = None  # or pin a commit hash for strict reproducibility
WORKSPACE_ROOT = "/content"
PROJECT_ROOT = "/content/CS5487-Project"
RUNTIME_ARTIFACTS_ROOT = "/content/CS5487-runtime-storage"
ARTIFACTS_ROOT = RUNTIME_ARTIFACTS_ROOT
EXPORT_RUNTIME_ARTIFACTS_ARCHIVE = False
DOWNLOAD_RUNTIME_ARTIFACTS_ARCHIVE = False
GRID_SEARCH_JOBS = 1
BATCH_PRESET = None
RUN_EXPERIMENTS = False
COMBINE_AFTER_RUN = False
SELECTED_TRIAL_NAMES = None
SELECTED_MODEL_NAMES = None
RUN_NAME = None
COMBINE_RUN_NAMES = ["batch_light", "batch_heavy"]
```

If you want persistent Drive storage, set `USE_GOOGLE_DRIVE = True` and change `ARTIFACTS_ROOT` to a Drive folder such as `/content/drive/MyDrive/CS5487 Course Project Code`. If you are running on a notebook service that is not Google Colab, keep `DOWNLOAD_RUNTIME_ARTIFACTS_ARCHIVE = False` because the browser download hook is Colab-specific.

After code cells 1 to 3 finish, refresh the left-side Files panel. You should see the Git checkout under `/content/CS5487-Project/`, and the artifact location will be either `/content/CS5487-runtime-storage/artifacts/`, a Drive folder, or `PROJECT_ROOT/artifacts/` depending on `ARTIFACTS_ROOT`.

### What `BATCH_PRESET` does

`BATCH_PRESET` is the main batching control for reproducing the full experiment without shrinking the model or preprocessing search space. It is applied in code cell 4, where it overrides both `RUN_NAME` and `SELECTED_MODEL_NAMES`.

- `light` writes to `artifacts/runs/batch_light/` and runs `knn_1`, `logistic_regression_ova`, and `linear_svm_ova`.
- `heavy` writes to `artifacts/runs/batch_heavy/` and runs `rbf_svm_ova`, `random_forest`, and `mlp`.
- `rbf_only`, `random_forest_only`, and `mlp_only` split the heavy batch further if the runtime is unstable.

When `BATCH_PRESET` is not `None`, any manual values you put in `RUN_NAME` or `SELECTED_MODEL_NAMES` in code cell 1 will be overwritten in code cell 4.

### Code-cell order

The current checked-in notebook has 8 code cells. Run them in order.

1. Code cell 1: review and set the configuration variables.
2. Code cell 2: mount Drive if requested, sync the GitHub checkout, and link `artifacts/`.
3. Code cell 3: install `requirements.txt`, add `src/` to `sys.path`, and clear cached `digits_project` imports.
4. Code cell 4: import the project package, apply `BATCH_PRESET`, and print the active runtime configuration.
5. Code cell 5: verify dataset shapes and official trial names.
6. Code cell 6: audit canonical outputs, batch folders, saved models, figures, and case-example files.
7. Code cell 7: start a new batch only when `RUN_EXPERIMENTS = True`.
8. Code cell 8: combine finished batch folders when `RUN_EXPERIMENTS = False` or `COMBINE_AFTER_RUN = True`, then optionally export the runtime archive.

Code cell 6 is only an audit. Before the combine stage, it is normal for it to report missing canonical outputs.

### Reproduce the full results with batches

1. Run the light batch. In code cell 1, keep the recommended starting configuration above, then set `BATCH_PRESET = "light"`, `RUN_EXPERIMENTS = True`, `COMBINE_AFTER_RUN = False`, `RUN_NAME = None`, `SELECTED_MODEL_NAMES = None`, and `COMBINE_RUN_NAMES = ["batch_light", "batch_heavy"]`. Run code cells 1 to 8 in order. Code cell 6 only audits the current state, so missing canonical outputs before combine are expected.
2. Run the heavy batch. Change only `BATCH_PRESET = "heavy"` and rerun code cells 1 to 8. If this batch is still too slow, use one of `rbf_only`, `random_forest_only`, or `mlp_only`, then later put the actual finished run-folder names into `COMBINE_RUN_NAMES`.
3. Combine the finished batches. Set `BATCH_PRESET = None`, `RUN_EXPERIMENTS = False`, then rerun code cells 1 to 8. Code cell 8 is the cell that actually calls `combine_experiment_runs(...)`.
4. Verify the canonical outputs. Rerun code cell 6. After a successful combine, the missing canonical summary files should disappear, and `Saved model files` plus `Case example files` should both be greater than `0`. `Detailed CV rows available` can still stay at `0` if the source batch folders do not contain `cv_results_detailed.csv`.
5. Preserve artifacts before the runtime resets. If `ARTIFACTS_ROOT = RUNTIME_ARTIFACTS_ROOT`, keep the whole `/content/CS5487-runtime-storage/artifacts/` tree. If `EXPORT_RUNTIME_ARTIFACTS_ARCHIVE = True`, you can instead download `/content/exports/artifacts_snapshot*.zip`. If `ARTIFACTS_ROOT = None`, keep `/content/CS5487-Project/artifacts/`. Do not keep only the source checkout.

The notebook imports the Python package instead of duplicating the experiment
logic, so the protocol stays identical between local and Colab runs. The code
checkout can now come directly from GitHub, which removes the need to manually
sync the full source tree into Google Drive.

## Local Fallback

The runner intentionally uses in-file settings instead of command-line flags so
the experiment stays fixed to the course protocol.

```powershell
.\.venv\Scripts\python.exe .\run_experiments.py
```

The script will:

1. Load the main dataset and the challenge dataset.
2. Reconstruct the two official trials.
3. Search each preprocessing/model combination with 5-fold CV on the training
   split only.
4. Evaluate the selected pipeline on the official MNIST test split.
5. Reuse the same fitted pipeline on the challenge digits without retraining.

For local fallback, keep `GRID_SEARCH_JOBS` conservative. The default runtime
configuration now uses `1` instead of `-1` so the script does not grab every CPU
core by default. The recommended path is still the notebook, because batch runs
can be resumed and combined more safely there than in one long local process.

## Outputs

Outputs are written under `artifacts/`:

The `artifacts/` tree is created at runtime and is intentionally not tracked by Git. If `ARTIFACTS_ROOT = None`, it stays under `PROJECT_ROOT/artifacts/`. If `ARTIFACTS_ROOT` points to `/content/CS5487-runtime-storage` or a Drive folder, `PROJECT_ROOT/artifacts` becomes a link to that external location.

- `artifacts/results/cv_leaderboard.csv`: best CV score for each trial/model/preprocessing combination
- `artifacts/results/cv_results_detailed.csv`: full `GridSearchCV` rows for sensitivity analysis when the source batch was run with the updated pipeline
- `artifacts/results/final_selected_models.csv`: per-trial selected model results on official test and challenge digits
- `artifacts/results/summary_by_model.csv`: mean and standard deviation by model across the two trials
- `artifacts/results/email_summary.csv`: compact table for the instructor email request
- `artifacts/results/model_tradeoff_summary.csv`: accuracy, runtime, preprocessing stability, and robustness summary by model
- `artifacts/results/preprocessing_tradeoff_summary.csv`: per-model preprocessing comparison for CV accuracy and search cost
- `artifacts/results/challenge_protocol.json`: reminder that challenge results are private to the email/report workflow
- `artifacts/results/predictions/`: prediction tables for official test and challenge digits
- `artifacts/results/per_class/`: per-class precision/recall/F1 tables
- `artifacts/results/case_examples/`: representative official-test success/failure case tables for report writing
- `artifacts/figures/`: confusion matrices, the summary bar chart, and the official-test accuracy/runtime trade-off plot
- `artifacts/models/`: fitted trial-specific pipelines saved with `joblib`

When Colab batching is enabled with `RUN_NAME`, each batch writes to `artifacts/runs/<run_name>/...`. After all batches finish, the notebook can combine them back into the canonical files under `artifacts/results/`.

When older batch folders do not contain `cv_results_detailed.csv`, the combine step still rebuilds canonical predictions, per-class tables, case-example CSVs, figures, and selected models from the saved batch outputs. However, parameter-sensitivity analysis still requires rerunning those source batches with the updated pipeline so the full `GridSearchCV` rows can be persisted.

When `ARTIFACTS_ROOT` points to Google Drive, those same `artifacts/` folders are persisted there even though the source checkout lives under `/content`.

## Verification

Run the smoke tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The runtime can also be configured from Python or notebook code:

```python
from digits_project import configure_project, run_project_experiments

configure_project(root_dir="/content/CS5487-Project", grid_search_jobs=2)
results = run_project_experiments(selected_model_names=["knn_1", "linear_svm_ova"])
```

The initial protocol sanity check for the 1-NN baseline should match the course
references closely:

- Official test mean accuracy: about `0.9160`
- Challenge mean accuracy: about `0.683`

## Challenge protocol

The file `challenge/README.txt` adds an extra rule:

- do not retrain on the challenge digits
- do not change any parameters after selecting the model on the official trial
- do not report challenge accuracy in the presentation/poster

The implementation already enforces the first two by evaluating challenge digits
with the saved trial-specific pipeline. You still need to keep challenge numbers
out of any public presentation material.

## Report scaffold

Use `report/course_project_report.md` as the canonical written report source.
It separates public official-test results from the private challenge section.

To build the submission assets from the canonical markdown files, run:

```powershell
.\.venv\Scripts\python.exe .\report\build_submission_assets.py
```

This generates:

- `report/course_project_report.docx`
- `report/course_project_report.pdf`
- `report/course_project_presentation.pptx`
