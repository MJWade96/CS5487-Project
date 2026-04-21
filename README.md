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

### Get the project folder and artifacts folder

1. Open `digits_project_colab.ipynb` in Colab. You can keep the notebook in Drive or open it from GitHub.
2. Keep `SYNC_PROJECT_FROM_GITHUB = True` and keep `PROJECT_ROOT = "/content/CS5487-Project"` unless you have a strong reason to change it.
3. Run Cells 1 to 4 in order. The setup cells clone or fast-forward the repository into `/content/CS5487-Project` and install the Python dependencies.
4. Open Colab's left-side Files panel and click refresh. You should now see the project folder `/content/CS5487-Project/`.
5. Decide where you want the `artifacts/` folder to live:
   - Persistent Google Drive: set `USE_GOOGLE_DRIVE = True` and keep `ARTIFACTS_ROOT = "/content/drive/MyDrive/CS5487 Course Project Code"`. The setup cell mounts Drive and creates or reuses `/content/drive/MyDrive/CS5487 Course Project Code/artifacts/`.
   - Runtime-local only: keep `USE_GOOGLE_DRIVE = False`. The setup cell creates or reuses `/content/CS5487-runtime-storage/artifacts/`.
6. If you keep artifacts in runtime-local storage, leave `EXPORT_RUNTIME_ARTIFACTS_ARCHIVE = True`. The final run cell exports `/content/exports/artifacts_snapshot*.zip`, which you should download before the Colab runtime resets.

### Notebook defaults

The checked-in notebook currently defaults to:

- `USE_GOOGLE_DRIVE = False`
- `SYNC_PROJECT_FROM_GITHUB = True`
- `PROJECT_ROOT = "/content/CS5487-Project"`
- `RUNTIME_ARTIFACTS_ROOT = "/content/CS5487-runtime-storage"`
- `DOWNLOAD_RUNTIME_ARTIFACTS_ARCHIVE = True`
- `GRID_SEARCH_JOBS = 2`
- `RUN_EXPERIMENTS = False`
- `COMBINE_AFTER_RUN = False`
- `COMBINE_RUN_NAMES = ["batch_light", "batch_heavy"]`

### Recommended run order

1. Run Cells 1 to 4 to create the project folder, set up the artifacts folder, and install dependencies.
2. Run Cell 5 to confirm the dataset shapes and the two official trial names.
3. Run Cell 6 to audit the canonical output folders and see what is still missing.
4. If you want to start a new batch, set `RUN_EXPERIMENTS = True` and then set `BATCH_PRESET`, `SELECTED_TRIAL_NAMES`, `SELECTED_MODEL_NAMES`, and `RUN_NAME` as needed.
5. If you only want to rebuild canonical outputs from finished batch folders, keep `RUN_EXPERIMENTS = False`, set `COMBINE_AFTER_RUN = False`, and fill `COMBINE_RUN_NAMES` with the finished batch folder names.
6. Run the final notebook cell. It will either execute the selected batch or combine finished batch folders, then print the output paths and the challenge-protocol reminder.

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

The `artifacts/` tree is created at runtime and is intentionally not tracked by Git. In Colab it appears either under the Google Drive folder you configured or under `/content/CS5487-runtime-storage/artifacts/`.

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
