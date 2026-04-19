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

## Run In Google Colab

The primary workflow is now the notebook [digits_project_colab.ipynb](f:\课程\Semester%20B\CS5487%20Machine%20Learning：Principle%20&%20Practice\作业\CS5487%20Course%20Project\Code\digits_project_colab.ipynb).

Use the notebook when local CPU and memory are limited:

1. Upload the whole project folder to Google Drive, or clone it under `/content`.
2. Open `digits_project_colab.ipynb` in Colab.
3. Set the notebook variables at the top:
   - `PROJECT_ROOT`
   - `GRID_SEARCH_JOBS` (start with `1`, raise to `2` only if the runtime stays stable)
   - `BATCH_PRESET`
   - `RUN_EXPERIMENTS`
   - `SELECTED_TRIAL_NAMES`
   - `SELECTED_MODEL_NAMES`
   - `RUN_NAME`
   - `COMBINE_RUN_NAMES`
4. Run the cells in order. The notebook now prints completed and pending `trial/model` pairs before starting a new run, so interrupted work can be resumed without guessing what is missing.
5. If you only want to rebuild canonical summary files from finished batch folders, set `RUN_EXPERIMENTS = False` and fill `COMBINE_RUN_NAMES` with the completed run names.

For a step-by-step Colab workflow and batching guidance, see [COLAB_RUN_GUIDE.md](COLAB_RUN_GUIDE.md).

The notebook imports the Python package instead of duplicating the experiment
logic, so the protocol stays identical between local and Colab runs.

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

- `artifacts/results/cv_leaderboard.csv`: best CV score for each trial/model/preprocessing combination
- `artifacts/results/final_selected_models.csv`: per-trial selected model results on official test and challenge digits
- `artifacts/results/summary_by_model.csv`: mean and standard deviation by model across the two trials
- `artifacts/results/email_summary.csv`: compact table for the instructor email request
- `artifacts/results/challenge_protocol.json`: reminder that challenge results are private to the email/report workflow
- `artifacts/results/predictions/`: prediction tables for official test and challenge digits
- `artifacts/results/per_class/`: per-class precision/recall/F1 tables
- `artifacts/figures/`: confusion matrices and summary bar chart
- `artifacts/models/`: fitted trial-specific pipelines saved with `joblib`

When Colab batching is enabled with `RUN_NAME`, each batch writes to `artifacts/runs/<run_name>/...`. After all batches finish, the notebook can combine them back into the canonical files under `artifacts/results/`.

## Verification

Run the smoke tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The runtime can also be configured from Python or notebook code:

```python
from digits_project import configure_project, run_project_experiments

configure_project(root_dir="/content/CS5487 Course Project Code", grid_search_jobs=2)
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

Use `report/course_project_report.md` as the working draft for the written
report. It already separates public results from the private challenge section.
