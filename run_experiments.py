"""Run the full digits project experiment suite.

The runner uses in-file settings instead of command-line flags so the workflow
stays reproducible and matches the project instructions for this workspace.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from digits_project import run_project_experiments
from digits_project import configure_project
from digits_project import config as project_config


PROJECT_ROOT = None
GRID_SEARCH_JOBS = 1
SELECTED_TRIAL_NAMES = None
SELECTED_MODEL_NAMES = None


def main() -> None:
    configure_project(root_dir=PROJECT_ROOT, grid_search_jobs=GRID_SEARCH_JOBS)
    print("Starting CS5487 digits experiments...", flush=True)
    results = run_project_experiments(
        root_dir=PROJECT_ROOT,
        grid_search_jobs=GRID_SEARCH_JOBS,
        selected_trial_names=SELECTED_TRIAL_NAMES,
        selected_model_names=SELECTED_MODEL_NAMES,
    )
    print("Finished experiments.", flush=True)
    print("Results saved under:", project_config.PATHS.results_dir, flush=True)
    print(results["summary"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
