"""Central experiment settings.

The runtime can be configured from a notebook without rewriting library code.
"""

from __future__ import annotations

from pathlib import Path


DEFAULT_ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GRID_SEARCH_JOBS = 1
DEFAULT_RUN_NAME = None

SEED = 5487
CV_FOLDS = 5
REPORT_CHALLENGE_IN_PUBLIC_MATERIAL = False
CHALLENGE_REFERENCE_ACCURACY = 0.683
CLASS_LABELS = tuple(range(10))

ROOT_DIR = DEFAULT_ROOT_DIR
GRID_SEARCH_JOBS = DEFAULT_GRID_SEARCH_JOBS
RUN_NAME = DEFAULT_RUN_NAME

_UNSET = object()


def _normalize_run_name(run_name: str | None) -> str | None:
    if run_name is None:
        return None
    if not isinstance(run_name, str):
        raise ValueError("run_name must be a string or None.")
    normalized = run_name.strip()
    return normalized or None


class ProjectPaths:
    """Resolved filesystem locations used by the experiment pipeline."""

    def __init__(self, root_dir: Path, run_name: str | None = None) -> None:
        self.root_dir = root_dir
        self.src_dir = root_dir / "src"
        self.digits_dir = root_dir / "digits4000_txt"
        self.challenge_dir = root_dir / "challenge"
        self.artifacts_dir = root_dir / "artifacts"
        self.runs_dir = self.artifacts_dir / "runs"
        self.run_name = _normalize_run_name(run_name)
        # Keep batch outputs isolated so Colab runs can be combined later.
        self.run_dir = self.artifacts_dir if self.run_name is None else self.runs_dir / self.run_name
        self.results_dir = self.run_dir / "results"
        self.figures_dir = self.run_dir / "figures"
        self.models_dir = self.run_dir / "models"
        self.predictions_dir = self.results_dir / "predictions"
        self.per_class_dir = self.results_dir / "per_class"
        self.case_examples_dir = self.results_dir / "case_examples"
        self.analysis_dir = self.results_dir / "analysis"
        self.candidate_predictions_dir = self.analysis_dir / "candidate_predictions"
        self.candidate_per_class_dir = self.analysis_dir / "candidate_per_class"
        self.candidate_confusions_dir = self.analysis_dir / "candidate_confusions"

        self.digits_vec_path = self.digits_dir / "digits4000_digits_vec.txt"
        self.digits_labels_path = self.digits_dir / "digits4000_digits_labels.txt"
        self.trainset_path = self.digits_dir / "digits4000_trainset.txt"
        self.testset_path = self.digits_dir / "digits4000_testset.txt"

        self.challenge_vec_path = self.challenge_dir / "cdigits_digits_vec.txt"
        self.challenge_labels_path = self.challenge_dir / "cdigits_digits_labels.txt"


PATHS = ProjectPaths(ROOT_DIR, RUN_NAME)


def configure_project(
    root_dir: str | Path | None = None,
    grid_search_jobs: int | None = None,
    run_name: str | None | object = _UNSET,
) -> ProjectPaths:
    """Update runtime settings for notebooks or alternate workspaces."""

    global ROOT_DIR, GRID_SEARCH_JOBS, RUN_NAME, PATHS
    refresh_paths = False

    if root_dir is not None:
        ROOT_DIR = Path(root_dir).expanduser().resolve()
        refresh_paths = True

    if run_name is not _UNSET:
        RUN_NAME = _normalize_run_name(run_name)
        refresh_paths = True

    if refresh_paths:
        PATHS = ProjectPaths(ROOT_DIR, RUN_NAME)

    if grid_search_jobs is not None:
        if not isinstance(grid_search_jobs, int) or grid_search_jobs == 0:
            raise ValueError("grid_search_jobs must be -1 or a non-zero integer.")
        GRID_SEARCH_JOBS = grid_search_jobs

    return PATHS


def reset_project_config() -> ProjectPaths:
    """Restore the default project runtime settings."""

    global ROOT_DIR, GRID_SEARCH_JOBS, RUN_NAME, PATHS

    ROOT_DIR = DEFAULT_ROOT_DIR
    GRID_SEARCH_JOBS = DEFAULT_GRID_SEARCH_JOBS
    RUN_NAME = DEFAULT_RUN_NAME
    PATHS = ProjectPaths(ROOT_DIR, RUN_NAME)
    return PATHS


def get_runtime_config() -> dict[str, object]:
    """Return the current runtime settings for logging and notebook display."""

    return {
        "root_dir": ROOT_DIR,
        "grid_search_jobs": GRID_SEARCH_JOBS,
        "run_name": RUN_NAME,
    }
