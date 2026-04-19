"""Tests for notebook-friendly runtime configuration."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from digits_project import combine_experiment_runs, configure_project, reset_project_config
from digits_project import config as project_config
from digits_project.data import load_digits_project_data


class RuntimeConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_project_config()

    def test_configure_project_updates_root_and_jobs(self) -> None:
        configure_project(root_dir=ROOT_DIR, grid_search_jobs=2, run_name="batch_a")

        self.assertEqual(project_config.ROOT_DIR, ROOT_DIR.resolve())
        self.assertEqual(project_config.GRID_SEARCH_JOBS, 2)
        self.assertEqual(project_config.PATHS.root_dir, ROOT_DIR.resolve())
        self.assertEqual(project_config.RUN_NAME, "batch_a")
        self.assertEqual(project_config.PATHS.run_dir, ROOT_DIR.resolve() / "artifacts" / "runs" / "batch_a")

    def test_reset_project_config_restores_defaults(self) -> None:
        configure_project(root_dir=ROOT_DIR, grid_search_jobs=2, run_name="batch_a")
        reset_project_config()

        self.assertEqual(project_config.ROOT_DIR, project_config.DEFAULT_ROOT_DIR)
        self.assertEqual(project_config.GRID_SEARCH_JOBS, project_config.DEFAULT_GRID_SEARCH_JOBS)
        self.assertEqual(project_config.PATHS.root_dir, project_config.DEFAULT_ROOT_DIR)
        self.assertEqual(project_config.RUN_NAME, project_config.DEFAULT_RUN_NAME)

    def test_data_loading_still_works_after_runtime_configuration(self) -> None:
        configure_project(root_dir=ROOT_DIR, grid_search_jobs=2)
        bundle = load_digits_project_data()

        self.assertEqual(bundle.X.shape, (4000, 784))
        self.assertEqual(bundle.challenge_X.shape, (150, 784))

    def test_combine_experiment_runs_merges_batch_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            self._write_batch_outputs(temp_root, "batch_light", "trial_1", "knn_1", "raw", 0.9135, 0.66)
            self._write_batch_outputs(temp_root, "batch_heavy", "trial_2", "mlp", "zscore", 0.9275, 0.71)

            combined = combine_experiment_runs(root_dir=temp_root, run_names=["batch_light", "batch_heavy"])

            self.assertEqual(set(combined["final"]["model"]), {"knn_1", "mlp"})
            self.assertEqual(set(combined["summary"]["model"]), {"knn_1", "mlp"})
            self.assertTrue((temp_root / "artifacts" / "results" / "summary_by_model.csv").exists())
            self.assertTrue((temp_root / "artifacts" / "results" / "challenge_protocol.json").exists())

    def _write_batch_outputs(
        self,
        root_dir: Path,
        run_name: str,
        trial_name: str,
        model_name: str,
        preprocessing_name: str,
        mnist_accuracy: float,
        challenge_accuracy: float,
    ) -> None:
        batch_paths = project_config.ProjectPaths(root_dir, run_name=run_name)
        batch_paths.results_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {
                    "trial": trial_name,
                    "model": model_name,
                    "preprocessing": preprocessing_name,
                    "best_cv_accuracy": mnist_accuracy,
                    "best_params": "{}",
                    "runtime_seconds": 1.0,
                }
            ]
        ).to_csv(batch_paths.results_dir / "cv_leaderboard.csv", index=False)

        pd.DataFrame(
            [
                {
                    "trial": trial_name,
                    "model": model_name,
                    "selected_preprocessing": preprocessing_name,
                    "best_cv_accuracy": mnist_accuracy,
                    "best_params": "{}",
                    "mnist_accuracy": mnist_accuracy,
                    "mnist_macro_f1": mnist_accuracy,
                    "mnist_macro_recall": mnist_accuracy,
                    "challenge_accuracy": challenge_accuracy,
                    "challenge_macro_f1": challenge_accuracy,
                    "challenge_macro_recall": challenge_accuracy,
                    "model_selection_runtime_seconds": 1.0,
                }
            ]
        ).to_csv(batch_paths.results_dir / "final_selected_models.csv", index=False)
