"""Tests for notebook-friendly runtime configuration."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
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
        self.assertEqual(
            project_config.PATHS.analysis_dir,
            ROOT_DIR.resolve() / "artifacts" / "runs" / "batch_a" / "results" / "analysis",
        )

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
            self.assertTrue((temp_root / "artifacts" / "results" / "model_tradeoff_summary.csv").exists())
            self.assertTrue((temp_root / "artifacts" / "results" / "preprocessing_tradeoff_summary.csv").exists())
            self.assertTrue((temp_root / "artifacts" / "results" / "cv_results_detailed.csv").exists())
            self.assertTrue((temp_root / "artifacts" / "results" / "analysis" / "candidate_per_class_long.csv").exists())
            self.assertTrue((temp_root / "artifacts" / "results" / "analysis" / "cross_model_sample_comparison.csv").exists())
            self.assertTrue((temp_root / "artifacts" / "results" / "analysis" / "case_examples_enriched.csv").exists())

            self.assertEqual(len(list((temp_root / "artifacts" / "results" / "predictions").glob("*.csv"))), 4)
            self.assertEqual(len(list((temp_root / "artifacts" / "results" / "per_class").glob("*.csv"))), 4)
            self.assertEqual(len(list((temp_root / "artifacts" / "results" / "case_examples").glob("*.csv"))), 2)
            self.assertEqual(len(list((temp_root / "artifacts" / "figures").glob("*.png"))), 4)
            self.assertEqual(len(list((temp_root / "artifacts" / "models").glob("*.joblib"))), 2)

            protocol_payload = json.loads((temp_root / "artifacts" / "results" / "challenge_protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(protocol_payload["artifact_metadata"]["missing_detailed_cv_runs"], ["batch_heavy"])

    def test_combine_experiment_runs_does_not_clear_canonical_outputs_before_source_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            self._write_batch_outputs(temp_root, "batch_light", "trial_1", "knn_1", "raw", 0.9135, 0.66)

            canonical_predictions_dir = temp_root / "artifacts" / "results" / "predictions"
            canonical_predictions_dir.mkdir(parents=True, exist_ok=True)
            sentinel_path = canonical_predictions_dir / "sentinel.csv"
            sentinel_path.write_text("keep-me", encoding="utf-8")

            missing_model_path = project_config.ProjectPaths(temp_root, run_name="batch_light").models_dir / "trial_1_knn_1.joblib"
            missing_model_path.unlink()

            with self.assertRaises(FileNotFoundError):
                combine_experiment_runs(root_dir=temp_root, run_names=["batch_light"])

            self.assertTrue(sentinel_path.exists())

    def test_combine_experiment_runs_recreates_candidate_analysis_dirs_before_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            self._write_batch_outputs(temp_root, "batch_light", "trial_1", "knn_1", "raw", 0.9135, 0.66)

            batch_paths = project_config.ProjectPaths(temp_root, run_name="batch_light")
            batch_paths.candidate_predictions_dir.mkdir(parents=True, exist_ok=True)
            candidate_prediction_name = "trial_1_knn_1_raw_mnist_test.csv"
            candidate_prediction_path = batch_paths.candidate_predictions_dir / candidate_prediction_name
            pd.DataFrame(
                [
                    {
                        "sample_index": 10,
                        "sample_index_1based": 11,
                        "y_true": 0,
                        "y_pred": 0,
                        "is_correct": True,
                    }
                ]
            ).to_csv(candidate_prediction_path, index=False)

            combine_experiment_runs(root_dir=temp_root, run_names=["batch_light"])

            combined_candidate_path = (
                project_config.ProjectPaths(temp_root).candidate_predictions_dir / candidate_prediction_name
            )
            self.assertTrue(combined_candidate_path.exists())

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
        batch_paths.models_dir.mkdir(parents=True, exist_ok=True)
        batch_paths.predictions_dir.mkdir(parents=True, exist_ok=True)

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

        if run_name == "batch_light":
            pd.DataFrame(
                [
                    {
                        "trial": trial_name,
                        "model": model_name,
                        "preprocessing": preprocessing_name,
                        "params_json": "{}",
                        "params": "{}",
                        "rank_test_score": 1,
                        "mean_test_score": mnist_accuracy,
                        "std_test_score": 0.0,
                        "mean_fit_time": 0.1,
                        "std_fit_time": 0.0,
                        "mean_score_time": 0.0,
                        "std_score_time": 0.0,
                    }
                ]
            ).to_csv(batch_paths.results_dir / "cv_results_detailed.csv", index=False)

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

        joblib.dump({"model": model_name}, batch_paths.models_dir / f"{trial_name}_{model_name}.joblib")

        pd.DataFrame(
            [
                {"sample_index": 10, "sample_index_1based": 11, "y_true": 0, "y_pred": 0, "is_correct": True},
                {"sample_index": 11, "sample_index_1based": 12, "y_true": 1, "y_pred": 0, "is_correct": False},
            ]
        ).to_csv(batch_paths.predictions_dir / f"{trial_name}_{model_name}_mnist_test.csv", index=False)

        pd.DataFrame(
            [
                {"sample_index": 0, "sample_index_1based": 1, "y_true": 0, "y_pred": 0, "is_correct": True},
                {"sample_index": 1, "sample_index_1based": 2, "y_true": 1, "y_pred": 1, "is_correct": True},
            ]
        ).to_csv(batch_paths.predictions_dir / f"{trial_name}_{model_name}_challenge.csv", index=False)
