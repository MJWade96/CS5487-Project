"""Tests for interruption-safe experiment resume behavior."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from digits_project import reset_project_config
from digits_project import config as project_config
from digits_project.data import DatasetBundle, TrialSplit
from digits_project.models import ModelSpec
from digits_project import experiment


class DummyEstimator:
    def __init__(self, label: int) -> None:
        self.label = label

    def predict(self, X):
        return np.full(X.shape[0], self.label, dtype=np.int64)


class DummySearch:
    def __init__(self, estimator, best_params: dict[str, object], best_score: float) -> None:
        self.best_estimator_ = estimator
        self.best_params_ = best_params
        self.best_score_ = best_score


def _build_dummy_estimator() -> DummyEstimator:
    return DummyEstimator(0)


class ExperimentResumeTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_project_config()

    def test_run_project_experiments_resumes_partial_preprocessing_and_skips_completed_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            batch_paths = project_config.ProjectPaths(temp_root, run_name="resume_batch")
            batch_paths.results_dir.mkdir(parents=True, exist_ok=True)
            batch_paths.models_dir.mkdir(parents=True, exist_ok=True)

            # Persist one preprocessing checkpoint up front so the rerun only computes the missing piece.
            joblib.dump(
                DummyEstimator(0),
                batch_paths.models_dir / "trial_1_resume_model_raw_search.joblib",
            )
            pd.DataFrame(
                [
                    {
                        "trial": "trial_1",
                        "model": "resume_model",
                        "preprocessing": "raw",
                        "best_cv_accuracy": 0.80,
                        "best_params": "{}",
                        "runtime_seconds": 1.5,
                    }
                ]
            ).to_csv(batch_paths.results_dir / "cv_leaderboard.csv", index=False)

            first_run_calls: list[str] = []

            def fake_first_run_grid_search(X_train, y_train, pipeline, param_grid):
                first_run_calls.append(pipeline)
                return DummySearch(DummyEstimator(1), {"alpha": 1.0}, 0.90)

            first_results = self._run_with_patches(
                temp_root=temp_root,
                run_grid_search=fake_first_run_grid_search,
            )

            self.assertEqual(first_run_calls, ["zscore"])
            self.assertEqual(first_results["final"].iloc[0]["selected_preprocessing"], "zscore")
            saved_cv_frame = pd.read_csv(batch_paths.results_dir / "cv_leaderboard.csv")
            self.assertEqual(set(saved_cv_frame["preprocessing"]), {"raw", "zscore"})
            self.assertTrue((batch_paths.models_dir / "trial_1_resume_model.joblib").exists())

            second_run_calls: list[str] = []

            def fail_if_called(*args, **kwargs):
                second_run_calls.append("called")
                raise AssertionError("Completed models should be skipped on resume.")

            second_results = self._run_with_patches(
                temp_root=temp_root,
                run_grid_search=fail_if_called,
            )

            self.assertEqual(second_run_calls, [])
            self.assertEqual(len(second_results["final"]), 1)

    def _build_bundle(self) -> DatasetBundle:
        return DatasetBundle(
            X=np.zeros((4, 2), dtype=np.float32),
            y=np.array([0, 1, 0, 1], dtype=np.int64),
            challenge_X=np.zeros((2, 2), dtype=np.float32),
            challenge_y=np.array([0, 1], dtype=np.int64),
            trials=(
                TrialSplit(
                    name="trial_1",
                    train_indices=np.array([0, 1], dtype=np.int64),
                    test_indices=np.array([2, 3], dtype=np.int64),
                ),
            ),
        )

    def _compute_metrics(self, y_true, y_pred) -> dict[str, object]:
        accuracy = float(np.mean(y_true == y_pred))
        return {
            "accuracy": accuracy,
            "macro_f1": accuracy,
            "macro_recall": accuracy,
            "confusion_matrix": np.zeros((10, 10), dtype=np.int64),
        }

    def _run_with_patches(self, temp_root: Path, run_grid_search):
        model_spec = ModelSpec(
            name="resume_model",
            estimator_builder=_build_dummy_estimator,
            param_grid={},
            preprocessors=("raw", "zscore"),
        )

        with patch.object(experiment, "MODEL_SPECS", (model_spec,)):
            with patch.object(experiment, "load_digits_project_data", return_value=self._build_bundle()):
                with patch.object(experiment, "build_pipeline", side_effect=lambda preprocessor_name, estimator: preprocessor_name):
                    with patch.object(experiment, "_run_grid_search", side_effect=run_grid_search):
                        with patch.object(experiment, "compute_classification_metrics", side_effect=self._compute_metrics):
                            with patch.object(experiment, "_save_selected_outputs"):
                                with patch.object(experiment, "save_accuracy_comparison_plot"):
                                    return experiment.run_project_experiments(
                                        root_dir=temp_root,
                                        grid_search_jobs=1,
                                        selected_trial_names=["trial_1"],
                                        selected_model_names=["resume_model"],
                                        run_name="resume_batch",
                                    )


if __name__ == "__main__":
    unittest.main()