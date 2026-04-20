"""Full experiment runner for the digits project."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config
from .data import TrialSplit, load_digits_project_data
from .metrics import build_per_class_metrics_frame, compute_classification_metrics
from .models import MODEL_SPECS, build_pipeline
from .reporting import (
    ensure_output_dirs,
    save_accuracy_comparison_plot,
    save_confusion_matrix_plot,
    save_dataframe,
    save_json,
    save_prediction_table,
)


@dataclass
class SearchOutcome:
    preprocessor_name: str
    best_estimator: object
    best_params: dict[str, object]
    best_cv_accuracy: float
    runtime_seconds: float


CV_ROW_COLUMNS = [
    "trial",
    "model",
    "preprocessing",
    "best_cv_accuracy",
    "best_params",
    "runtime_seconds",
]

FINAL_ROW_COLUMNS = [
    "trial",
    "model",
    "selected_preprocessing",
    "best_cv_accuracy",
    "best_params",
    "mnist_accuracy",
    "mnist_macro_f1",
    "mnist_macro_recall",
    "challenge_accuracy",
    "challenge_macro_f1",
    "challenge_macro_recall",
    "model_selection_runtime_seconds",
]


@dataclass
class ExperimentProgress:
    """Persisted run state used to resume interrupted Colab executions."""

    paths: config.ProjectPaths
    cv_frame: pd.DataFrame
    final_frame: pd.DataFrame

    @classmethod
    def load(cls, paths: config.ProjectPaths) -> "ExperimentProgress":
        ensure_output_dirs(paths)
        cv_frame = _sort_cv_frame(_load_optional_result_frame(paths.results_dir / "cv_leaderboard.csv", CV_ROW_COLUMNS))
        final_frame = _sort_final_frame(
            _load_optional_result_frame(paths.results_dir / "final_selected_models.csv", FINAL_ROW_COLUMNS)
        )
        return cls(paths=paths, cv_frame=cv_frame, final_frame=final_frame)

    def completed_model_pairs(self) -> set[tuple[str, str]]:
        if self.final_frame.empty:
            return set()
        return set(zip(self.final_frame["trial"], self.final_frame["model"]))

    def has_completed_model(self, trial_name: str, model_name: str) -> bool:
        return (trial_name, model_name) in self.completed_model_pairs()

    def load_saved_outcomes(self, trial_name: str, model_name: str) -> dict[str, SearchOutcome]:
        if self.cv_frame.empty:
            return {}

        saved_rows = self.cv_frame[
            (self.cv_frame["trial"] == trial_name) & (self.cv_frame["model"] == model_name)
        ]
        outcomes: dict[str, SearchOutcome] = {}
        for row in saved_rows.itertuples(index=False):
            checkpoint_path = _search_checkpoint_path(self.paths, trial_name, model_name, row.preprocessing)
            if not checkpoint_path.exists():
                print(
                    (
                        f"    checkpoint metadata exists for {row.preprocessing}, "
                        "but the saved estimator is missing; rerunning it."
                    ),
                    flush=True,
                )
                continue
            outcomes[row.preprocessing] = SearchOutcome(
                preprocessor_name=row.preprocessing,
                best_estimator=joblib.load(checkpoint_path),
                best_params=_deserialize_params(row.best_params),
                best_cv_accuracy=float(row.best_cv_accuracy),
                runtime_seconds=float(row.runtime_seconds),
            )
        return outcomes

    def record_cv_outcome(self, trial_name: str, model_name: str, outcome: SearchOutcome) -> Path:
        checkpoint_path = _search_checkpoint_path(self.paths, trial_name, model_name, outcome.preprocessor_name)
        # Save the estimator before the CSV row so resume never points at a missing artifact.
        joblib.dump(outcome.best_estimator, checkpoint_path)
        self.cv_frame = _sort_cv_frame(
            _upsert_result_row(
                self.cv_frame,
                {
                    "trial": trial_name,
                    "model": model_name,
                    "preprocessing": outcome.preprocessor_name,
                    "best_cv_accuracy": outcome.best_cv_accuracy,
                    "best_params": _serialize_params(outcome.best_params),
                    "runtime_seconds": outcome.runtime_seconds,
                },
                key_columns=["trial", "model", "preprocessing"],
                columns=CV_ROW_COLUMNS,
            )
        )
        save_dataframe(self.cv_frame, self.paths.results_dir / "cv_leaderboard.csv")
        return checkpoint_path

    def record_final_row(self, final_row: dict[str, object]) -> None:
        self.final_frame = _sort_final_frame(
            _upsert_result_row(
                self.final_frame,
                final_row,
                key_columns=["trial", "model"],
                columns=FINAL_ROW_COLUMNS,
            )
        )
        summary_frame, email_frame = _build_summary_frames(self.final_frame)
        _save_result_tables(self.cv_frame, self.final_frame, summary_frame, email_frame, self.paths)


def _json_safe(value):
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _serialize_params(params: dict[str, object]) -> str:
    safe_params = {key: _json_safe(value) for key, value in params.items()}
    return json.dumps(safe_params, sort_keys=True)


def _deserialize_params(serialized_params: str) -> dict[str, object]:
    if not serialized_params:
        return {}
    return json.loads(serialized_params)


def _selected_model_path(paths: config.ProjectPaths, trial_name: str, model_name: str) -> Path:
    return paths.models_dir / f"{trial_name}_{model_name}.joblib"


def _search_checkpoint_path(
    paths: config.ProjectPaths,
    trial_name: str,
    model_name: str,
    preprocessor_name: str,
) -> Path:
    return paths.models_dir / f"{trial_name}_{model_name}_{preprocessor_name}_search.joblib"


def _load_optional_result_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)

    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Result file {path} is missing required columns: {missing}")
    return frame[columns].copy()


def _upsert_result_row(
    frame: pd.DataFrame,
    row: dict[str, object],
    key_columns: list[str],
    columns: list[str],
) -> pd.DataFrame:
    # Reuse one upsert helper so checkpoint writes stay consistent across resume tables.
    updated = pd.concat([frame, pd.DataFrame([row], columns=columns)], ignore_index=True)
    return updated.drop_duplicates(subset=key_columns, keep="last")


def _sort_cv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reindex(columns=CV_ROW_COLUMNS)
    return frame.sort_values(["trial", "model", "best_cv_accuracy"], ascending=[True, True, False]).reset_index(drop=True)


def _sort_final_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reindex(columns=FINAL_ROW_COLUMNS)
    return frame.sort_values(["model", "trial"]).reset_index(drop=True)


def _run_grid_search(X_train, y_train, pipeline, param_grid: dict[str, list[object]]) -> GridSearchCV:
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED),
        n_jobs=config.GRID_SEARCH_JOBS,
        refit=True,
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    return search


def _normalize_name_filter(selected_names: Iterable[str] | None) -> set[str] | None:
    if selected_names is None:
        return None
    if isinstance(selected_names, str):
        return {selected_names}
    return set(selected_names)


def _select_trials(
    trials: tuple[TrialSplit, ...],
    selected_trial_names: Iterable[str] | None,
) -> tuple[TrialSplit, ...]:
    selected_names = _normalize_name_filter(selected_trial_names)
    if selected_names is None:
        return trials

    filtered = tuple(trial for trial in trials if trial.name in selected_names)
    missing = selected_names.difference({trial.name for trial in filtered})
    if missing:
        raise ValueError(f"Unknown trial names requested: {sorted(missing)}")
    return filtered


def _select_model_specs(selected_model_names: Iterable[str] | None):
    selected_names = _normalize_name_filter(selected_model_names)
    if selected_names is None:
        return MODEL_SPECS

    filtered = tuple(model_spec for model_spec in MODEL_SPECS if model_spec.name in selected_names)
    missing = selected_names.difference({model_spec.name for model_spec in filtered})
    if missing:
        raise ValueError(f"Unknown model names requested: {sorted(missing)}")
    return filtered


def _print_runtime_summary(trials: tuple[TrialSplit, ...], model_specs) -> None:
    runtime_config = config.get_runtime_config()
    print(
        (
            f"Runtime config: root_dir={runtime_config['root_dir']}, "
            f"grid_search_jobs={runtime_config['grid_search_jobs']}, "
            f"run_name={runtime_config['run_name']}, "
            f"trials={[trial.name for trial in trials]}, "
            f"models={[model_spec.name for model_spec in model_specs]}"
        ),
        flush=True,
    )


def _build_summary_frames(final_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_frame = (
        final_frame.groupby("model", as_index=False)
        .agg(
            mnist_accuracy_mean=("mnist_accuracy", "mean"),
            mnist_accuracy_std=("mnist_accuracy", "std"),
            challenge_accuracy_mean=("challenge_accuracy", "mean"),
            challenge_accuracy_std=("challenge_accuracy", "std"),
            mnist_macro_f1_mean=("mnist_macro_f1", "mean"),
            challenge_macro_f1_mean=("challenge_macro_f1", "mean"),
        )
        .sort_values("mnist_accuracy_mean", ascending=False)
        .reset_index(drop=True)
    )

    email_frame = final_frame[["trial", "model", "mnist_accuracy", "challenge_accuracy"]].copy()
    email_frame["challenge_reference_knn_mean"] = config.CHALLENGE_REFERENCE_ACCURACY
    return summary_frame, email_frame


def _save_result_tables(
    cv_frame: pd.DataFrame,
    final_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    email_frame: pd.DataFrame,
    paths: config.ProjectPaths,
    source_runs: Iterable[str] | None = None,
) -> None:
    save_dataframe(cv_frame, paths.results_dir / "cv_leaderboard.csv")
    save_dataframe(final_frame, paths.results_dir / "final_selected_models.csv")
    save_dataframe(summary_frame, paths.results_dir / "summary_by_model.csv")
    save_dataframe(email_frame, paths.results_dir / "email_summary.csv")
    save_accuracy_comparison_plot(summary_frame, paths.figures_dir / "mnist_accuracy_by_model.png")
    protocol_payload = {
        "challenge_reference_accuracy": config.CHALLENGE_REFERENCE_ACCURACY,
        "public_material_can_include_challenge": False,
    }
    if source_runs is not None:
        protocol_payload["source_runs"] = list(source_runs)
    save_json(protocol_payload, paths.results_dir / "challenge_protocol.json")


def _load_required_result_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required results file: {path}")
    return pd.read_csv(path)


def _merge_result_frames(frames: list[pd.DataFrame], key_columns: list[str]) -> pd.DataFrame:
    # Reuse one merge path for all persisted tables so batch aggregation stays consistent.
    non_empty_frames = [frame for frame in frames if not frame.empty]
    if not non_empty_frames:
        raise ValueError("No result rows were available to combine.")
    merged = pd.concat(non_empty_frames, ignore_index=True)
    return merged.drop_duplicates(subset=key_columns, keep="last")


def _resolve_batch_run_names(root_dir: Path, run_names: Iterable[str] | None) -> list[str]:
    selected_names = _normalize_name_filter(run_names)
    if selected_names is not None:
        return sorted(selected_names)

    runs_dir = config.ProjectPaths(root_dir).runs_dir
    if not runs_dir.exists():
        raise FileNotFoundError(f"No batch run directory found under {runs_dir}")

    available_names = sorted(path.name for path in runs_dir.iterdir() if path.is_dir())
    if not available_names:
        raise ValueError("No batch runs are available to combine.")
    return available_names


def _save_selected_outputs(
    trial: TrialSplit,
    model_name: str,
    dataset_label: str,
    y_true,
    y_pred,
    confusion,
) -> None:
    stem = f"{trial.name}_{model_name}_{dataset_label}"
    save_prediction_table(config.PATHS.predictions_dir / f"{stem}.csv", y_true, y_pred)
    per_class_frame = build_per_class_metrics_frame(y_true, y_pred)
    save_dataframe(per_class_frame, config.PATHS.per_class_dir / f"{stem}.csv")
    if dataset_label == "mnist_test":
        save_confusion_matrix_plot(
            config.PATHS.figures_dir / f"{stem}_confusion.png",
            confusion,
            title=f"{trial.name} - {model_name} on official test set",
        )


def run_project_experiments(
    root_dir: str | Path | None = None,
    grid_search_jobs: int | None = None,
    selected_trial_names: Iterable[str] | None = None,
    selected_model_names: Iterable[str] | None = None,
    run_name: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Run model selection on official trials and evaluate challenge digits."""

    config.configure_project(root_dir=root_dir, grid_search_jobs=grid_search_jobs, run_name=run_name)
    ensure_output_dirs()
    dataset = load_digits_project_data()
    trials = _select_trials(dataset.trials, selected_trial_names)
    model_specs = _select_model_specs(selected_model_names)
    _print_runtime_summary(trials, model_specs)

    progress = ExperimentProgress.load(config.PATHS)
    target_pairs = {(trial.name, model_spec.name) for trial in trials for model_spec in model_specs}
    completed_pairs = progress.completed_model_pairs().intersection(target_pairs)
    if completed_pairs:
        print(
            (
                f"Resume state: {len(completed_pairs)} / {len(target_pairs)} selected trial/model "
                "checkpoints are already complete."
            ),
            flush=True,
        )

    total_model_pairs = len(trials) * len(model_specs)
    model_pair_index = 0

    for trial in trials:
        X_train = dataset.X[trial.train_indices]
        y_train = dataset.y[trial.train_indices]
        X_test = dataset.X[trial.test_indices]
        y_test = dataset.y[trial.test_indices]

        print(f"=== {trial.name}: train={X_train.shape}, test={X_test.shape} ===", flush=True)

        for model_spec in model_specs:
            model_pair_index += 1
            model_prefix = f"[{model_pair_index}/{total_model_pairs}] {trial.name} / {model_spec.name}"
            if progress.has_completed_model(trial.name, model_spec.name):
                print(f"{model_prefix} already completed, skipping saved checkpoint.", flush=True)
                continue

            saved_outcomes = progress.load_saved_outcomes(trial.name, model_spec.name)
            if saved_outcomes:
                print(
                    (
                        f"{model_prefix} resuming with {len(saved_outcomes)} / "
                        f"{len(model_spec.preprocessors)} preprocessing checkpoints."
                    ),
                    flush=True,
                )
            else:
                print(f"{model_prefix} selecting model.", flush=True)

            winning_outcome: SearchOutcome | None = None
            model_total_runtime = sum(outcome.runtime_seconds for outcome in saved_outcomes.values())

            for preprocessor_index, preprocessor_name in enumerate(model_spec.preprocessors, start=1):
                progress_prefix = f"    [{preprocessor_index}/{len(model_spec.preprocessors)}]"
                if preprocessor_name in saved_outcomes:
                    outcome = saved_outcomes[preprocessor_name]
                    print(
                        (
                            f"{progress_prefix} reusing checkpoint: {preprocessor_name} "
                            f"(cv_acc={outcome.best_cv_accuracy:.4f}, runtime={outcome.runtime_seconds:.1f}s)"
                        ),
                        flush=True,
                    )
                    if winning_outcome is None or outcome.best_cv_accuracy > winning_outcome.best_cv_accuracy:
                        winning_outcome = outcome
                    continue

                print(f"{progress_prefix} starting CV with preprocessing: {preprocessor_name}", flush=True)
                pipeline = build_pipeline(preprocessor_name, model_spec.estimator_builder())

                start_time = time.perf_counter()
                search = _run_grid_search(X_train, y_train, pipeline, model_spec.param_grid)
                runtime_seconds = time.perf_counter() - start_time
                model_total_runtime += runtime_seconds

                outcome = SearchOutcome(
                    preprocessor_name=preprocessor_name,
                    best_estimator=search.best_estimator_,
                    best_params=search.best_params_,
                    best_cv_accuracy=search.best_score_,
                    runtime_seconds=runtime_seconds,
                )
                checkpoint_path = progress.record_cv_outcome(
                    trial_name=trial.name,
                    model_name=model_spec.name,
                    outcome=outcome,
                )
                print(
                    (
                        f"{progress_prefix} finished {preprocessor_name} in {runtime_seconds:.1f}s: "
                        f"cv_acc={search.best_score_:.4f}, best_params={search.best_params_}"
                    ),
                    flush=True,
                )
                print(f"{progress_prefix} saved resume checkpoint: {checkpoint_path.name}", flush=True)

                if winning_outcome is None or outcome.best_cv_accuracy > winning_outcome.best_cv_accuracy:
                    winning_outcome = outcome

            if winning_outcome is None:
                raise RuntimeError(f"No winning configuration found for {trial.name} / {model_spec.name}.")

            model_path = _selected_model_path(config.PATHS, trial.name, model_spec.name)
            joblib.dump(winning_outcome.best_estimator, model_path)

            mnist_predictions = winning_outcome.best_estimator.predict(X_test)
            challenge_predictions = winning_outcome.best_estimator.predict(dataset.challenge_X)

            mnist_metrics = compute_classification_metrics(y_test, mnist_predictions)
            challenge_metrics = compute_classification_metrics(dataset.challenge_y, challenge_predictions)

            _save_selected_outputs(
                trial=trial,
                model_name=model_spec.name,
                dataset_label="mnist_test",
                y_true=y_test,
                y_pred=mnist_predictions,
                confusion=mnist_metrics["confusion_matrix"],
            )
            _save_selected_outputs(
                trial=trial,
                model_name=model_spec.name,
                dataset_label="challenge",
                y_true=dataset.challenge_y,
                y_pred=challenge_predictions,
                confusion=challenge_metrics["confusion_matrix"],
            )

            final_row = {
                "trial": trial.name,
                "model": model_spec.name,
                "selected_preprocessing": winning_outcome.preprocessor_name,
                "best_cv_accuracy": winning_outcome.best_cv_accuracy,
                "best_params": _serialize_params(winning_outcome.best_params),
                "mnist_accuracy": mnist_metrics["accuracy"],
                "mnist_macro_f1": mnist_metrics["macro_f1"],
                "mnist_macro_recall": mnist_metrics["macro_recall"],
                "challenge_accuracy": challenge_metrics["accuracy"],
                "challenge_macro_f1": challenge_metrics["macro_f1"],
                "challenge_macro_recall": challenge_metrics["macro_recall"],
                "model_selection_runtime_seconds": model_total_runtime,
            }
            progress.record_final_row(final_row)
            print(
                (
                    f"    selected {winning_outcome.preprocessor_name}: "
                    f"mnist_acc={mnist_metrics['accuracy']:.4f}, "
                    f"challenge_acc={challenge_metrics['accuracy']:.4f}, "
                    f"model_runtime={model_total_runtime:.1f}s"
                ),
                flush=True,
            )
            print(f"    refreshed run tables under {config.PATHS.results_dir}", flush=True)

    cv_frame = progress.cv_frame
    final_frame = progress.final_frame
    summary_frame, email_frame = _build_summary_frames(final_frame)
    _save_result_tables(cv_frame, final_frame, summary_frame, email_frame, config.PATHS)

    return {
        "cv": cv_frame,
        "final": final_frame,
        "summary": summary_frame,
        "email": email_frame,
    }


def combine_experiment_runs(
    root_dir: str | Path | None = None,
    run_names: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Combine batch runs into one canonical results set under artifacts/results."""

    resolved_root = config.ROOT_DIR if root_dir is None else Path(root_dir).expanduser().resolve()
    selected_run_names = _resolve_batch_run_names(resolved_root, run_names)
    combined_paths = config.ProjectPaths(resolved_root)
    ensure_output_dirs(combined_paths)

    cv_frames: list[pd.DataFrame] = []
    final_frames: list[pd.DataFrame] = []
    for run_name in selected_run_names:
        batch_paths = config.ProjectPaths(resolved_root, run_name=run_name)
        cv_frames.append(_load_required_result_frame(batch_paths.results_dir / "cv_leaderboard.csv"))
        final_frames.append(_load_required_result_frame(batch_paths.results_dir / "final_selected_models.csv"))

    cv_frame = _merge_result_frames(cv_frames, ["trial", "model", "preprocessing"])
    cv_frame = cv_frame.sort_values(["trial", "model", "best_cv_accuracy"], ascending=[True, True, False]).reset_index(drop=True)
    final_frame = _merge_result_frames(final_frames, ["trial", "model"])
    final_frame = final_frame.sort_values(["model", "trial"]).reset_index(drop=True)
    summary_frame, email_frame = _build_summary_frames(final_frame)
    _save_result_tables(cv_frame, final_frame, summary_frame, email_frame, combined_paths, source_runs=selected_run_names)

    return {
        "cv": cv_frame,
        "final": final_frame,
        "summary": summary_frame,
        "email": email_frame,
    }
