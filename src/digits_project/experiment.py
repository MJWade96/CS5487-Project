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


def _json_safe(value):
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _serialize_params(params: dict[str, object]) -> str:
    safe_params = {key: _json_safe(value) for key, value in params.items()}
    return json.dumps(safe_params, sort_keys=True)


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

    cv_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []

    for trial in trials:
        X_train = dataset.X[trial.train_indices]
        y_train = dataset.y[trial.train_indices]
        X_test = dataset.X[trial.test_indices]
        y_test = dataset.y[trial.test_indices]

        print(f"=== {trial.name}: train={X_train.shape}, test={X_test.shape} ===", flush=True)

        for model_spec in model_specs:
            print(f"[{trial.name}] selecting model: {model_spec.name}", flush=True)
            winning_outcome: SearchOutcome | None = None
            model_total_runtime = 0.0

            for preprocessor_name in model_spec.preprocessors:
                print(f"    CV with preprocessing: {preprocessor_name}", flush=True)
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
                cv_rows.append(
                    {
                        "trial": trial.name,
                        "model": model_spec.name,
                        "preprocessing": preprocessor_name,
                        "best_cv_accuracy": search.best_score_,
                        "best_params": _serialize_params(search.best_params_),
                        "runtime_seconds": runtime_seconds,
                    }
                )

                if winning_outcome is None or outcome.best_cv_accuracy > winning_outcome.best_cv_accuracy:
                    winning_outcome = outcome

            if winning_outcome is None:
                raise RuntimeError(f"No winning configuration found for {trial.name} / {model_spec.name}.")

            model_path = config.PATHS.models_dir / f"{trial.name}_{model_spec.name}.joblib"
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

            final_rows.append(
                {
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
            )
            print(
                (
                    f"    selected {winning_outcome.preprocessor_name}: "
                    f"mnist_acc={mnist_metrics['accuracy']:.4f}, "
                    f"challenge_acc={challenge_metrics['accuracy']:.4f}"
                ),
                flush=True,
            )

    cv_frame = pd.DataFrame(cv_rows).sort_values(["trial", "model", "best_cv_accuracy"], ascending=[True, True, False])
    final_frame = pd.DataFrame(final_rows).sort_values(["model", "trial"])
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
