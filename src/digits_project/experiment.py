"""Full experiment runner for the digits project."""

from __future__ import annotations

import json
import shutil
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
    clear_directory_contents,
    ensure_output_dirs,
    save_accuracy_comparison_plot,
    save_accuracy_runtime_tradeoff_plot,
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
    cv_results_frame: pd.DataFrame | None = None


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

CV_DETAILED_KEY_COLUMNS = ["trial", "model", "preprocessing", "params_json"]
CV_DETAILED_BLOCK_COLUMNS = ["trial", "model", "preprocessing"]

PREDICTION_COLUMNS = ["sample_index", "sample_index_1based", "y_true", "y_pred", "is_correct"]

CASE_EXAMPLE_COLUMNS = [
    "sample_index",
    "sample_index_1based",
    "true_label",
    "pred_label",
    "is_correct",
    "case_type",
    "case_reason",
    "group_label",
    "group_rank",
    "example_rank",
]

MODEL_TRADEOFF_COLUMNS = [
    "model",
    "trial_count",
    "selected_preprocessing_mode",
    "selected_preprocessing_unique_count",
    "preprocessing_consistency_ratio",
    "mnist_accuracy_mean",
    "mnist_accuracy_std",
    "mnist_accuracy_range",
    "mnist_macro_f1_mean",
    "challenge_accuracy_mean",
    "challenge_accuracy_std",
    "challenge_accuracy_range",
    "challenge_macro_f1_mean",
    "model_selection_runtime_mean_seconds",
    "model_selection_runtime_std_seconds",
]

PREPROCESSING_TRADEOFF_COLUMNS = [
    "model",
    "preprocessing",
    "trial_count",
    "best_cv_accuracy_mean",
    "best_cv_accuracy_std",
    "runtime_seconds_mean",
    "runtime_seconds_std",
    "selected_trial_count",
    "selected_trial_ratio",
]


@dataclass
class ExperimentProgress:
    """Persisted run state used to resume interrupted Colab executions."""

    paths: config.ProjectPaths
    cv_frame: pd.DataFrame
    cv_detailed_frame: pd.DataFrame
    final_frame: pd.DataFrame

    @classmethod
    def load(cls, paths: config.ProjectPaths) -> "ExperimentProgress":
        ensure_output_dirs(paths)
        cv_frame = _sort_cv_frame(_load_optional_result_frame(paths.results_dir / "cv_leaderboard.csv", CV_ROW_COLUMNS))
        cv_detailed_frame = _sort_cv_detailed_frame(_load_optional_dataframe(paths.results_dir / "cv_results_detailed.csv"))
        final_frame = _sort_final_frame(
            _load_optional_result_frame(paths.results_dir / "final_selected_models.csv", FINAL_ROW_COLUMNS)
        )
        return cls(paths=paths, cv_frame=cv_frame, cv_detailed_frame=cv_detailed_frame, final_frame=final_frame)

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

        if outcome.cv_results_frame is not None and not outcome.cv_results_frame.empty:
            self.cv_detailed_frame = _sort_cv_detailed_frame(
                _replace_result_block(self.cv_detailed_frame, outcome.cv_results_frame, CV_DETAILED_BLOCK_COLUMNS)
            )
            save_dataframe(self.cv_detailed_frame, self.paths.results_dir / "cv_results_detailed.csv")

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
        _save_result_tables(
            self.cv_frame,
            self.cv_detailed_frame,
            self.final_frame,
            summary_frame,
            email_frame,
            self.paths,
        )


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(inner_value) for key, inner_value in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
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


def _prediction_path(paths: config.ProjectPaths, trial_name: str, model_name: str, dataset_label: str) -> Path:
    return paths.predictions_dir / f"{trial_name}_{model_name}_{dataset_label}.csv"


def _per_class_path(paths: config.ProjectPaths, trial_name: str, model_name: str, dataset_label: str) -> Path:
    return paths.per_class_dir / f"{trial_name}_{model_name}_{dataset_label}.csv"


def _case_examples_path(paths: config.ProjectPaths, trial_name: str, model_name: str) -> Path:
    return paths.case_examples_dir / f"{trial_name}_{model_name}_mnist_test_cases.csv"


def _confusion_figure_path(paths: config.ProjectPaths, trial_name: str, model_name: str) -> Path:
    return paths.figures_dir / f"{trial_name}_{model_name}_mnist_test_confusion.png"


def _load_optional_result_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)

    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Result file {path} is missing required columns: {missing}")
    return frame[columns].copy()


def _load_optional_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _upsert_result_row(
    frame: pd.DataFrame,
    row: dict[str, object],
    key_columns: list[str],
    columns: list[str],
) -> pd.DataFrame:
    updated = pd.concat([frame, pd.DataFrame([row], columns=columns)], ignore_index=True)
    return updated.drop_duplicates(subset=key_columns, keep="last")


def _upsert_result_rows(frame: pd.DataFrame, rows: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    if rows.empty:
        return frame.copy()
    updated = pd.concat([frame, rows], ignore_index=True, sort=False)
    return updated.drop_duplicates(subset=key_columns, keep="last")


def _replace_result_block(frame: pd.DataFrame, rows: pd.DataFrame, block_columns: list[str]) -> pd.DataFrame:
    if rows.empty:
        return frame.copy()
    if frame.empty:
        return rows.reset_index(drop=True)

    remaining = frame.copy()
    unique_blocks = rows[block_columns].drop_duplicates()
    for block in unique_blocks.itertuples(index=False):
        block_mask = pd.Series(True, index=remaining.index)
        for column, value in zip(block_columns, block):
            block_mask &= remaining[column] == value
        remaining = remaining.loc[~block_mask]
    return pd.concat([remaining, rows], ignore_index=True, sort=False)


def _sort_cv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reindex(columns=CV_ROW_COLUMNS)
    return frame.sort_values(["trial", "model", "best_cv_accuracy"], ascending=[True, True, False]).reset_index(drop=True)


def _sort_cv_detailed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reindex(columns=CV_DETAILED_KEY_COLUMNS)

    sort_columns = [column for column in ["trial", "model", "preprocessing", "rank_test_score", "mean_test_score", "params_json"] if column in frame.columns]
    ascending = []
    for column in sort_columns:
        if column == "mean_test_score":
            ascending.append(False)
        elif column == "rank_test_score":
            ascending.append(True)
        else:
            ascending.append(True)
    return frame.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


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


def _build_detailed_cv_frame(
    search: GridSearchCV,
    trial_name: str,
    model_name: str,
    preprocessor_name: str,
) -> pd.DataFrame:
    detailed_frame = pd.DataFrame(search.cv_results_).copy()
    if detailed_frame.empty:
        return pd.DataFrame(columns=CV_DETAILED_KEY_COLUMNS)

    detailed_frame.insert(0, "preprocessing", preprocessor_name)
    detailed_frame.insert(0, "model", model_name)
    detailed_frame.insert(0, "trial", trial_name)
    detailed_frame["params_json"] = detailed_frame["params"].apply(_serialize_params)
    detailed_frame["params"] = detailed_frame["params_json"]

    preferred_columns = [
        "trial",
        "model",
        "preprocessing",
        "params_json",
        "params",
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    ]
    ordered_columns = [column for column in preferred_columns if column in detailed_frame.columns]
    ordered_columns.extend(column for column in detailed_frame.columns if column not in ordered_columns)
    return _sort_cv_detailed_frame(detailed_frame[ordered_columns])


def _build_prediction_frame(sample_indices, y_true, y_pred) -> pd.DataFrame:
    sample_index_list = list(sample_indices)
    if len(sample_index_list) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("Prediction frame inputs must have matching lengths.")

    prediction_frame = pd.DataFrame(
        {
            "sample_index": sample_index_list,
            "y_true": list(y_true),
            "y_pred": list(y_pred),
        }
    )
    return _normalize_prediction_frame(prediction_frame)


def _normalize_prediction_frame(
    prediction_frame: pd.DataFrame,
    sample_indices: Iterable[int] | None = None,
) -> pd.DataFrame:
    if "y_true" not in prediction_frame.columns or "y_pred" not in prediction_frame.columns:
        raise ValueError("Prediction tables must include y_true and y_pred columns.")

    normalized = prediction_frame.copy()
    if "sample_index" not in normalized.columns:
        if sample_indices is None:
            raise ValueError("Prediction tables without sample_index require explicit sample indices.")
        normalized.insert(0, "sample_index", list(sample_indices))

    normalized["sample_index"] = normalized["sample_index"].astype(int)
    if "sample_index_1based" not in normalized.columns:
        normalized["sample_index_1based"] = normalized["sample_index"] + 1
    normalized["sample_index_1based"] = normalized["sample_index_1based"].astype(int)
    normalized["y_true"] = normalized["y_true"].astype(int)
    normalized["y_pred"] = normalized["y_pred"].astype(int)
    normalized["is_correct"] = normalized["y_true"] == normalized["y_pred"]
    return normalized[PREDICTION_COLUMNS].reset_index(drop=True)


def _build_case_examples_frame(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_prediction_frame(prediction_frame)
    rows: list[dict[str, object]] = []

    failure_frame = normalized[~normalized["is_correct"]]
    if not failure_frame.empty:
        pair_counts = failure_frame.groupby(["y_true", "y_pred"]).size().sort_values(ascending=False)
        for group_rank, ((true_label, pred_label), _) in enumerate(pair_counts.items(), start=1):
            pair_examples = failure_frame[
                (failure_frame["y_true"] == true_label) & (failure_frame["y_pred"] == pred_label)
            ].head(2)
            for example_rank, row in enumerate(pair_examples.itertuples(index=False), start=1):
                rows.append(
                    {
                        "sample_index": int(row.sample_index),
                        "sample_index_1based": int(row.sample_index_1based),
                        "true_label": int(row.y_true),
                        "pred_label": int(row.y_pred),
                        "is_correct": bool(row.is_correct),
                        "case_type": "failure",
                        "case_reason": "top_confusion_pair",
                        "group_label": f"{int(true_label)}->{int(pred_label)}",
                        "group_rank": group_rank,
                        "example_rank": example_rank,
                    }
                )
            if group_rank >= 3:
                break

    per_digit_recall = (
        normalized.groupby("y_true", as_index=False)
        .agg(support=("y_true", "size"), recall=("is_correct", "mean"))
        .sort_values(["recall", "support", "y_true"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    correct_frame = normalized[normalized["is_correct"]]
    for group_rank, stat in enumerate(per_digit_recall.itertuples(index=False), start=1):
        digit_examples = correct_frame[correct_frame["y_true"] == stat.y_true].head(1)
        if digit_examples.empty:
            continue
        for example_rank, row in enumerate(digit_examples.itertuples(index=False), start=1):
            rows.append(
                {
                    "sample_index": int(row.sample_index),
                    "sample_index_1based": int(row.sample_index_1based),
                    "true_label": int(row.y_true),
                    "pred_label": int(row.y_pred),
                    "is_correct": bool(row.is_correct),
                    "case_type": "success",
                    "case_reason": "high_recall_digit",
                    "group_label": f"digit_{int(stat.y_true)}",
                    "group_rank": group_rank,
                    "example_rank": example_rank,
                }
            )
        if group_rank >= 3:
            break

    return pd.DataFrame(rows, columns=CASE_EXAMPLE_COLUMNS)


def _save_dataset_outputs(
    paths: config.ProjectPaths,
    trial_name: str,
    model_name: str,
    dataset_label: str,
    prediction_frame: pd.DataFrame,
) -> pd.DataFrame:
    normalized = _normalize_prediction_frame(prediction_frame)
    save_prediction_table(_prediction_path(paths, trial_name, model_name, dataset_label), normalized)

    y_true = normalized["y_true"]
    y_pred = normalized["y_pred"]
    per_class_frame = build_per_class_metrics_frame(y_true, y_pred)
    save_dataframe(per_class_frame, _per_class_path(paths, trial_name, model_name, dataset_label))

    if dataset_label == "mnist_test":
        metrics = compute_classification_metrics(y_true, y_pred)
        save_confusion_matrix_plot(
            _confusion_figure_path(paths, trial_name, model_name),
            metrics["confusion_matrix"],
            title=f"{trial_name} - {model_name} on official test set",
        )
        case_examples_frame = _build_case_examples_frame(normalized)
        save_dataframe(case_examples_frame, _case_examples_path(paths, trial_name, model_name))

    return normalized


def _safe_mode(values: pd.Series) -> object:
    modes = values.mode()
    if modes.empty:
        return None
    return modes.iloc[0]


def _consistency_ratio(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(values.value_counts(normalize=True).iloc[0])


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


def _build_model_tradeoff_frame(final_frame: pd.DataFrame) -> pd.DataFrame:
    if final_frame.empty:
        return pd.DataFrame(columns=MODEL_TRADEOFF_COLUMNS)

    tradeoff_frame = (
        final_frame.groupby("model", as_index=False)
        .agg(
            trial_count=("trial", "count"),
            selected_preprocessing_mode=("selected_preprocessing", _safe_mode),
            selected_preprocessing_unique_count=("selected_preprocessing", "nunique"),
            preprocessing_consistency_ratio=("selected_preprocessing", _consistency_ratio),
            mnist_accuracy_mean=("mnist_accuracy", "mean"),
            mnist_accuracy_std=("mnist_accuracy", "std"),
            mnist_accuracy_range=("mnist_accuracy", lambda values: float(values.max() - values.min())),
            mnist_macro_f1_mean=("mnist_macro_f1", "mean"),
            challenge_accuracy_mean=("challenge_accuracy", "mean"),
            challenge_accuracy_std=("challenge_accuracy", "std"),
            challenge_accuracy_range=("challenge_accuracy", lambda values: float(values.max() - values.min())),
            challenge_macro_f1_mean=("challenge_macro_f1", "mean"),
            model_selection_runtime_mean_seconds=("model_selection_runtime_seconds", "mean"),
            model_selection_runtime_std_seconds=("model_selection_runtime_seconds", "std"),
        )
        .sort_values("mnist_accuracy_mean", ascending=False)
        .reset_index(drop=True)
    )
    return tradeoff_frame.reindex(columns=MODEL_TRADEOFF_COLUMNS)


def _build_preprocessing_tradeoff_frame(cv_frame: pd.DataFrame, final_frame: pd.DataFrame) -> pd.DataFrame:
    if cv_frame.empty:
        return pd.DataFrame(columns=PREPROCESSING_TRADEOFF_COLUMNS)

    tradeoff_frame = (
        cv_frame.groupby(["model", "preprocessing"], as_index=False)
        .agg(
            trial_count=("trial", "count"),
            best_cv_accuracy_mean=("best_cv_accuracy", "mean"),
            best_cv_accuracy_std=("best_cv_accuracy", "std"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
            runtime_seconds_std=("runtime_seconds", "std"),
        )
    )

    if final_frame.empty:
        selected_counts = pd.DataFrame(columns=["model", "preprocessing", "selected_trial_count"])
    else:
        selected_counts = (
            final_frame.rename(columns={"selected_preprocessing": "preprocessing"})
            .groupby(["model", "preprocessing"], as_index=False)
            .agg(selected_trial_count=("trial", "count"))
        )

    tradeoff_frame = tradeoff_frame.merge(selected_counts, on=["model", "preprocessing"], how="left")
    tradeoff_frame["selected_trial_count"] = tradeoff_frame["selected_trial_count"].fillna(0).astype(int)
    tradeoff_frame["selected_trial_ratio"] = tradeoff_frame["selected_trial_count"] / tradeoff_frame["trial_count"]
    tradeoff_frame = tradeoff_frame.sort_values(["model", "best_cv_accuracy_mean"], ascending=[True, False]).reset_index(drop=True)
    return tradeoff_frame.reindex(columns=PREPROCESSING_TRADEOFF_COLUMNS)


def _save_result_tables(
    cv_frame: pd.DataFrame,
    cv_detailed_frame: pd.DataFrame,
    final_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    email_frame: pd.DataFrame,
    paths: config.ProjectPaths,
    source_runs: Iterable[str] | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_tradeoff_frame = _build_model_tradeoff_frame(final_frame)
    preprocessing_tradeoff_frame = _build_preprocessing_tradeoff_frame(cv_frame, final_frame)

    save_dataframe(cv_frame, paths.results_dir / "cv_leaderboard.csv")
    save_dataframe(final_frame, paths.results_dir / "final_selected_models.csv")
    save_dataframe(summary_frame, paths.results_dir / "summary_by_model.csv")
    save_dataframe(email_frame, paths.results_dir / "email_summary.csv")
    save_dataframe(model_tradeoff_frame, paths.results_dir / "model_tradeoff_summary.csv")
    save_dataframe(preprocessing_tradeoff_frame, paths.results_dir / "preprocessing_tradeoff_summary.csv")

    detailed_path = paths.results_dir / "cv_results_detailed.csv"
    if cv_detailed_frame.empty:
        if detailed_path.exists():
            detailed_path.unlink()
    else:
        save_dataframe(cv_detailed_frame, detailed_path)

    save_accuracy_comparison_plot(summary_frame, paths.figures_dir / "mnist_accuracy_by_model.png")
    save_accuracy_runtime_tradeoff_plot(model_tradeoff_frame, paths.figures_dir / "mnist_accuracy_vs_runtime.png")

    protocol_payload = {
        "challenge_reference_accuracy": config.CHALLENGE_REFERENCE_ACCURACY,
        "public_material_can_include_challenge": False,
    }
    if source_runs is not None:
        protocol_payload["source_runs"] = list(source_runs)
    if extra_metadata:
        protocol_payload["artifact_metadata"] = extra_metadata
    save_json(protocol_payload, paths.results_dir / "challenge_protocol.json")

    return model_tradeoff_frame, preprocessing_tradeoff_frame


def _load_required_result_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required results file: {path}")
    return pd.read_csv(path)


def _merge_result_frames(frames: list[pd.DataFrame], key_columns: list[str]) -> pd.DataFrame:
    non_empty_frames = [frame for frame in frames if not frame.empty]
    if not non_empty_frames:
        raise ValueError("No result rows were available to combine.")
    merged = pd.concat(non_empty_frames, ignore_index=True, sort=False)
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


def _load_trial_lookup(root_dir: Path) -> dict[str, TrialSplit]:
    previous_runtime = config.get_runtime_config()
    try:
        config.configure_project(
            root_dir=root_dir,
            grid_search_jobs=previous_runtime["grid_search_jobs"],
            run_name=None,
        )
        bundle = load_digits_project_data()
    finally:
        config.configure_project(
            root_dir=previous_runtime["root_dir"],
            grid_search_jobs=previous_runtime["grid_search_jobs"],
            run_name=previous_runtime["run_name"],
        )
    return {trial.name: trial for trial in bundle.trials}


def _clear_canonical_artifact_dirs(paths: config.ProjectPaths) -> None:
    for path in (
        paths.models_dir,
        paths.predictions_dir,
        paths.per_class_dir,
        paths.case_examples_dir,
        paths.figures_dir,
    ):
        clear_directory_contents(path)


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
                    cv_results_frame=_build_detailed_cv_frame(
                        search,
                        trial_name=trial.name,
                        model_name=model_spec.name,
                        preprocessor_name=preprocessor_name,
                    ),
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

            _save_dataset_outputs(
                paths=config.PATHS,
                trial_name=trial.name,
                model_name=model_spec.name,
                dataset_label="mnist_test",
                prediction_frame=_build_prediction_frame(trial.test_indices, y_test, mnist_predictions),
            )
            _save_dataset_outputs(
                paths=config.PATHS,
                trial_name=trial.name,
                model_name=model_spec.name,
                dataset_label="challenge",
                prediction_frame=_build_prediction_frame(range(len(dataset.challenge_y)), dataset.challenge_y, challenge_predictions),
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
    cv_detailed_frame = progress.cv_detailed_frame
    final_frame = progress.final_frame
    summary_frame, email_frame = _build_summary_frames(final_frame)
    model_tradeoff_frame, preprocessing_tradeoff_frame = _save_result_tables(
        cv_frame,
        cv_detailed_frame,
        final_frame,
        summary_frame,
        email_frame,
        config.PATHS,
    )

    return {
        "cv": cv_frame,
        "cv_detailed": cv_detailed_frame,
        "final": final_frame,
        "summary": summary_frame,
        "email": email_frame,
        "model_tradeoff": model_tradeoff_frame,
        "preprocessing_tradeoff": preprocessing_tradeoff_frame,
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
    cv_detailed_frames: list[pd.DataFrame] = []
    run_results: list[tuple[str, config.ProjectPaths, pd.DataFrame]] = []
    missing_detailed_cv_runs: list[str] = []

    for run_name in selected_run_names:
        batch_paths = config.ProjectPaths(resolved_root, run_name=run_name)
        cv_frames.append(_load_required_result_frame(batch_paths.results_dir / "cv_leaderboard.csv"))
        batch_final_frame = _load_required_result_frame(batch_paths.results_dir / "final_selected_models.csv")
        run_results.append((run_name, batch_paths, batch_final_frame))

        detailed_path = batch_paths.results_dir / "cv_results_detailed.csv"
        if detailed_path.exists():
            cv_detailed_frames.append(pd.read_csv(detailed_path))
        else:
            missing_detailed_cv_runs.append(run_name)

    cv_frame = _sort_cv_frame(_merge_result_frames(cv_frames, ["trial", "model", "preprocessing"]))
    final_frame = _sort_final_frame(
        _merge_result_frames([batch_final_frame for _, _, batch_final_frame in run_results], ["trial", "model"])
    )
    if cv_detailed_frames:
        cv_detailed_frame = _sort_cv_detailed_frame(_merge_result_frames(cv_detailed_frames, CV_DETAILED_KEY_COLUMNS))
    else:
        cv_detailed_frame = pd.DataFrame(columns=CV_DETAILED_KEY_COLUMNS)

    staged_artifacts: list[tuple[str, str, Path, dict[str, pd.DataFrame]]] = []
    trial_lookup: dict[str, TrialSplit] | None = None

    for _, batch_paths, batch_final_frame in run_results:
        for row in batch_final_frame.itertuples(index=False):
            source_model_path = _selected_model_path(batch_paths, row.trial, row.model)
            if not source_model_path.exists():
                raise FileNotFoundError(f"Missing required selected model artifact: {source_model_path}")

            normalized_prediction_frames: dict[str, pd.DataFrame] = {}
            for dataset_label in ("mnist_test", "challenge"):
                source_prediction_frame = _load_required_result_frame(
                    _prediction_path(batch_paths, row.trial, row.model, dataset_label)
                )
                sample_indices = None
                if "sample_index" not in source_prediction_frame.columns:
                    if dataset_label == "mnist_test":
                        if trial_lookup is None:
                            trial_lookup = _load_trial_lookup(resolved_root)
                        sample_indices = trial_lookup[row.trial].test_indices
                    else:
                        sample_indices = range(len(source_prediction_frame))

                normalized_prediction_frames[dataset_label] = _normalize_prediction_frame(
                    source_prediction_frame,
                    sample_indices,
                )

            staged_artifacts.append((row.trial, row.model, source_model_path, normalized_prediction_frames))

    _clear_canonical_artifact_dirs(combined_paths)
    for trial_name, model_name, source_model_path, normalized_prediction_frames in staged_artifacts:
        shutil.copy2(source_model_path, _selected_model_path(combined_paths, trial_name, model_name))
        for dataset_label, normalized_prediction_frame in normalized_prediction_frames.items():
            _save_dataset_outputs(
                paths=combined_paths,
                trial_name=trial_name,
                model_name=model_name,
                dataset_label=dataset_label,
                prediction_frame=normalized_prediction_frame,
            )

    summary_frame, email_frame = _build_summary_frames(final_frame)
    extra_metadata: dict[str, object] | None = None
    if missing_detailed_cv_runs:
        extra_metadata = {"missing_detailed_cv_runs": missing_detailed_cv_runs}
    model_tradeoff_frame, preprocessing_tradeoff_frame = _save_result_tables(
        cv_frame,
        cv_detailed_frame,
        final_frame,
        summary_frame,
        email_frame,
        combined_paths,
        source_runs=selected_run_names,
        extra_metadata=extra_metadata,
    )

    return {
        "cv": cv_frame,
        "cv_detailed": cv_detailed_frame,
        "final": final_frame,
        "summary": summary_frame,
        "email": email_frame,
        "model_tradeoff": model_tradeoff_frame,
        "preprocessing_tradeoff": preprocessing_tradeoff_frame,
    }
