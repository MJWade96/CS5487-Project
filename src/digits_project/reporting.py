"""Result persistence and figure helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config


def ensure_output_dirs(paths: config.ProjectPaths | None = None) -> None:
    resolved_paths = paths or config.PATHS
    for path in (
        resolved_paths.results_dir,
        resolved_paths.figures_dir,
        resolved_paths.models_dir,
        resolved_paths.predictions_dir,
        resolved_paths.per_class_dir,
        resolved_paths.case_examples_dir,
        resolved_paths.analysis_dir,
        resolved_paths.candidate_predictions_dir,
        resolved_paths.candidate_per_class_dir,
        resolved_paths.candidate_confusions_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def clear_directory_contents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def save_json(payload: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_prediction_table(path: Path, prediction_frame: pd.DataFrame) -> None:
    prediction_frame.to_csv(path, index=False)


def save_confusion_matrix_plot(path: Path, confusion, title: str) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASS_LABELS,
        yticklabels=config.CLASS_LABELS,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_accuracy_comparison_plot(summary: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=summary, x="model", y="mnist_accuracy_mean", color="#2c7fb8")
    plt.ylabel("Mean accuracy on official test set")
    plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_accuracy_runtime_tradeoff_plot(tradeoff_summary: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    if tradeoff_summary.empty:
        plt.text(0.5, 0.5, "No trade-off rows", ha="center", va="center")
        plt.axis("off")
    else:
        x_values = tradeoff_summary["model_selection_runtime_mean_seconds"]
        y_values = tradeoff_summary["mnist_accuracy_mean"]
        x_errors = tradeoff_summary["model_selection_runtime_std_seconds"].fillna(0.0)
        y_errors = tradeoff_summary["mnist_accuracy_std"].fillna(0.0)

        plt.errorbar(
            x_values,
            y_values,
            xerr=x_errors,
            yerr=y_errors,
            fmt="o",
            ecolor="#8c96a0",
            color="#2c7fb8",
            capsize=4,
        )
        for row in tradeoff_summary.itertuples(index=False):
            plt.annotate(
                row.model,
                (row.model_selection_runtime_mean_seconds, row.mnist_accuracy_mean),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=9,
            )
        plt.xlabel("Mean model-selection runtime (seconds)")
        plt.ylabel("Mean official-test accuracy")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
