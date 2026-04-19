"""Result persistence and figure helpers."""

from __future__ import annotations

import json
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
    ):
        path.mkdir(parents=True, exist_ok=True)


def save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def save_json(payload: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_prediction_table(path: Path, y_true, y_pred) -> None:
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(path, index=False)


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
