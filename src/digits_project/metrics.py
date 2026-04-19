"""Metric helpers shared by MNIST-test and challenge evaluation."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .config import CLASS_LABELS


def compute_classification_metrics(y_true, y_pred) -> dict[str, object]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS),
    }


def build_per_class_metrics_frame(y_true, y_pred) -> pd.DataFrame:
    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "digit": CLASS_LABELS,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "support": support,
        }
    )
