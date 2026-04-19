"""Dataset loading helpers for the digits project.

The text exports already store samples row-wise, so one shared loader handles
both the main dataset and the challenge set without special-case branches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass(frozen=True)
class TrialSplit:
    """Official train/test split for one experimental trial."""

    name: str
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass(frozen=True)
class DatasetBundle:
    """All datasets required by the project."""

    X: np.ndarray
    y: np.ndarray
    challenge_X: np.ndarray
    challenge_y: np.ndarray
    trials: tuple[TrialSplit, ...]


def _load_tab_delimited_matrix(path: str | bytes | "np.PathLike[str]") -> np.ndarray:
    matrix = np.loadtxt(path, delimiter="\t", dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    return matrix


def _load_feature_matrix(path: str | bytes | "np.PathLike[str]") -> np.ndarray:
    matrix = _load_tab_delimited_matrix(path)
    return matrix.astype(np.float32, copy=False)


def _load_label_vector(path: str | bytes | "np.PathLike[str]") -> np.ndarray:
    labels = _load_tab_delimited_matrix(path).reshape(-1)
    return labels.astype(np.int64, copy=False)


def _load_trial_columns(path: str | bytes | "np.PathLike[str]") -> np.ndarray:
    index_matrix = _load_tab_delimited_matrix(path).astype(np.int64, copy=False)
    return index_matrix - 1


def _build_trials(train_columns: np.ndarray, test_columns: np.ndarray) -> tuple[TrialSplit, ...]:
    trials: list[TrialSplit] = []
    for column_index in range(train_columns.shape[1]):
        trials.append(
            TrialSplit(
                name=f"trial_{column_index + 1}",
                train_indices=train_columns[:, column_index],
                test_indices=test_columns[:, column_index],
            )
        )
    return tuple(trials)


def _validate_bundle(bundle: DatasetBundle) -> None:
    if bundle.X.shape != (4000, 784):
        raise ValueError(f"Expected digits matrix shape (4000, 784), got {bundle.X.shape}.")
    if bundle.y.shape != (4000,):
        raise ValueError(f"Expected label vector shape (4000,), got {bundle.y.shape}.")
    if bundle.challenge_X.shape != (150, 784):
        raise ValueError(
            f"Expected challenge matrix shape (150, 784), got {bundle.challenge_X.shape}."
        )
    if bundle.challenge_y.shape != (150,):
        raise ValueError(
            f"Expected challenge label vector shape (150,), got {bundle.challenge_y.shape}."
        )

    valid_labels = set(config.CLASS_LABELS)
    if set(np.unique(bundle.y)) != valid_labels:
        raise ValueError("Main dataset labels do not cover digits 0-9 exactly once each class set.")
    if not set(np.unique(bundle.challenge_y)).issubset(valid_labels):
        raise ValueError("Challenge labels contain values outside digits 0-9.")

    for trial in bundle.trials:
        if trial.train_indices.shape[0] != 2000 or trial.test_indices.shape[0] != 2000:
            raise ValueError(f"{trial.name} does not have 2000 train and 2000 test indices.")
        if np.intersect1d(trial.train_indices, trial.test_indices).size != 0:
            raise ValueError(f"{trial.name} has overlapping train/test indices.")
        if np.any(trial.train_indices < 0) or np.any(trial.test_indices < 0):
            raise ValueError(f"{trial.name} contains negative indices after 1-based conversion.")
        if np.any(trial.train_indices >= bundle.X.shape[0]) or np.any(trial.test_indices >= bundle.X.shape[0]):
            raise ValueError(f"{trial.name} contains out-of-range sample indices.")


def load_digits_project_data() -> DatasetBundle:
    """Load the main and challenge datasets under the official protocol."""

    X = _load_feature_matrix(config.PATHS.digits_vec_path)
    y = _load_label_vector(config.PATHS.digits_labels_path)
    challenge_X = _load_feature_matrix(config.PATHS.challenge_vec_path)
    challenge_y = _load_label_vector(config.PATHS.challenge_labels_path)

    train_columns = _load_trial_columns(config.PATHS.trainset_path)
    test_columns = _load_trial_columns(config.PATHS.testset_path)
    trials = _build_trials(train_columns, test_columns)

    bundle = DatasetBundle(
        X=X,
        y=y,
        challenge_X=challenge_X,
        challenge_y=challenge_y,
        trials=trials,
    )
    _validate_bundle(bundle)
    return bundle
