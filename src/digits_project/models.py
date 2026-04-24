"""Model and preprocessing factories.

The experiment searches one shared registry so model definitions and parameter
grids stay centralized instead of being copied across trial loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC

from .config import SEED


@dataclass(frozen=True)
class PreprocessorSpec:
    name: str
    scaler: str | None = None
    pca_components: int | None = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator_builder: Callable[[], BaseEstimator]
    param_grid: dict[str, list[object]]
    preprocessors: tuple[str, ...]


PREPROCESSORS: dict[str, PreprocessorSpec] = {
    "raw": PreprocessorSpec(name="raw"),
    "minmax": PreprocessorSpec(name="minmax", scaler="minmax"),
    "zscore": PreprocessorSpec(name="zscore", scaler="zscore"),
    "pca_50": PreprocessorSpec(name="pca_50", scaler="zscore", pca_components=50),
    "pca_100": PreprocessorSpec(name="pca_100", scaler="zscore", pca_components=100),
    "pca_150": PreprocessorSpec(name="pca_150", scaler="zscore", pca_components=150),
}


def _build_knn() -> BaseEstimator:
    return KNeighborsClassifier(n_neighbors=1)


def _build_logistic_regression() -> BaseEstimator:
    return OneVsRestClassifier(
        LogisticRegression(
            solver="liblinear",
            max_iter=1000,
        ),
        n_jobs=-1,
    )


def _build_linear_svm() -> BaseEstimator:
    return OneVsRestClassifier(
        LinearSVC(
            dual=False,
            max_iter=8000,
            random_state=SEED,
        )
    )


def _build_rbf_svm() -> BaseEstimator:
    return OneVsRestClassifier(
        SVC(
            kernel="rbf",
        )
    )


def _build_random_forest() -> BaseEstimator:
    return RandomForestClassifier(
        random_state=SEED,
        n_jobs=1,
    )


def _build_mlp() -> BaseEstimator:
    return MLPClassifier(
        random_state=SEED,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=15,
    )


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="knn_1",
        estimator_builder=_build_knn,
        param_grid={
            "classifier__n_neighbors": [1, 3, 5, 7, 9],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        },
        preprocessors=("zscore", "pca_100"),
    ),
    ModelSpec(
        name="logistic_regression_ova",
        estimator_builder=_build_logistic_regression,
        param_grid={
            "classifier__estimator__C": [0.1, 1.0, 5.0, 10.0],
            "classifier__estimator__penalty": ["l2"],
        },
        preprocessors=("zscore", "pca_100"),
    ),
    ModelSpec(
        name="linear_svm_ova",
        estimator_builder=_build_linear_svm,
        param_grid={
            "classifier__estimator__C": [0.01, 0.1, 1.0, 5.0, 10.0],
            "classifier__estimator__class_weight": [None, "balanced"],
        },
        preprocessors=("zscore", "pca_100"),
    ),
    ModelSpec(
        name="rbf_svm_ova",
        estimator_builder=_build_rbf_svm,
        param_grid={
            "classifier__estimator__C": [1.0, 5.0, 10.0, 20.0, 50.0],
            "classifier__estimator__gamma": ["scale", 0.0005, 0.001, 0.005, 0.01],
        },
        preprocessors=("zscore", "pca_100"),
    ),
    ModelSpec(
        name="random_forest",
        estimator_builder=_build_random_forest,
        param_grid={
            "classifier__n_estimators": [300, 600],
            "classifier__max_depth": [None, 20, 40],
            "classifier__max_features": ["sqrt", 0.5],
            "classifier__min_samples_leaf": [1, 2],
        },
        preprocessors=("raw", "pca_100"),
    ),
    ModelSpec(
        name="mlp",
        estimator_builder=_build_mlp,
        param_grid={
            "classifier__hidden_layer_sizes": [(128,), (256,), (256, 128)],
            "classifier__alpha": [0.0001, 0.0005, 0.001],
            "classifier__learning_rate_init": [0.0005, 0.001, 0.005],
        },
        preprocessors=("minmax", "zscore", "pca_100"),
    ),
)


def build_pipeline(preprocessor_name: str, estimator: BaseEstimator) -> Pipeline:
    spec = PREPROCESSORS[preprocessor_name]
    steps: list[tuple[str, BaseEstimator]] = []

    if spec.scaler == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    elif spec.scaler == "zscore":
        steps.append(("scaler", StandardScaler()))

    if spec.pca_components is not None:
        steps.append(("pca", PCA(n_components=spec.pca_components, random_state=SEED)))

    steps.append(("classifier", estimator))
    return Pipeline(steps)