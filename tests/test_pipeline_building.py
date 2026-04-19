"""Smoke tests for preprocessing and pipeline assembly."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from digits_project.models import MODEL_SPECS, PREPROCESSORS, build_pipeline


class PipelineBuildingTests(unittest.TestCase):
    def test_pca_pipeline_contains_scaler_and_pca(self) -> None:
        logistic_spec = next(spec for spec in MODEL_SPECS if spec.name == "logistic_regression_ova")
        pipeline = build_pipeline("pca_50", logistic_spec.estimator_builder())

        self.assertIn("scaler", pipeline.named_steps)
        self.assertIn("pca", pipeline.named_steps)
        self.assertEqual(pipeline.named_steps["pca"].n_components, 50)

    def test_registered_preprocessors_cover_expected_variants(self) -> None:
        self.assertIn("raw", PREPROCESSORS)
        self.assertIn("minmax", PREPROCESSORS)
        self.assertIn("zscore", PREPROCESSORS)
        self.assertIn("pca_100", PREPROCESSORS)


if __name__ == "__main__":
    unittest.main()