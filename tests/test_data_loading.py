"""Smoke tests for dataset loading and official split reconstruction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from digits_project.data import load_digits_project_data


class DataLoadingTests(unittest.TestCase):
    def test_official_protocol_shapes(self) -> None:
        bundle = load_digits_project_data()

        self.assertEqual(bundle.X.shape, (4000, 784))
        self.assertEqual(bundle.y.shape, (4000,))
        self.assertEqual(bundle.challenge_X.shape, (150, 784))
        self.assertEqual(bundle.challenge_y.shape, (150,))
        self.assertEqual(len(bundle.trials), 2)

    def test_trials_are_disjoint_and_complete(self) -> None:
        bundle = load_digits_project_data()

        for trial in bundle.trials:
            self.assertEqual(trial.train_indices.shape[0], 2000)
            self.assertEqual(trial.test_indices.shape[0], 2000)
            self.assertEqual(set(trial.train_indices).intersection(set(trial.test_indices)), set())


if __name__ == "__main__":
    unittest.main()
