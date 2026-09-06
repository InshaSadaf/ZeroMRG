from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zeromrg.utils.seed import DEFAULT_SEED, derive_seed, seed_everything


class SeedTests(unittest.TestCase):
    def test_python_and_numpy_are_repeatable(self) -> None:
        first_state = seed_everything(DEFAULT_SEED)
        first_python = [random.random() for _ in range(4)]
        first_numpy = np.random.random(4)

        second_state = seed_everything(DEFAULT_SEED)
        second_python = [random.random() for _ in range(4)]
        second_numpy = np.random.random(4)

        self.assertEqual(first_state.seed, 42)
        self.assertEqual(first_state, second_state)
        self.assertEqual(first_python, second_python)
        np.testing.assert_array_equal(first_numpy, second_numpy)

    def test_derived_seeds_are_stable_and_namespaced(self) -> None:
        self.assertEqual(derive_seed(42, "cov_ctr", "split"), derive_seed(42, "cov_ctr", "split"))
        self.assertNotEqual(derive_seed(42, "cov_ctr", "split"), derive_seed(42, "iu_xray", "split"))

    def test_negative_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            seed_everything(-1)


if __name__ == "__main__":
    unittest.main()

