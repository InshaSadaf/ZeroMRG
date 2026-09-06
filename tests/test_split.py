from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zeromrg.data.split import (
    SplitIntegrityError,
    allocate_split_counts,
    build_or_reuse_selection,
    build_or_reuse_splits,
    validate_split_membership,
)


def records(dataset: str, count: int) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"sample-{index}",
            "study_id": str(index) if dataset == "iu_xray" else None,
            "image_ids": [f"image-{index}-front", f"image-{index}-side"]
            if dataset == "iu_xray"
            else [f"image-{index}"],
        }
        for index in range(count)
    ]


class SplitTests(unittest.TestCase):
    def test_expected_allocation_counts(self) -> None:
        self.assertEqual(
            allocate_split_counts(714, {"train": 0.8, "validation": 0.1, "test": 0.1}),
            {"train": 571, "validation": 71, "test": 72},
        )
        self.assertEqual(
            allocate_split_counts(3331, {"train": 0.7, "validation": 0.1, "test": 0.2}),
            {"train": 2332, "validation": 333, "test": 666},
        )

    def test_split_is_deterministic_disjoint_and_reused(self) -> None:
        population = records("cov_ctr", 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            first, reused_first, hash_first = build_or_reuse_splits(
                population,
                "cov_ctr",
                {"train": 0.8, "validation": 0.1, "test": 0.1},
                42,
                path,
                expected_counts={"train": 8, "validation": 1, "test": 1},
            )
            second, reused_second, hash_second = build_or_reuse_splits(
                population,
                "cov_ctr",
                {"train": 0.8, "validation": 0.1, "test": 0.1},
                42,
                path,
                expected_counts={"train": 8, "validation": 1, "test": 1},
            )
            self.assertFalse(reused_first)
            self.assertTrue(reused_second)
            self.assertEqual(first, second)
            self.assertEqual(hash_first, hash_second)
            validate_split_membership(first, population, "cov_ctr")
            self.assertFalse(set(first["train_ids"]) & set(first["test_ids"]))

    def test_existing_tampered_split_is_rejected(self) -> None:
        population = records("cov_ctr", 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            build_or_reuse_splits(
                population,
                "cov_ctr",
                {"train": 0.8, "validation": 0.1, "test": 0.1},
                42,
                path,
            )
            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact["train_ids"][0] = "tampered"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(SplitIntegrityError):
                build_or_reuse_splits(
                    population,
                    "cov_ctr",
                    {"train": 0.8, "validation": 0.1, "test": 0.1},
                    42,
                    path,
                )

    def test_iu_uses_uid_and_detects_view_leakage(self) -> None:
        population = records("iu_xray", 2)
        split = {
            "train_ids": ["0"],
            "validation_ids": ["1"],
            "test_ids": [],
        }
        validate_split_membership(split, population, "iu_xray")
        population[1]["image_ids"] = ["image-0-front"]
        with self.assertRaises(SplitIntegrityError):
            validate_split_membership(split, population, "iu_xray")

    def test_paired_and_prompt_selections_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired.json"
            args = dict(
                dataset="cov_ctr",
                purpose="paired_10",
                train_ids=[f"train-{index}" for index in range(20)],
                validation_ids=["validation"],
                test_ids=["test"],
                count=2,
                base_seed=42,
                split_hash="split-hash",
                destination=path,
            )
            first, reused_first, first_hash = build_or_reuse_selection(**args)
            second, reused_second, second_hash = build_or_reuse_selection(**args)
            self.assertFalse(reused_first)
            self.assertTrue(reused_second)
            self.assertEqual(first["ids"], second["ids"])
            self.assertEqual(first_hash, second_hash)
            self.assertEqual(len(first["ids"]), 2)
            self.assertTrue(set(first["ids"]).issubset(set(args["train_ids"])))
            prompt, prompt_reused, _ = build_or_reuse_selection(
                dataset="cov_ctr",
                purpose="prompt_ids",
                train_ids=args["train_ids"],
                validation_ids=args["validation_ids"],
                test_ids=args["test_ids"],
                count=5,
                base_seed=42,
                split_hash="split-hash",
                destination=Path(directory) / "prompts.json",
            )
            self.assertFalse(prompt_reused)
            self.assertEqual(len(prompt["ids"]), 5)
            self.assertEqual(len(prompt["ids"]), len(set(prompt["ids"])))
            self.assertTrue(set(prompt["ids"]).issubset(set(args["train_ids"])))
            self.assertNotEqual(first["derived_seed"], prompt["derived_seed"])


if __name__ == "__main__":
    unittest.main()
