from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from fixture_utils import write_png
from zeromrg.data.iu_xray_package import (
    build_iu_xray_subset_package,
    validate_iu_xray_subset_package,
)
from zeromrg.data.iu_xray_subset import build_or_reuse_iu_subset
from zeromrg.data.schemas import canonical_json_bytes, sha256_bytes, write_json
from zeromrg.data.split import (
    build_or_reuse_selection,
    build_or_reuse_splits,
    file_sha256,
    validate_split_membership,
)
from zeromrg.data.vocabulary import build_vocabulary_artifact


def _records(count: int) -> list[dict[str, object]]:
    return [
        {
            "dataset": "iu_xray",
            "record_type": "valid_sample",
            "sample_id": f"sample-{uid}",
            "report_id": str(uid),
            "study_id": str(uid),
            "image_ids": [f"{uid}-front.png", f"{uid}-lateral.png"] if uid % 3 == 0 else [f"{uid}-front.png"],
            "source_image_paths": [],
            "report_fields": {"impression": f"impression {uid}", "findings": f"findings {uid}"},
            "image_metadata": [],
        }
        for uid in range(1, count + 1)
    ]


class IuXraySubsetTests(unittest.TestCase):
    def test_500_uid_selection_views_splits_and_train_only_assets(self) -> None:
        population = _records(600)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, selected, reused, first_hash = build_or_reuse_iu_subset(
                population,
                source_population_hash="source-population",
                source_manifest_sha256="source-manifest",
                population_size=500,
                base_seed=42,
                destination=root / "subset.json",
            )
            second, second_records, reused_second, second_hash = build_or_reuse_iu_subset(
                population,
                source_population_hash="source-population",
                source_manifest_sha256="source-manifest",
                population_size=500,
                base_seed=42,
                destination=root / "subset.json",
            )
            self.assertFalse(reused)
            self.assertTrue(reused_second)
            self.assertEqual(selection, second)
            self.assertEqual(first_hash, second_hash)
            self.assertEqual(len(selected), 500)
            self.assertEqual(selected, second_records)
            self.assertEqual(
                selection["physical_image_count"], sum(len(record["image_ids"]) for record in selected)
            )
            original = {record["study_id"]: record["image_ids"] for record in population}
            self.assertTrue(all(record["image_ids"] == original[record["study_id"]] for record in selected))
            changed_seed, _, _, _ = build_or_reuse_iu_subset(
                population,
                source_population_hash="source-population",
                source_manifest_sha256="source-manifest",
                population_size=500,
                base_seed=43,
                destination=root / "subset_changed_seed.json",
            )
            self.assertNotEqual(selection["derived_seed"], changed_seed["derived_seed"])
            self.assertNotEqual(selection["selected_uids"], changed_seed["selected_uids"])

            split, _, _ = build_or_reuse_splits(
                selected,
                "iu_xray",
                {"train": 0.7, "validation": 0.1, "test": 0.2},
                42,
                root / "splits.json",
                expected_counts={"train": 350, "validation": 50, "test": 100},
                seed_namespace="iu_xray:kaggle_500",
            )
            validate_split_membership(split, selected, "iu_xray")
            self.assertEqual(split["counts"], {"train": 350, "validation": 50, "test": 100})
            paired, _, _ = build_or_reuse_selection(
                dataset="iu_xray", purpose="paired_10", train_ids=split["train_ids"],
                validation_ids=split["validation_ids"], test_ids=split["test_ids"], count=35,
                base_seed=42, split_hash=split["split_hash"], destination=root / "paired.json",
                seed_namespace="iu_xray:kaggle_500",
            )
            prompts, _, _ = build_or_reuse_selection(
                dataset="iu_xray", purpose="prompt_ids", train_ids=split["train_ids"],
                validation_ids=split["validation_ids"], test_ids=split["test_ids"], count=250,
                base_seed=42, split_hash=split["split_hash"], destination=root / "prompts.json",
                seed_namespace="iu_xray:kaggle_500",
            )
            self.assertEqual(len(paired["ids"]), 35)
            self.assertEqual(len(prompts["ids"]), 250)
            self.assertTrue(set(paired["ids"]) <= set(split["train_ids"]))
            self.assertTrue(set(prompts["ids"]) <= set(split["train_ids"]))

            reports = {uid: f"common token_{uid}" for uid in split["train_ids"]}
            _, vocabulary = build_vocabulary_artifact(reports, split["training_membership_hash"], "test")
            held_out_uid = split["test_ids"][0]
            self.assertNotIn(f"token_{held_out_uid}", vocabulary["token_to_id"])
            self.assertEqual(vocabulary["training_report_count"], 350)

    def test_package_filters_metadata_preserves_bytes_and_revalidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            image_root = source / "images" / "images_normalized"
            for index in range(1, 5):
                write_png(image_root / f"image{index}.png", (index, index, index))
            with (source / "indiana_reports.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["uid", "MeSH", "Problems", "image", "indication", "comparison", "findings", "impression"])
                writer.writerows([
                    ["1", "one", "p1", "i1", "", "", "finding one", "impression one"],
                    ["2", "two", "p2", "i2", "", "", "finding two", "impression two"],
                    ["3", "three", "p3", "i3", "", "", "finding three", "impression three"],
                ])
            with (source / "indiana_projections.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["uid", "filename", "projection"])
                writer.writerows([
                    ["1", "image1.png", "Frontal"], ["1", "image2.png", "Lateral"],
                    ["2", "image3.png", "Frontal"], ["3", "image4.png", "Frontal"],
                ])
            subset = {
                "selected_uids": ["1", "3"], "selected_uid_count": 2,
                "artifact_hash": "subset-artifact",
            }
            subset_path = root / "subset.json"
            write_json(subset, subset_path)
            preparation = {
                "artifact_hash": "preparation-artifact", "dependency_file_hashes": {},
                "split_counts": {"train": 1, "validation": 0, "test": 1},
                "paired_10_count": 0, "prompt_count": 1,
            }
            preparation_path = root / "preparation.json"
            write_json(preparation, preparation_path)
            package_dir = root / "artifacts" / "IU-Xray-500"
            result = build_iu_xray_subset_package(
                source_path=source, subset_artifact_path=subset_path,
                preparation_summary_path=preparation_path, package_dir=package_dir,
                create_zip=True,
            )
            self.assertEqual(result["selected_uid_count"], 2)
            self.assertEqual(result["physical_image_count"], 3)
            self.assertTrue(package_dir.with_suffix(".zip").is_file())
            first_zip_hash = file_sha256(package_dir.with_suffix(".zip"))
            self.assertEqual(set(result["image_sha256"]), {"image1.png", "image2.png", "image4.png"})
            self.assertFalse((package_dir / "images" / "images_normalized" / "image3.png").exists())
            self.assertEqual(
                file_sha256(source / "images" / "images_normalized" / "image1.png"),
                result["image_sha256"]["image1.png"],
            )
            validation = validate_iu_xray_subset_package(package_dir, ["1", "3"], result["image_sha256"])
            self.assertEqual(validation["counts"]["valid_uids"], 2)
            with (package_dir / "indiana_reports.csv").open(encoding="utf-8", newline="") as handle:
                packaged_reports = {row["uid"]: row for row in csv.DictReader(handle)}
                self.assertEqual(set(packaged_reports), {"1", "3"})
                self.assertEqual(packaged_reports["1"]["findings"], "finding one")
                self.assertEqual(packaged_reports["3"]["impression"], "impression three")
            with (package_dir / "indiana_projections.csv").open(encoding="utf-8", newline="") as handle:
                self.assertEqual({row["uid"] for row in csv.DictReader(handle)}, {"1", "3"})
            package_dir.with_suffix(".zip").unlink()
            package_dir.with_suffix(".zip.sha256").unlink()
            reused = build_iu_xray_subset_package(
                source_path=source, subset_artifact_path=subset_path,
                preparation_summary_path=preparation_path, package_dir=package_dir,
                create_zip=True,
            )
            self.assertTrue(reused["reused"])
            self.assertEqual(first_zip_hash, file_sha256(package_dir.with_suffix(".zip")))


if __name__ == "__main__":
    unittest.main()
