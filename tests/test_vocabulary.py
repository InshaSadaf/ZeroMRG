from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zeromrg.data.vocabulary import (
    DecoderVocabulary,
    build_vocabulary_artifact,
    pad_token_sequences,
    tokenize,
)


class VocabularyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary, self.artifact = build_vocabulary_artifact(
            {"train-1": "No focal opacity.", "train-2": "Heart-size is normal!"},
            training_split_hash="train-hash",
            normalization_version="nfc_whitespace_v1",
        )

    def test_punctuation_is_separate_and_case_is_preserved(self) -> None:
        self.assertEqual(
            tokenize("Heart-size isn't abnormal."),
            ["Heart", "-", "size", "isn't", "abnormal", "."],
        )

    def test_vocabulary_is_train_only_and_unknown_is_used(self) -> None:
        self.assertNotIn("HiddenValidationWord", self.vocabulary.token_to_id)
        self.assertEqual(
            self.vocabulary.encode("HiddenValidationWord"), [self.vocabulary.unk_id]
        )

    def test_encode_decode_round_trip_and_boundaries(self) -> None:
        text = "Heart-size is normal!"
        ids = self.vocabulary.encode(text)
        decoded = self.vocabulary.decode(ids)
        self.assertEqual(tokenize(decoded), tokenize(text))
        decoder_input, decoder_target = self.vocabulary.decoder_pair(text)
        self.assertEqual(decoder_input[0], self.vocabulary.bos_id)
        self.assertEqual(decoder_target[-1], self.vocabulary.eos_id)
        self.assertEqual(decoder_input[1:], decoder_target[:-1])

    def test_padding_and_mask_do_not_truncate(self) -> None:
        padded, mask = pad_token_sequences([[1, 2], [3]], self.vocabulary.pad_id)
        self.assertEqual(padded, [[1, 2], [3, 0]])
        self.assertEqual(mask, [[True, True], [True, False]])
        with self.assertRaises(ValueError):
            pad_token_sequences([[1, 2]], self.vocabulary.pad_id, length=1)

    def test_artifact_hash_detects_mutation(self) -> None:
        DecoderVocabulary.from_artifact(self.artifact)
        changed = copy.deepcopy(self.artifact)
        changed["vocabulary_size"] += 1
        with self.assertRaises(ValueError):
            DecoderVocabulary.from_artifact(changed)


if __name__ == "__main__":
    unittest.main()

