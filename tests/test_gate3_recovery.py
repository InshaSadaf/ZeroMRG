"""Synthetic small CPU fixtures only: no real decoder execution or training."""
import copy
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import zeromrg_gate3_runtime as recovery
from zeromrg_gate1_runtime import ARCHITECTURE, CheckpointReportDecoder, load_checkpoint, sha256


class Gate3RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.cuda_patch = patch('torch.cuda.is_available', return_value=False)
        self.cuda_patch.start()
        self.addCleanup(self.cuda_patch.stop)
        torch.set_num_threads(1)
        architecture = {**ARCHITECTURE, 'vocab_size': 8, 'max_tokens': 8, 'd_model': 8,
                        'num_layers': 1, 'num_heads': 2, 'ffn_dim': 16, 'dropout': 0.0}
        self.model = CheckpointReportDecoder(architecture)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.001, momentum=.99)
        # Create synthetic momentum directly; no gradients, backward or optimizer updates.
        for index, parameter in enumerate(self.model.parameters()):
            self.optimizer.state[parameter]['momentum_buffer'] = torch.full_like(parameter, (index+1)/100)
        self.history = [dict(epoch=i, training_loss=1.0,
                             validation_bleu4=56.44423922571333 if i == 85 else 50.0) for i in range(1, 91)]
        self.fingerprint = dict(metric='fixture-only')
        self.hashes = {'fixture': 'a'*64}
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        baseline = self.measurement(90, 51.0, 'baseline')
        self.metadata = recovery.new_recovery_metadata(self.history, baseline, 'fixture', self.hashes,
                                                       self.fingerprint, dict(torch=torch.__version__))
        path = self.directory / 'recovery_best_epoch_090.pt'
        recovery.exclusive_save(recovery.recovery_best_payload(self.model, baseline, 'fixture', self.hashes), path)
        self.reference = recovery.best_reference(path, baseline)

    def measurement(self, epoch, score, stage='continuation'):
        filename = f'validation_epoch_{epoch:03d}.json'
        value = dict(epoch=epoch, split='validation', image_ids=['fixture-validation-id'],
                     references=['fixture reference'], predictions=['fixture prediction'], validation_bleu4=score,
                     metric_fingerprint_sha256=recovery.json_sha256(self.fingerprint))
        (self.directory / filename).write_text(json.dumps(value), encoding='utf-8')
        return dict(epoch=epoch, stage=stage, validation_bleu4=score, predictions_file=filename,
                    predictions_sha256=sha256(self.directory / filename),
                    metric_fingerprint_sha256=recovery.json_sha256(self.fingerprint))

    def payload(self, metadata=None, reference=None):
        return recovery.latest_recovery_payload(metadata or self.metadata, self.model, self.optimizer,
                                                reference or self.reference)

    def verify(self, payload):
        return recovery.validate_recovery_files(payload, self.model, self.history, self.hashes,
                                                 self.directory, ['fixture-validation-id'])

    def test_baseline_does_not_reset_original_patience_or_metric(self):
        result = self.verify(self.payload())
        self.assertEqual(result['next_epoch'], 91)
        self.assertEqual(result['epochs_without_improvement'], 5)
        self.assertEqual(result['best_validation_bleu4'], 56.44423922571333)
        self.assertEqual(result['recovery_best_validation_bleu4'], 51.0)
        self.assertFalse(result['legacy_rng_missing'])  # New checkpoint does contain RNG state.
        self.assertTrue(result['historical_rng_missing'])  # Original90 did not.
        self.assertEqual(self.metadata['original_history'], self.history)
        self.assertEqual(type(self.metadata['provenance']['torch']), str)

    def test_lower_recovery_improvement_keeps_historical_stopping(self):
        updated, improved = recovery.append_recovery_epoch(self.metadata, 91, .5, self.measurement(91, 52.0))
        self.assertTrue(improved)
        self.assertEqual(updated['original_stop_tracking']['epochs_without_improvement'], 6)
        self.assertEqual(self.metadata['epoch'], 90)
        self.assertEqual(updated['original_history'], self.history)

    def test_strict_ties_keep_previous_winner(self):
        updated, improved = recovery.append_recovery_epoch(self.metadata, 91, .5, self.measurement(91, 51.0))
        self.assertFalse(improved)
        result = self.verify(self.payload(updated))
        self.assertEqual(result['recovery_best_epoch'], 90)

    def test_new_historical_improvement_resets_only_tracking_record(self):
        updated, improved = recovery.append_recovery_epoch(self.metadata, 91, .5, self.measurement(91, 57.0))
        self.assertTrue(improved)
        self.assertEqual(updated['original_stop_tracking']['epochs_without_improvement'], 0)
        self.assertEqual(updated['historical_record'], recovery.ORIGINAL_RECORD)
        self.assertEqual(updated['original_history'][84]['validation_bleu4'], 56.44423922571333)

    def test_original_patience_stops_at135_if_no_improvement(self):
        value = self.metadata
        for epoch in range(91, 136):
            value, _ = recovery.append_recovery_epoch(value, epoch, .5, self.measurement(epoch, 50.0))
        result = self.verify(self.payload(value))
        self.assertTrue(result['stop'])
        self.assertEqual(result['epochs_without_improvement'], 50)
        with self.assertRaisesRegex(ValueError, 'stopping'):
            recovery.append_recovery_epoch(value, 136, .5, self.measurement(136, 50.0))

    def test_epoch_ceiling_and_wrong_next_epoch_fail(self):
        with self.assertRaisesRegex(ValueError, 'next completed'):
            recovery.append_recovery_epoch(self.metadata, 95, .5, self.measurement(95, 51.0))
        value = copy.deepcopy(self.metadata)
        value['epoch'] = 200
        with self.assertRaisesRegex(ValueError, 'stopping'):
            recovery.append_recovery_epoch(value, 201, .5, self.measurement(201, 51.0))

    def test_missing_optimizer_or_momentum_fails(self):
        value = self.payload()
        del value['optimizer']
        with self.assertRaisesRegex(ValueError, 'Required recovery state'):
            self.verify(value)
        value = self.payload()
        pid = next(iter(value['optimizer']['state']))
        value['optimizer'] = copy.deepcopy(value['optimizer'])
        value['optimizer']['state'][pid] = {}
        with self.assertRaisesRegex(ValueError, 'SGD state'):
            self.verify(value)
        value = self.payload()
        value['scheduler'] = {}
        with self.assertRaisesRegex(ValueError, 'no scheduler'):
            self.verify(value)

    def test_historical_mutation_and_patience_reset_fail(self):
        value = self.payload()
        value['historical_record']['best_epoch'] = 90
        with self.assertRaisesRegex(ValueError, 'historical evidence'):
            self.verify(value)
        value = self.payload()
        value['original_stop_tracking']['epochs_without_improvement'] = 0
        with self.assertRaisesRegex(ValueError, 'stopping ledger'):
            self.verify(value)

    def test_best_file_hash_and_selection_metadata_fail_closed(self):
        value = self.payload()
        value['recovery_best']['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.verify(value)
        value = self.payload()
        value['recovery_best']['epoch'] = 85
        with self.assertRaisesRegex(ValueError, 'measured validation history'):
            self.verify(value)

    def test_prediction_leakage_and_references_length_fail(self):
        for changes in (dict(split='test'), dict(image_ids=['test-id']), dict(references=[])):
            value = self.payload()
            path = self.directory / value['recovery_best']['predictions_file']
            raw = json.loads(path.read_text())
            raw.update(changes)
            path.write_text(json.dumps(raw))
            digest = sha256(path)
            value['recovery_history'][0]['predictions_sha256'] = digest
            value['recovery_best']['predictions_sha256'] = digest
            # Also retag fixture best to reach deeper population checks, never actual scientific weights.
            best_path = self.directory / value['recovery_best']['file']
            best = load_checkpoint(best_path)
            best['validation_predictions_sha256'] = digest
            recovery.atomic_save(best, best_path)
            value['recovery_best']['sha256'] = sha256(best_path)
            with self.assertRaises(ValueError):
                self.verify(value)

    def test_safe_roundtrip_restores_all_rng_and_momentum(self):
        value = self.payload()
        path = self.directory / 'recovery_latest.pt'
        recovery.exclusive_save(value, path)
        loaded = load_checkpoint(path)
        self.verify(loaded)
        state = loaded['rng_state']
        recovery.restore_recovery_rng(state)
        expected = (random.random(), float(np.random.random()), torch.rand(2))
        restored, result = recovery.restore_recovery(loaded, self.model, self.history, self.hashes,
                                                     self.directory, ['fixture-validation-id'])
        actual = (random.random(), float(np.random.random()), torch.rand(2))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        self.assertEqual(result['momentum_buffers'], len(list(self.model.parameters())))
        for parameter in self.model.parameters():
            self.assertTrue(torch.equal(restored.state[parameter]['momentum_buffer'],
                                        self.optimizer.state[parameter]['momentum_buffer']))
        with self.assertRaisesRegex(ValueError, 'refuse overwrite'):
            recovery.exclusive_save(value, path)

    def test_orphan_best_does_not_change_authoritative_latest(self):
        path = self.directory / 'recovery_best_epoch_091.pt'
        measured = self.measurement(91, 57.0)
        recovery.exclusive_save(recovery.recovery_best_payload(self.model, measured, 'fixture', self.hashes), path)
        self.assertEqual(self.verify(self.payload())['recovery_best_epoch'], 90)

    def test_changed_metric_and_nonfinite_scores_fail(self):
        measured = self.measurement(91, 52.0)
        measured['metric_fingerprint_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'Metric implementation'):
            recovery.append_recovery_epoch(self.metadata, 91, .5, measured)
        measured['validation_bleu4'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'BLEU4'):
            recovery.append_recovery_epoch(self.metadata, 91, .5, measured)

    def test_failed_training_attempt_cannot_reuse_partially_updated_kernel(self):
        from build_gate3_notebook import TRAIN
        namespace = dict(ENABLE_APPROVED_RECOVERY_TRAINING=False, MAX_NEW_EPOCHS=5,
                         metadata_state=self.metadata, recovery_checks=dict(next_epoch=91), resume_rng={},
                         require=recovery.require, print=Mock(),
                         restore_recovery_rng=Mock(side_effect=RuntimeError('simulated interruption')))
        exec(compile(TRAIN, 'guarded-training-fixture', 'exec'), namespace)
        namespace['ENABLE_APPROVED_RECOVERY_TRAINING'] = True
        with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
            namespace['run_approved_training_chunk']()
        with self.assertRaisesRegex(ValueError, 'Training already attempted'):
            namespace['run_approved_training_chunk']()
        # Re-executing the definition cell must not clear the one-attempt marker.
        namespace['ENABLE_APPROVED_RECOVERY_TRAINING'] = False
        exec(compile(TRAIN, 'guarded-training-fixture', 'exec'), namespace)
        namespace['ENABLE_APPROVED_RECOVERY_TRAINING'] = True
        with self.assertRaisesRegex(ValueError, 'Training already attempted'):
            namespace['run_approved_training_chunk']()


if __name__ == '__main__':
    unittest.main()
