"""Small CPU tests: recovery failure modes, historical behavior, and round trips."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import torch
from zeromrg_gate1_runtime import (
    ARCHITECTURE, CheckpointReportDecoder, atomic_save, enrich,
    inspect_training_state, load_checkpoint, restore_training,
    capture_rng, restore_rng,
)


class Gate1RecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.arch = {**ARCHITECTURE, "vocab_size": 8, "max_tokens": 8,
                     "d_model": 8, "num_layers": 1, "num_heads": 2,
                     "ffn_dim": 16, "dropout": 0.0}
        self.model = CheckpointReportDecoder(self.arch)
        opt = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        for index, parameter in enumerate(self.model.parameters()):
            opt.state[parameter]["momentum_buffer"] = torch.full_like(parameter, (index + 1) / 100)
        self.last = dict(epoch=2, decoder=copy.deepcopy(self.model.state_dict()),
                         optimizer=copy.deepcopy(opt.state_dict()),
                         history=[dict(epoch=1, training_loss=2.0, validation_bleu4=10.0),
                                  dict(epoch=2, training_loss=1.0, validation_bleu4=9.0)],
                         best_validation_bleu4=10.0, epochs_without_improvement=1,
                         architecture=self.arch)
        self.best = dict(epoch=1, decoder=copy.deepcopy(self.model.state_dict()), validation_bleu4=10.0)

    def test_restores_every_momentum_buffer_and_next_epoch(self):
        optimizer, result = restore_training(self.last, self.model, self.best)
        self.assertEqual(result["next_epoch"], 3)
        self.assertEqual(result["epochs_without_improvement"], 1)
        for index, parameter in enumerate(self.model.parameters()):
            self.assertTrue(torch.equal(optimizer.state[parameter]["momentum_buffer"],
                                       torch.full_like(parameter, (index + 1) / 100)))

    def test_fails_for_required_state_and_history_corruption(self):
        for field in ("epoch", "optimizer", "history", "epochs_without_improvement"):
            broken = copy.deepcopy(self.last)
            del broken[field]
            with self.assertRaises(ValueError):
                restore_training(broken, self.model, self.best)
        broken = copy.deepcopy(self.last)
        broken["history"][-1]["epoch"] = 95
        with self.assertRaisesRegex(ValueError, "contiguous"):
            restore_training(broken, self.model, self.best)

    def test_rejects_stale_best_and_unknown_scheduler(self):
        stale = {**self.best, "epoch": 8}
        with self.assertRaisesRegex(ValueError, "Recover matching weights"):
            restore_training(self.last, self.model, stale)
        self.last["scheduler"] = {"last_epoch": 2}
        with self.assertRaisesRegex(ValueError, "scheduler"):
            restore_training(self.last, self.model, self.best)

    def test_rejects_momentum_loss_misconfiguration_and_shape(self):
        mutations = (lambda c: c["optimizer"]["state"].clear(),
                     lambda c: c["optimizer"]["param_groups"][0].update(momentum=0.9),
                     lambda c: c["optimizer"]["state"][0].update(momentum_buffer=torch.zeros(1)),
                     lambda c: c["decoder"].update(position_embedding=torch.zeros(1)))
        for mutate in mutations:
            broken = copy.deepcopy(self.last)
            mutate(broken)
            with self.assertRaises(ValueError):
                inspect_training_state(broken, self.model)

    def test_rejects_nonfinite_weights_and_wrong_architecture(self):
        self.last["decoder"]["output_projection.bias"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            inspect_training_state(self.last, self.model)
        self.last["decoder"] = copy.deepcopy(self.model.state_dict())
        self.last["architecture"] = {**self.arch, "norm_first": True}
        with self.assertRaisesRegex(ValueError, "architecture"):
            inspect_training_state(self.last, self.model)

    def test_preserves_unscaled_embeddings_no_final_norm_and_causality(self):
        self.model.eval()
        captured = []
        hook = self.model.transformer.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        ids = torch.tensor([[1, 4, 5]])
        condition = torch.ones(1, 8)
        with torch.no_grad():
            logits = self.model(condition, ids)
            expected = self.model.token_embedding(ids) + self.model.position_embedding(torch.arange(3)[None])
            self.assertTrue(torch.equal(captured[0], expected))
            alternate = self.model(condition, torch.tensor([[1, 6, 7]]))
        hook.remove()
        self.assertIsNone(self.model.transformer.norm)
        self.assertTrue(torch.allclose(logits[:, 0], alternate[:, 0], atol=1e-6))

    def test_greedy_eos_and_sequence_limit(self):
        with torch.no_grad():
            self.model.output_projection.weight.zero_()
            self.model.output_projection.bias.fill_(-10)
            self.model.output_projection.bias[2] = 10
        self.assertEqual(self.model.generate(torch.ones(2, 8)).tolist(), [[1, 2], [1, 2]])
        with self.assertRaises(ValueError):
            self.model.generate(torch.ones(1, 8), max_length=9)

    def test_paper_attention_uses_raw_vectors_and_residual(self):
        image = torch.arange(768, dtype=torch.float32)[None] / 1000
        prompts = torch.stack((torch.ones(768), torch.ones(768) * 2))
        phi, attention = enrich(image, prompts)
        expected_attention = (image @ prompts.T / (768 ** 0.5)).softmax(-1)
        self.assertTrue(torch.allclose(attention, expected_attention))
        expected_phi = torch.nn.functional.normalize(image + expected_attention @ prompts, dim=-1)
        self.assertTrue(torch.allclose(phi, expected_phi))

    def test_disposable_full_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "diagnostic.pt"
            atomic_save(self.last, destination)
            restored = load_checkpoint(destination)
            _, result = restore_training(restored, self.model, self.best)
            self.assertEqual(result["epoch"], 2)
            self.assertFalse(destination.with_suffix(".pt.tmp").exists())

    def test_production_profile_parameter_count_without_allocating_weights(self):
        with torch.device("meta"):
            model = CheckpointReportDecoder()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 57_399_152)
        self.assertEqual(len(list(model.parameters())), 112)

    def test_missing_history_fields_and_non_dictionary_fail_clearly(self):
        del self.last['history'][0]['training_loss']
        with self.assertRaisesRegex(ValueError, 'History record fields'):
            inspect_training_state(self.last, self.model)
        with self.assertRaisesRegex(ValueError, 'dictionary payload'):
            inspect_training_state(torch.zeros(1), self.model)

    def test_future_checkpoint_rng_roundtrip(self):
        state = capture_rng()
        expected = torch.rand(4)
        restore_rng(state)
        self.assertTrue(torch.equal(expected, torch.rand(4)))


if __name__ == "__main__":
    unittest.main()
