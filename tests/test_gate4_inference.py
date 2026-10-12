"""CPU-only Gate4 synthetic/mocked checks and existing epoch90 restoration; no downloads."""
import ast
import hashlib
import io
import uuid
import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
class Image:
    def __init__(self, mode): self.mode = mode
    @classmethod
    def new(cls, mode, size): return cls(mode)
    def convert(self, mode): return Image(mode)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from zeromrg_gate1_runtime import ARCHITECTURE, CheckpointReportDecoder, enrich, validate_model_state
from zeromrg_gate4_runtime import (
    FrozenImageToReport, load_inference_artifacts, preprocess_image, resolve_training_image,
    restore_epoch90, validate_prompt_bank,
)

torch.set_num_threads(2)
VDIR = ROOT / "artifacts/gate3/baseline_retrieval_20261011T055727985114Z/verified_export_20261011T060635057524Z"


class Vision(nn.Module):
    def __init__(self, width=768, nan=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Module()
        self.conv1.weight = nn.Parameter(torch.ones(1,dtype=torch.float16))
        self.width, self.nan = width, nan
    def forward(self, pixels):
        assert not torch.is_grad_enabled()
        assert pixels.dtype == self.conv1.weight.dtype
        return torch.full((len(pixels), self.width), float("nan") if self.nan else 0.5)


class MockDecoder(nn.Module):
    generate = CheckpointReportDecoder.generate
    def __init__(self, eos_after=2, nan=False):
        super().__init__()
        self.architecture = dict(ARCHITECTURE)
        self.anchor = nn.Parameter(torch.ones(1))
        self.eos_after, self.nan = eos_after, nan
    def forward(self, condition, ids):
        assert tuple(condition.shape) == (len(ids), 768)
        assert not torch.is_grad_enabled()
        logits = torch.zeros(len(ids), ids.shape[1], 368)
        for i in range(len(ids)):
            token = 2 if ids.shape[1] >= self.eos_after + i else 4
            logits[i, -1, token] = float("nan") if self.nan else 10
        return logits


def prep(image):
    assert image.mode == "RGB"
    return torch.zeros(3, 224, 224)


def pipeline(vision=None, decoder=None):
    return FrozenImageToReport(vision or Vision(), prep, decoder or MockDecoder(),
                              torch.ones(500, 768), ["<pad>", "<bos>", "<eos>", "<unk>"] + ["word"]*364)


class Gate4Tests(unittest.TestCase):
    def test_preprocess_rgb_and_shape(self):
        self.assertEqual(preprocess_image(Image.new("L", (50, 70)), prep).shape, (1, 3, 224, 224))

    def test_preprocess_rejects_shape_and_nonfinite(self):
        for tensor in (torch.zeros(224, 224), torch.full((3,224,224), float("nan"))):
            with self.assertRaises(ValueError):
                preprocess_image(Image.new("RGB", (10,10)), lambda _: tensor)

    def test_equations_raw_attention_residual_l2(self):
        gen = torch.Generator().manual_seed(42)
        image, prompts = torch.randn(2,768,generator=gen), 4*torch.randn(500,768,generator=gen)
        phi, weights = enrich(image,prompts)
        expected = torch.softmax(image @ prompts.T / math.sqrt(768),dim=-1)
        residual = image + expected @ prompts
        self.assertEqual(weights.shape,(2,500))
        torch.testing.assert_close(weights,expected)
        torch.testing.assert_close(phi,residual/residual.norm(dim=-1,keepdim=True))
        torch.testing.assert_close(phi.norm(dim=-1),torch.ones(2))
        self.assertFalse(torch.allclose(weights,torch.softmax(image @ torch.nn.functional.normalize(prompts).T/math.sqrt(768),dim=-1)))

    def test_zero_residual_and_dimension_guard(self):
        for image,prompts in ((torch.zeros(1,768),torch.zeros(500,768)),(torch.zeros(1,512),torch.ones(500,768))):
            with self.assertRaises(ValueError): enrich(image,prompts)

    def test_prompt_order_size_raw_and_membership(self):
        ids=[str(i) for i in range(500)]
        bank=dict(model="M-CLIP/XLM-Roberta-Large-Vit-L-14",normalization="raw",prompt_ids=ids,embeddings=torch.ones(500,768))
        validate_prompt_bank(bank,ids)
        for field,value in (("prompt_ids",list(reversed(ids))),("normalization","l2"),("embeddings",torch.ones(499,768))):
            changed=dict(bank,**{field:value})
            with self.assertRaises(ValueError): validate_prompt_bank(changed,ids)

    def test_greedy_eos_frozen_no_updates(self):
        pipe=pipeline()
        before=[p.clone() for module in (pipe.vision,pipe.decoder) for p in module.parameters()]
        with patch.object(torch.optim.SGD,"step",side_effect=AssertionError("optimizer forbidden")):
            result=pipe.generate_one(Image.new("L",(5,5)))
        self.assertEqual(result["output_tokens"],[1,4,2])
        self.assertEqual(result["termination"],"EOS")
        self.assertEqual(result["report_text"],"word")
        self.assertEqual(result["shapes"]["decoder_condition"],[1,768])
        self.assertFalse(result["test_references_accessed"])
        after=[p for module in (pipe.vision,pipe.decoder) for p in module.parameters()]
        for a,b in zip(before,after): self.assertTrue(torch.equal(a,b)); self.assertIsNone(b.grad); self.assertFalse(b.requires_grad)
        with self.assertRaises(ValueError): pipe.generate_one(Image.new("RGB",(5,5)))

    def test_max_length_and_finished_batch_padding(self):
        result=pipeline(decoder=MockDecoder(eos_after=999)).generate_one(Image.new("RGB",(5,5)),max_length=4)
        self.assertEqual(result["output_tokens"],[1,4,4,4]); self.assertEqual(result["termination"],"MAX_LENGTH")
        decoder=MockDecoder()
        with torch.no_grad(): tokens=decoder.generate(torch.ones(2,768),max_length=8)
        self.assertEqual(tokens.tolist(),[[1,4,2,0],[1,4,4,2]])

    def test_encoder_dimension_and_nonfinite_guard(self):
        for vision in (Vision(width=512),Vision(nan=True)):
            with self.assertRaises(ValueError): pipeline(vision=vision).generate_one(Image.new("RGB",(5,5)))

    def test_nonfinite_logits_guard(self):
        with self.assertRaises(ValueError): pipeline(decoder=MockDecoder(nan=True)).generate_one(Image.new("RGB",(5,5)))

    def test_train_image_only(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,"train.png").touch()
            self.assertEqual(resolve_training_image(directory,"train.png",["train.png"]).name,"train.png")
            for name in ("test.png","../train.png"):
                with self.assertRaises(ValueError): resolve_training_image(directory,name,["train.png"])

    def test_real_artifacts_no_heldout_reads(self):
        manifest=json.loads((VDIR/"sha256_manifest.json").read_text())["files"]
        hashes={name:manifest["dependencies/"+name]["sha256"] for name in
                ("train_manifest.csv","prompt_manifest.csv","decoder_vocabulary.json","mclip_prompt_bank.pt")}
        original_open=Path.open
        opened=[]
        def guarded_open(path,*args,**kwargs):
            opened.append(path.name)
            if path.name in {"test_manifest.csv","validation_manifest.csv","validated_manifest.csv","mclip_text_features.pt"}:
                raise AssertionError("Held-out references/cache access forbidden")
            return original_open(path,*args,**kwargs)
        with patch.object(Path,"open",guarded_open):
            artifacts=load_inference_artifacts(VDIR/"dependencies",hashes)
        self.assertEqual(len(artifacts["train_ids"]),571)
        self.assertEqual(len(artifacts["prompt_ids"]),500)
        self.assertTrue(set(artifacts["prompt_ids"]) <= set(artifacts["train_ids"]))
        self.assertEqual(artifacts["prompts"].shape,(500,768))
        self.assertTrue(all("test" not in f and "validation" not in f for f in opened))
        with self.assertRaises(ValueError): load_inference_artifacts(VDIR/"dependencies",dict(hashes,**{"train_manifest.csv":"0"*64}))

    def test_real_epoch90_strict_restore_and_decoder_input(self):
        decoder=restore_epoch90(VDIR/"recovery_best_epoch_090.pt")
        self.assertEqual(len(decoder.transformer.layers),6)
        self.assertIsNone(decoder.transformer.norm)
        self.assertEqual(decoder.position_embedding.num_embeddings,160)
        self.assertEqual(sum(p.numel() for p in decoder.parameters()),57399152)
        with torch.inference_mode():
            output=decoder(torch.ones(1,768)/math.sqrt(768),torch.tensor([[1]]))
        self.assertEqual(output.shape,(1,1,368)); self.assertTrue(torch.isfinite(output).all())
        altered=dict(decoder.state_dict()); altered.pop(next(iter(altered)))
        with self.assertRaises(ValueError): validate_model_state(dict(decoder=altered),decoder)
        with self.assertRaises(RuntimeError): decoder.load_state_dict(altered,strict=True)
        with self.assertRaises(ValueError): decoder(torch.ones(1,512),torch.tensor([[1]]))
        with self.assertRaises(ValueError): restore_epoch90(VDIR/"recovery_best_epoch_090.pt",expected_sha256="0"*64)


class NotebookTests(unittest.TestCase):
    def test_private_metadata_embedded_source_config_compile(self):
        directory=ROOT/"notebooks/gate4/single_image"
        meta=json.loads((directory/"kernel-metadata.json").read_text())
        self.assertIs(meta["is_private"],True); self.assertIs(meta["enable_gpu"],True)
        self.assertFalse(meta["enable_tpu"])
        self.assertEqual(meta["dataset_sources"],["inshasadaf114/dataset","inshasadaf114/zeromrg-checkpoint-backup5-zip"])
        self.assertEqual(meta["kernel_sources"],["inshasadaf114/zeromrg-covctr-gate3-recovery-baseline"])
        nb=json.loads((directory/meta["code_file"]).read_text())
        ns={}
        for i,c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                self.assertIsNone(c["execution_count"]); self.assertEqual(c["outputs"],[])
                compile("".join(c["source"]),"cell"+str(i),"exec")
        exec("".join(nb["cells"][1]["source"]),ns)
        for name,source in ns["SOURCES"].items():
            self.assertEqual(source,(ROOT/"scripts"/name).read_text())
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),ns["SOURCE_HASHES"][name])
            ast.parse(source)
        cfg=ns["CONFIG"]
        self.assertEqual(cfg["max_images"],1); self.assertEqual(cfg["hard_timeout_seconds"],1800)
        self.assertFalse(cfg["training_enabled"]); self.assertFalse(cfg["test_inference_enabled"])
        self.assertEqual(len(cfg["support_lock"].strip().splitlines()),4)
        worker=ns["SOURCES"]["gate4_diagnostic_worker.py"]
        self.assertIn('dir="/tmp"',worker)
        self.assertNotIn('run_root / "selected_inputs"',worker)
        self.assertNotIn('run_root/"clip_weights"',worker)
        self.assertFalse(any("test_manifest" in f or "validation" in f for f in cfg["selected_hashes"]))

    def supervisor_fixture(self, timeout):
        nb=json.loads((ROOT/"notebooks/gate4/single_image/ZeroMRG_COV_CTR_Gate4_Single_Image_Diagnostic.ipynb").read_text())
        source="".join(nb["cells"][2]["source"]).replace("Path('/kaggle/working')","Path(TEST_ROOT)")
        ns=dict(Path=Path,uuid=uuid,hashlib=hashlib,json=json,sys=sys,SOURCES={},SOURCE_HASHES={})
        ns["CONFIG"]=dict(hard_timeout_seconds=0 if timeout else 1800)
        class Process:
            def __init__(self,command,**kwargs):
                self.pid=99999999; self.returncode=None if timeout else 0
                self.stdout=io.StringIO("fixture progress\n")
                destination=Path(command[command.index("--run-root")+1])
                (destination/"diagnostic.json").write_text(json.dumps(dict(image_count=1,split="checkpoint_train",
                    test_references_accessed=False,optimizer_created=False)))
            def poll(self): return self.returncode
            def wait(self,timeout=None): self.returncode=-9; return self.returncode
        with tempfile.TemporaryDirectory() as directory:
            ns["TEST_ROOT"]=directory
            with patch.object(Path,"is_dir",return_value=True), patch("subprocess.Popen",side_effect=Process) as launch,\
                 patch("os.killpg",create=True) as kill, patch("signal.SIGKILL",9,create=True), patch("time.monotonic",side_effect=range(1000)), patch("builtins.print"):
                if timeout:
                    with self.assertRaises(TimeoutError): exec(source,ns)
                    self.assertEqual(kill.call_count,1)
                    self.assertEqual(json.loads((ns["RUN_ROOT"]/"supervisor_failure.json").read_text())["status"],"HARD_TIMEOUT")
                else:
                    exec(source,ns)
                    self.assertEqual(kill.call_count,0)
                self.assertEqual(launch.call_count,1)
                with self.assertRaises(ValueError): exec(source,ns)
                self.assertEqual(launch.call_count,1)

    def test_supervisor_success_and_rerun_guard(self): self.supervisor_fixture(False)
    def test_supervisor_hard_timeout_no_retry(self): self.supervisor_fixture(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
