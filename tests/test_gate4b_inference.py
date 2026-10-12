"""Gate4B CPU mocks, train-only membership and static notebook checks; no encoder weights."""
import ast
import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT/"tests"))
from test_gate4_inference import Image, MockDecoder, Vision
from zeromrg_gate1_runtime import ARCHITECTURE, enrich
from zeromrg_gate4b_runtime import compare_runs, diagnose_images, select_nonprompt_ids
from build_gate4b_notebook import OUT, VERIFIED, image_dimensions, selection


class SensitiveVision(Vision):
    def forward(self,pixels):
        assert not torch.is_grad_enabled()
        assert pixels.dtype == self.conv1.weight.dtype
        index = int(pixels[0,0,0,0])
        result = torch.zeros(1,768)
        result[0,0],result[0,1] = index+1,6-index
        return result


class SensitiveDecoder(MockDecoder):
    def forward(self,condition,ids):
        assert not torch.is_grad_enabled()
        logits = torch.zeros(1,ids.shape[1],368)
        token = 2 if ids.shape[1] >= 2 else (4 if condition[0,0] > condition[0,1] else 5)
        logits[0,-1,token] = 10
        return logits


class Gate4BTests(unittest.TestCase):
    def test_selection_order_bounds_and_guards(self):
        self.assertEqual(select_nonprompt_ids(["p","b","a","c"],["p"]),["b","a","c"])
        for train,prompts,limit in [(["p"],["p"],5),(["a","a"],[],5),(["a"],["x"],5),
                                    (["a"],[],6),(["a"],[],True)]:
            with self.assertRaises(ValueError): select_nonprompt_ids(train,prompts,limit)

    def exercise(self, sensitive=True):
        ids = ["a","b","c","d","e"]
        vision = SensitiveVision() if sensitive else Vision()
        decoder = SensitiveDecoder() if sensitive else MockDecoder()
        prompts = torch.full((500,768),0.001)
        vocab = ["<pad>","<bos>","<eos>","<unk>","alpha","beta"]+["word"]*362
        before = [p.clone() for module in (vision,decoder) for p in module.parameters()]
        opened = []
        @contextmanager
        def opener(identifier):
            opened.append(identifier)
            image = Image.new("L",(5,5))
            image.index = ids.index(identifier)
            yield image
        def preprocess(image):
            # convert() in the historical path retains only mode in this mock.
            assert image.mode == "RGB"
            return torch.full((3,224,224),float(ids.index(opened[-1])))
        with patch.object(torch.optim.SGD,"step",side_effect=AssertionError("No optimizer update")),\
             patch.object(torch.Tensor,"backward",side_effect=AssertionError("No backward")):
            result = diagnose_images(vision,preprocess,decoder,prompts,vocab,ids,ids+["p"],["p"],opener)
        self.assertEqual(opened,ids+[ids[0]])
        self.assertEqual(result["generation_count"],6)
        self.assertEqual(result["image_count"],5)
        self.assertTrue(result["parameter_version_counters_unchanged"])
        for old,param in zip(before,[p for module in (vision,decoder) for p in module.parameters()]):
            self.assertTrue(torch.equal(old,param)); self.assertFalse(param.requires_grad); self.assertIsNone(param.grad)
        torch.testing.assert_close(prompts,torch.full((500,768),0.001),rtol=0,atol=0)
        self.assertTrue(result["comparisons"]["determinism"]["all_observed_outputs_exact_equal"])
        for record in result["records"]:
            self.assertEqual(record["termination"],"EOS")
            self.assertEqual(record["shapes"]["decoder_condition"],[1,768])
            self.assertEqual(record["output_tokens"][-1],2)
            self.assertFalse(record["in_prompt_bank"])
        return result

    def test_captured_features_cosines_reports_and_repeat(self):
        result = self.exercise()
        comparisons = result["comparisons"]
        self.assertEqual(len(comparisons["pairs"]),10)
        self.assertTrue(comparisons["different_enriched_features_observed"])
        self.assertTrue(comparisons["different_reports_observed"])
        a,b = torch.tensor([[1.0,6.0]+[0.0]*766]),torch.tensor([[2.0,5.0]+[0.0]*766])
        expected = torch.nn.functional.cosine_similarity(a.double(),b.double()).item()
        self.assertAlmostEqual(comparisons["pairwise_cosine"]["raw"][0][1],expected,places=12)
        phi,_ = enrich(a,torch.full((500,768),0.001))
        torch.testing.assert_close(torch.tensor(result["records"][0]["enriched_embedding"]),phi)
        json.dumps(result,allow_nan=False)

    def test_identical_outputs_reported_without_claiming_sensitivity(self):
        comparisons = self.exercise(False)["comparisons"]
        self.assertFalse(comparisons["different_reports_observed"])
        self.assertFalse(comparisons["different_enriched_features_observed"])
        self.assertEqual(comparisons["distinct_report_count"],1)
        self.assertFalse(comparisons["clinical_correctness_assessed"])

    def test_repeat_mismatch_is_reported_and_invalid_features_rejected(self):
        def entry(identifier,vector,tokens):
            return dict(image_id=identifier,raw=vector,enriched=vector,
                        result=dict(output_tokens=tokens,report_text=str(tokens)))
        a,b = torch.ones(1,768),torch.ones(1,768)*2
        result = compare_runs([entry("a",a,[1,4,2]),entry("b",b,[1,4,2]),entry("a",a,[1,5,2])])
        self.assertFalse(result["determinism"]["all_observed_outputs_exact_equal"])
        self.assertFalse(result["determinism"]["tokens_exact_equal"])
        with self.assertRaises(ValueError):
            compare_runs([entry("a",torch.full((1,768),float("nan")),[1,2]),entry("a",a,[1,2])])

    def test_mutating_model_and_nonfinite_fail_closed_hooks_removed(self):
        class MutatingVision(Vision):
            def forward(self,pixels):
                self.anchor.add_(1)
                return super().forward(pixels)
        for vision in (MutatingVision(),Vision(nan=True)):
            decoder=MockDecoder()
            @contextmanager
            def opener(identifier): yield Image.new("L",(5,5))
            with self.assertRaises(ValueError):
                diagnose_images(vision,lambda _:torch.zeros(3,224,224),decoder,torch.ones(500,768),["word"]*368,
                                ["a"],["a","p"],["p"],opener)
            self.assertEqual(len(vision._forward_hooks),0)
            self.assertEqual(len(decoder._forward_pre_hooks),0)
            self.assertEqual(len(decoder._forward_hooks),0)

    def test_nonprompt_guard_prevents_any_image_open(self):
        with self.assertRaises(ValueError):
            diagnose_images(Vision(),lambda _:None,MockDecoder(),torch.ones(500,768),["w"]*368,
                ["p"],["a","p"],["p"],lambda _:self.fail("Image opened before membership validation"))

    def test_raw_image_header_formats_and_malformed_rejection(self):
        png=b"\x89PNG\r\n\x1a\n"+b"\x00\x00\x00\x0dIHDR"+struct.pack(">II",20,30)
        jpeg=b"\xff\xd8\xff\xc0"+struct.pack(">H",8)+b"\x08"+struct.pack(">HH",30,20)+b"\x03"
        self.assertEqual(image_dimensions(png),("PNG",20,30))
        self.assertEqual(image_dimensions(jpeg),("JPEG",20,30))
        for content in (b"bad",b"\xff\xd8\xff",jpeg[:8]):
            with self.assertRaises(ValueError): image_dimensions(content)

    def test_actual_train_prompt_membership_hashes_no_heldout_reads(self):
        cfg=json.loads((OUT/"configuration.json").read_text())
        original_open=Path.open
        opened=[]
        def guarded_open(filename,*args,**kwargs):
            opened.append(filename.name)
            self.assertFalse(any(word in filename.name for word in ("test_manifest","validation_manifest","mclip_text_features")))
            return original_open(filename,*args,**kwargs)
        with patch.object(Path,"open",guarded_open): evidence=selection(cfg)
        self.assertEqual(evidence["eligible_count"],71)
        self.assertEqual([r["image_id"] for r in evidence["images"]],cfg["image_ids"])
        self.assertEqual(len({r["image_id"] for r in evidence["images"]}),5)
        self.assertTrue(all(not r["in_prompt_bank"] and r["width"]>0 for r in evidence["images"]))
        self.assertFalse(evidence["heldout_reports_accessed"])

    def test_notebook_source_scope_pins_and_immutable_base(self):
        meta=json.loads((OUT/"kernel-metadata.json").read_text())
        self.assertIs(meta["is_private"],True);self.assertTrue(meta["enable_gpu"])
        self.assertEqual(meta["dataset_sources"],["inshasadaf114/dataset","inshasadaf114/zeromrg-checkpoint-backup5-zip"])
        self.assertEqual(meta["kernel_sources"],["inshasadaf114/zeromrg-covctr-gate3-recovery-baseline"])
        nb=json.loads((OUT/meta["code_file"]).read_text());ns={}
        for c in nb["cells"]:
            if c["cell_type"]=="code":
                self.assertIsNone(c["execution_count"]);self.assertEqual(c["outputs"],[])
                compile("".join(c["source"]),"notebook","exec")
        exec("".join(nb["cells"][1]["source"]),ns)
        for name,source in ns["SOURCES"].items():
            self.assertEqual(source,(ROOT/"scripts"/name).read_text())
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),ns["SOURCE_HASHES"][name])
            ast.parse(source)
        cfg=ns["CONFIG"]
        contract=json.loads((OUT/"dataset-contract.json").read_text())
        self.assertEqual(ns["DATASET_CONTRACT"],contract)
        self.assertEqual(hashlib.sha256((OUT/"dataset-contract.json").read_bytes()).hexdigest(),cfg["dataset_contract_sha256"])
        self.assertEqual(cfg["approved_dataset_attachment"],contract["attachment"])
        self.assertEqual(contract["attachment"]["ref"],meta["dataset_sources"][0])
        self.assertEqual([len(contract["splits"][n]) for n in ("validated","train","validation","test","prompt")],[714,571,71,72,500])
        self.assertEqual(cfg["max_images"],5);self.assertEqual(cfg["max_generations"],6)
        self.assertEqual(cfg["repeat_image_id"],cfg["image_ids"][0])
        self.assertEqual(cfg["hard_timeout_seconds"],1800);self.assertEqual(cfg["max_tokens"],160)
        self.assertFalse(cfg["training_enabled"]);self.assertFalse(cfg["test_inference_enabled"])
        old=json.loads((ROOT/"notebooks/gate4/single_image/configuration.json").read_text())
        for name in ("dependency_hashes","selected_hashes","zip_sha256","support_lock","prompt_ids_sha256"):
            self.assertEqual(cfg[name],old[name])
        for name in ("zeromrg_gate4b_runtime.py","gate4b_diagnostic_worker.py"):
            tree=ast.parse(ns["SOURCES"][name])
            self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
                n.func.attr in {"backward","step","zero_grad"} for n in ast.walk(tree)))
        prep=json.loads((ROOT/"docs/evidence/gate4_preparation.json").read_text())
        for name,item in prep["dirty_source_identity"].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),item["sha256"])

    def supervisor_fixture(self,timeout):
        from build_gate4b_notebook import SUPERVISOR_4B
        source=SUPERVISOR_4B.replace("Path('/kaggle/working')","Path(TEST_ROOT)")
        ns=dict(Path=Path,uuid=uuid,hashlib=hashlib,json=json,sys=sys,SOURCES={},SOURCE_HASHES={},DATASET_CONTRACT={},
                CONFIG=dict(hard_timeout_seconds=0 if timeout else 1800,max_images=5,max_generations=6,
                            dataset_contract_sha256=hashlib.sha256((json.dumps({},indent=2)+'\n').encode()).hexdigest()))
        class Process:
            def __init__(self,command,**kwargs):
                self.pid=99999999;self.returncode=None if timeout else 0
                self.stdout=io.StringIO("fixture progress\n")
                dest=Path(command[command.index("--run-root")+1])
                (dest/"diagnostic.json").write_text(json.dumps(dict(image_count=5,generation_count=6,
                    split="checkpoint_train",test_references_accessed=False,optimizer_created=False)))
            def poll(self):return self.returncode
            def wait(self,timeout=None):self.returncode=-9;return self.returncode
        with tempfile.TemporaryDirectory() as folder:
            ns["TEST_ROOT"]=folder
            with patch.object(Path,"is_dir",return_value=True),patch("subprocess.Popen",side_effect=Process) as launch,\
                 patch("os.killpg",create=True) as kill,patch("signal.SIGKILL",9,create=True),\
                 patch("time.monotonic",side_effect=range(1000)),patch("builtins.print"):
                if timeout:
                    with self.assertRaises(TimeoutError):exec(source,ns)
                    self.assertEqual(kill.call_count,1)
                    self.assertEqual(json.loads((ns["RUN_ROOT"]/"supervisor_failure.json").read_text())["status"],"HARD_TIMEOUT")
                else:
                    exec(source,ns);self.assertEqual(kill.call_count,0)
                self.assertEqual(launch.call_count,1)
                with self.assertRaises(ValueError):exec(source,ns)
                self.assertEqual(launch.call_count,1)

    def test_supervisor_success_one_attempt(self):self.supervisor_fixture(False)
    def test_supervisor_timeout_no_retry(self):self.supervisor_fixture(True)


if __name__=="__main__":unittest.main(verbosity=2)
