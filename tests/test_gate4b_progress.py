"""CPU mocks and stdlib progress/export fixtures; no weights, GPU or references."""
import ast,hashlib,io,json,sys,tempfile,unittest,uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'))
from zeromrg_gate4b_progress import StageProgress,run_stage
from zeromrg_gate4b_runtime import diagnose_images
from test_gate4_inference import Image,MockDecoder,Vision
from build_gate4b_notebook import SUPERVISOR_4B,build,OUT


class ProgressTests(unittest.TestCase):
    def assert_timestamp_record(self,record):
        stamp=datetime.fromisoformat(record['timestamp_utc'])
        self.assertIsNotNone(stamp.tzinfo)
        self.assertEqual(stamp.utcoffset(),timedelta(0))
        self.assertGreaterEqual(record['elapsed_seconds'],0)
        self.assertTrue(record['phase']);self.assertTrue(record['state'])
        return stamp

    def test_journal_latest_success_and_safe_failure(self):
        with tempfile.TemporaryDirectory() as folder,patch('builtins.print'):
            counter=iter(range(100));utc_counter=iter(range(100))
            origin=datetime(2026,10,12,tzinfo=timezone.utc)
            observer=StageProgress(folder,clock=lambda:next(counter),utc_clock=lambda:origin+timedelta(seconds=next(utc_counter)))
            observer.emit('worker_start')
            self.assertEqual(run_stage(observer,'fixture',lambda:7),7)
            with self.assertRaises(RuntimeError):
                with observer.stage('network_fixture'):raise RuntimeError('secret_URL_must_not_be_recorded')
            lines=(Path(folder)/'stage_progress.jsonl').read_text().splitlines()
            records=[json.loads(line) for line in lines]
            self.assertEqual([r['sequence'] for r in records],list(range(1,6)))
            self.assertEqual([r['state'] for r in records],['event','start','complete','start','failed'])
            self.assertEqual(json.loads((Path(folder)/'latest_progress.json').read_text()),records[-1])
            self.assertEqual(records[-1]['error_type'],'RuntimeError')
            self.assertNotIn('secret_URL','\n'.join(lines))
            stamps=[self.assert_timestamp_record(r) for r in records]
            self.assertTrue(all(a<b for a,b in zip(stamps,stamps[1:])))
            elapsed=[r['elapsed_seconds'] for r in records]
            self.assertTrue(all(a<b for a,b in zip(elapsed,elapsed[1:])))
            self.assertTrue(all(r['duration_seconds']>=0 for r in records if 'duration_seconds' in r))

    def test_default_stage_timestamp_is_aware_UTC(self):
        with tempfile.TemporaryDirectory() as folder,patch('builtins.print'):
            before=datetime.now(timezone.utc)
            record=StageProgress(folder).emit('fixture')
            stamp=self.assert_timestamp_record(record)
            self.assertLessEqual(before,stamp);self.assertLessEqual(stamp,datetime.now(timezone.utc))

    def test_notebook_syntax_and_worker_failure_timestamp(self):
        book=json.loads((OUT/'ZeroMRG_COV_CTR_Gate4B_Image_Conditioning_Diagnostic.ipynb').read_text())
        for cell in book['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'notebook','exec')
        source=(ROOT/'scripts/gate4b_diagnostic_worker.py').read_text()
        self.assertIn('timestamp_utc=utc_now().isoformat()',source)

    def test_operation_without_observer_is_unchanged(self):
        self.assertEqual(run_stage(None,'unused',lambda a,b:a+b,3,b=4),7)

    def test_worker_progress_precedes_heavy_imports(self):
        source=(ROOT/'scripts/gate4b_diagnostic_worker.py').read_text()
        self.assertLess(source.index('PROGRESS.emit("worker_start")'),source.index('import gate4_diagnostic_worker as base'))
        self.assertIn('PROGRESS.stage("runtime_imports")',source)
        self.assertIn('"cuda_availability",torch.cuda.is_available',source)
        progress=(ROOT/'scripts/zeromrg_gate4b_progress.py').read_text()
        self.assertNotIn('import torch',progress)
        compile(source,'worker','exec')

    def test_real_notebook_contract_export_exact_LF_bytes(self):
        meta=json.loads((OUT/'kernel-metadata.json').read_text())
        book=json.loads((OUT/meta['code_file']).read_text())
        tree=ast.parse(''.join(book['cells'][1]['source']))
        values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)}
        supervisor=''.join(book['cells'][2]['source']);tree=ast.parse(supervisor)
        node=next(n for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                  and 'dataset-contract.json' in ast.get_source_segment(supervisor,n))
        with tempfile.TemporaryDirectory() as folder:
            exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_contract_export','exec'),
                 dict(RUN_ROOT=Path(folder),json=json,DATASET_CONTRACT=values['DATASET_CONTRACT']))
            content=(Path(folder)/'dataset-contract.json').read_bytes()
            self.assertNotIn(b'\r\n',content)
            self.assertEqual(hashlib.sha256(content).hexdigest(),values['CONFIG']['dataset_contract_sha256'])
            self.assertEqual(json.loads(content),values['DATASET_CONTRACT'])

    def supervise(self,timeout):
        source=SUPERVISOR_4B.replace("Path('/kaggle/working')","Path(TEST_ROOT)")
        ns=dict(Path=Path,uuid=uuid,hashlib=hashlib,json=json,sys=sys,SOURCES={},SOURCE_HASHES={},DATASET_CONTRACT={},
                CONFIG=dict(hard_timeout_seconds=0 if timeout else 1800,max_images=5,max_generations=6,
                    dataset_contract_sha256=hashlib.sha256(b'{}\n').hexdigest()))
        processes=[]
        class Process:
            def __init__(self,command,**kwargs):
                self.pid=99999999;self.returncode=None if timeout else 0;self.stdout=io.StringIO('fixture progress\n')
                self.destination=Path(command[command.index('--run-root')+1]);self.kwargs=kwargs;processes.append(self)
                (self.destination/'diagnostic.json').write_text(json.dumps(dict(image_count=5,generation_count=6,
                    split='checkpoint_train',test_references_accessed=False,optimizer_created=False)))
            def poll(self):return self.returncode
            def wait(self,timeout=None):self.returncode=-9;return self.returncode
        def kill(*args):
            failure=json.loads((processes[0].destination/'supervisor_failure.json').read_text())
            self.assertEqual(failure['status'],'HARD_TIMEOUT')
            self.assert_timestamp_record(failure)
            self.assertEqual(failure['state'],'failed');self.assertEqual(failure['error_type'],'TimeoutError')
        with tempfile.TemporaryDirectory() as folder:
            ns['TEST_ROOT']=folder
            with patch.object(Path,'is_dir',return_value=True),patch('subprocess.Popen',side_effect=Process) as spawn,\
                 patch('os.killpg',side_effect=kill,create=True) as terminate,patch('signal.SIGKILL',9,create=True),\
                 patch('time.monotonic',side_effect=range(0,3000,31)),patch('builtins.print'):
                if timeout:
                    with self.assertRaises(TimeoutError):exec(source,ns)
                    self.assertEqual(terminate.call_count,1)
                    failure=json.loads((ns['RUN_ROOT']/'supervisor_exception.json').read_text())
                    self.assert_timestamp_record(failure)
                    self.assertEqual(failure['error_type'],'TimeoutError')
                else:
                    exec(source,ns);self.assertEqual(terminate.call_count,0)
                    self.assertIn('supervisor_heartbeat',(ns['RUN_ROOT']/'execution.log').read_text())
                    self.assertTrue((ns['RUN_ROOT']/'supervisor_status.json').is_file())
                    heartbeat=json.loads((ns['RUN_ROOT']/'supervisor_status.json').read_text())
                    self.assert_timestamp_record(heartbeat)
                    self.assertEqual(heartbeat['state'],'running')
                    logged=[json.loads(line) for line in (ns['RUN_ROOT']/'execution.log').read_text().splitlines() if line.startswith('{')]
                    self.assertEqual(heartbeat,logged[-1])
                self.assertEqual(processes[0].kwargs['env']['PYTHONUNBUFFERED'],'1')
                self.assertTrue(processes[0].kwargs['start_new_session'])
                with self.assertRaises(ValueError):exec(source,ns)
                self.assertEqual(spawn.call_count,1)

    def test_quiet_child_heartbeat_persisted(self):self.supervise(False)
    def test_timeout_record_precedes_kill_no_retry(self):self.supervise(True)

    def test_timeout_exited_child_pipe_cleanup_handles_missing_group(self):
        source=SUPERVISOR_4B.replace("Path('/kaggle/working')","Path(TEST_ROOT)")
        ns=dict(Path=Path,uuid=uuid,hashlib=hashlib,json=json,sys=sys,SOURCES={},SOURCE_HASHES={},DATASET_CONTRACT={},
                CONFIG=dict(hard_timeout_seconds=0,max_images=5,max_generations=6,
                    dataset_contract_sha256=hashlib.sha256(b'{}\n').hexdigest()))
        class Process:
            pid=99999999;returncode=0;stdout=io.StringIO('')
            def poll(self):return 0
            def wait(self,timeout=None):raise AssertionError('Already exited')
        class Reader:
            def __init__(self,*args,**kwargs):pass
            def start(self):pass
            def is_alive(self):return True
        with tempfile.TemporaryDirectory() as folder:
            ns['TEST_ROOT']=folder
            with patch.object(Path,'is_dir',return_value=True),patch('subprocess.Popen',return_value=Process()),\
                 patch('threading.Thread',Reader),patch('os.killpg',side_effect=ProcessLookupError,create=True) as kill,\
                 patch('signal.SIGKILL',9,create=True),patch('time.monotonic',side_effect=range(10)),patch('builtins.print'):
                with self.assertRaises(TimeoutError):exec(source,ns)
                self.assertEqual(kill.call_count,1)
                self.assertEqual(json.loads((ns['RUN_ROOT']/'supervisor_failure.json').read_text())['status'],'HARD_TIMEOUT')

    def exercise_callback(self,fail_after=None):
        ids=['a','b','c','d','e'];records=[];opened=[];phases=[]
        @contextmanager
        def opener(identifier):opened.append(identifier);yield Image.new('L',(5,5))
        def callback(index,record):
            records.append(record)
            if index==fail_after:raise OSError('fixture output failure')
        kwargs=dict(vision=Vision(),preprocess=lambda _:torch.zeros(3,224,224),decoder=MockDecoder(),
                    prompts=torch.ones(500,768),vocabulary=['word']*368,image_ids=ids,train_ids=ids+['p'],
                    prompt_ids=['p'],open_image=opener,progress=phases.append,on_record=callback)
        if fail_after is None:
            result=diagnose_images(**kwargs)
            self.assertEqual(records,result['records']);self.assertEqual(len(records),6)
            self.assertEqual(len([p for p in phases if p.startswith('generation_completed_')]),6)
            self.assertEqual(opened,ids+[ids[0]])
        else:
            with self.assertRaises(OSError):diagnose_images(**kwargs)
            self.assertEqual(len(records),fail_after+1);self.assertEqual(len(opened),fail_after+1)

    def test_completed_generation_callback_same_outputs(self):self.exercise_callback()
    def test_partial_export_failure_stops_before_next_image(self):self.exercise_callback(2)

    def test_build_does_not_open_heldout_or_raw_report_members(self):
        original_open=Path.open
        def guarded(path,*args,**kwargs):
            self.assertFalse(any(n in path.name for n in ('test_manifest','validation_manifest','validated_manifest','reports_ZH_EN')))
            return original_open(path,*args,**kwargs)
        import zipfile
        original_zip_open=zipfile.ZipFile.open
        def zip_guard(archive,name,*args,**kwargs):
            filename=name.filename if isinstance(name,zipfile.ZipInfo) else str(name)
            self.assertNotIn('reports_ZH_EN',filename)
            return original_zip_open(archive,name,*args,**kwargs)
        with patch.object(Path,'open',guarded),patch.object(zipfile.ZipFile,'open',zip_guard),patch('builtins.print'):
            build()


if __name__=='__main__':unittest.main(verbosity=2)
