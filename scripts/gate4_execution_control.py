"""Scoped Gate4 read-only execution checks/retrieval. Never submits or retries a job."""
import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'notebooks/gate4/single_image'
TARGET='inshasadaf114/zeromrg-covctr-gate4-single-image-diagnostic'
BASELINE='inshasadaf114/zeromrg-covctr-gate3-recovery-baseline'

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def write(directory,name,value):
    with (directory/name).open('x',encoding='utf-8') as stream: json.dump(value,stream,indent=2)

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def cells(book): return [(c['cell_type'],''.join(c['source'])) for c in book['cells']]

def api_client():
    os.environ['KAGGLE_CONFIG_DIR']=str(ROOT/'.kaggle')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();return api

def get_kernel(api,target):
    from kaggle.api.kaggle_api_extended import ApiGetKernelRequest
    with api.build_kaggle_client() as client:
        request=ApiGetKernelRequest();request.user_name,request.kernel_slug=target.split('/')
        return client.kernels.kernels_api_client.get_kernel(request)

def list_output(api,target):
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
    with api.build_kaggle_client() as client:
        request=ApiListKernelSessionOutputRequest();request.user_name,request.kernel_slug=target.split('/')
        result=client.kernels.kernels_api_client.list_kernel_session_output(request)
        assert not result.next_page_token,'Bounded output page incomplete'
        return result

def parse_dataset_metadata(value):
    """Accept SDK JSON text or dictionaries; fail closed on identity/schema drift."""
    if isinstance(value,str):
        try: value=json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError('Dataset metadata is malformed JSON') from error
    if not isinstance(value,dict):
        raise ValueError('Dataset metadata must be a dictionary or JSON object string')
    if 'info' in value:
        error=value.get('errorMessage','')
        if not isinstance(error,str) or error:
            raise ValueError('Dataset metadata envelope contains an invalid/error response')
        value=value['info']
        if not isinstance(value,dict):
            raise ValueError('Dataset metadata info must be a dictionary')
    expected={'datasetId':int,'datasetSlug':str,'ownerUser':str,'title':str,'isPrivate':bool}
    for key,kind in expected.items():
        if key not in value: raise ValueError('Dataset metadata missing required key: '+key)
        if type(value[key]) is not kind:
            raise ValueError('Dataset metadata field '+key+' must be '+kind.__name__)
        if kind is str and not value[key].strip():
            raise ValueError('Dataset metadata field '+key+' must be nonempty')
    if value['datasetId']<=0: raise ValueError('Dataset metadata datasetId must be positive')
    return value

def precheck(directory):
    prep=read(ROOT/'docs/evidence/gate4_preparation.json')
    for name,item in prep['dirty_source_identity'].items(): assert digest(ROOT/name)==item['sha256'],name+' changed'
    meta=read(FOLDER/'kernel-metadata.json');book=read(FOLDER/meta['code_file'])
    assert meta['id']==TARGET and meta['is_private'] is True and meta['enable_gpu'] is True and not meta['enable_tpu']
    assert meta['dataset_sources']==['inshasadaf114/dataset','inshasadaf114/zeromrg-checkpoint-backup5-zip']
    assert meta['kernel_sources']==[BASELINE]
    for c in book['cells']:
        if c['cell_type']=='code':
            assert c['execution_count'] is None and c['outputs']==[]
            compile(''.join(c['source']),'notebook','exec')
    tree=ast.parse(''.join(book['cells'][1]['source']))
    literals={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)}
    sources=literals['SOURCES'];config=literals['CONFIG']
    assert config==read(FOLDER/'configuration.json')
    for name,text in sources.items(): assert text==(ROOT/'scripts'/name).read_text(encoding='utf-8')
    assert config['image_id']=='PIIS0140673620301549_0%0.png' and config['max_images']==1
    assert not config['training_enabled'] and not config['test_inference_enabled'] and config['hard_timeout_seconds']==1800
    for name in ('zeromrg_gate4_runtime.py','gate4_diagnostic_worker.py'):
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in
                       {'backward','step','zero_grad'} for n in ast.walk(ast.parse(sources[name])))
    assert all('test_manifest' not in f and 'validation' not in f for f in config['selected_hashes'])
    for name in (meta['code_file'],'kernel-metadata.json','configuration.json'): shutil.copy2(FOLDER/name,directory/name)
    api=api_client()
    owned=[]
    for page in range(1,6):
        rows=api.kernels_list(page=page,page_size=100,mine=True);owned.extend(row.ref for row in rows)
        if len(rows)<100:break
    else:raise ValueError('Owned notebook catalog exceeded bounded500 entries')
    assert TARGET not in owned,'Target already exists; do not submit another version'
    assert BASELINE in owned,'Verified private baseline not in owned catalog'
    from kaggle.api.kaggle_api_extended import ApiGetDatasetMetadataRequest
    datasets=[]
    with api.build_kaggle_client() as client:
        for target in meta['dataset_sources']:
            req=ApiGetDatasetMetadataRequest();req.owner_slug,req.dataset_slug=target.split('/')
            response=client.datasets.dataset_api_client.get_dataset_metadata(req)
            assert not response.error_message,'Dataset metadata inaccessible: '+target
            if response.info is None: raise ValueError('Dataset metadata info is absent: '+target)
            serialized=response.to_json(response.info,ignore_defaults=False)
            info=parse_dataset_metadata(serialized)
            assert info['datasetSlug']==req.dataset_slug and info['ownerUser'].lower()==req.owner_slug.lower(), 'Dataset identity differs: '+target
            datasets.append(dict(target=target,accessible=True,metadata_keys=sorted(info),
                                 serialized_type=type(serialized).__name__,dataset_id=info['datasetId'],
                                 slug=info['datasetSlug'],owner=info['ownerUser'],privacy=info['isPrivate']))
    baseline=get_kernel(api,BASELINE);server=baseline.metadata
    assert server.is_private is True and server.current_version_number==1,'Baseline privacy/version drift'
    assert cells(json.loads(baseline.blob.source))==cells(read(ROOT/'notebooks/gate3/ZeroMRG_COV_CTR_Recovery_Kaggle.ipynb')),'Baseline source changed'
    output=list_output(api,BASELINE);names=[f.file_name for f in output.files]
    assert sum(Path(f).name==config['zip_name'] for f in names)==1,'Verified baseline ZIP unavailable'
    result=dict(status='PASS_PRE_SUBMISSION',observed_utc=datetime.now(timezone.utc).isoformat(),target=TARGET,
        target_absent=True,planned_private=True,datasets=datasets,baseline=dict(id=server.id,version=1,private=True,
        source_cells_match=True,zip_listed=True),config=config,notebook_sha256=digest(FOLDER/meta['code_file']),
        metadata_sha256=digest(FOLDER/'kernel-metadata.json'),source_checks='All prepared source hashes and15 existing CPU checks reused',
        checks_passed=list(range(1,13)),gpu_memory_telemetry='NOT_INSTRUMENTED; report unavailable if absent in logs',
        quota='No authenticated CLI quota endpoint; server submission/allocation and worker CUDA guards are prospective checks',
        submitted=False,no_gate3_changes=True)
    write(directory,'precheck.json',result);print(json.dumps(result,indent=2))

def identity(directory):
    api=api_client();response=get_kernel(api,TARGET);s=response.metadata
    book=read(FOLDER/'ZeroMRG_COV_CTR_Gate4_Single_Image_Diagnostic.ipynb')
    match=cells(json.loads(response.blob.source))==cells(book)
    write(directory,'server_source.ipynb',json.loads(response.blob.source))
    result=dict(observed_utc=datetime.now(timezone.utc).isoformat(),target=TARGET,id=s.id,
        version=s.current_version_number,private=s.is_private,enable_gpu=s.enable_gpu,
        datasets=s.dataset_data_sources,kernel_sources=s.kernel_data_sources,source_cells_match=match,
        server_source_sha256=digest(directory/'server_source.ipynb'))
    write(directory,'identity.json',result)
    assert s.current_version_number==1 and s.is_private is True and s.enable_gpu is True and match,'Server identity mismatch'
    assert list(s.dataset_data_sources)==read(FOLDER/'kernel-metadata.json')['dataset_sources'],'Server dataset attachments differ'
    assert list(s.kernel_data_sources)==[BASELINE],'Server baseline output attachment differs'
    print(json.dumps(result,indent=2))

def status(directory):
    api=api_client();r=api.kernels_status(TARGET)
    result=dict(observed_utc=datetime.now(timezone.utc).isoformat(),target=TARGET,status=str(r.status),failure_message=r.failure_message)
    write(directory,'status_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json',result)
    print(json.dumps(result))

def retrieve(directory):
    import requests
    api=api_client();out=list_output(api,TARGET)
    result=dict(observed_utc=datetime.now(timezone.utc).isoformat(),target=TARGET,listed=[f.file_name for f in out.files],downloads=[])
    if out.log:
        (directory/'platform_execution.log').write_text(out.log,encoding='utf-8')
        result['platform_log_sha256']=digest(directory/'platform_execution.log')
    allowed={'diagnostic.json','failure.json','supervisor_failure.json','execution.log','configuration.json',
             'zeromrg_gate1_runtime.py','zeromrg_gate4_runtime.py','gate4_diagnostic_worker.py','support.lock'}
    selected=[f for f in out.files if Path(f.file_name).name in allowed]
    assert len(set(f.file_name for f in selected))==len(selected),'Duplicate outputs'
    count=0
    for entry in selected:
        destination=(directory/'retrieved'/entry.file_name).resolve()
        assert destination.is_relative_to((directory/'retrieved').resolve()) and not destination.exists(),'Unsafe/existing retrieval'
        destination.parent.mkdir(parents=True,exist_ok=True)
        # Signed URLs are used only in memory; never included in evidence/error messages.
        try:
            with requests.get(entry.url,stream=True,timeout=(20,45)) as response:
                assert response.status_code==200,'HTTP'+str(response.status_code)
                length=int(response.headers.get('content-length','0'))
                assert count+length<=20*2**20,'Small-output20MiB bound exceeded'
                with destination.open('xb') as stream:
                    for block in response.iter_content(65536):
                        if not block:continue
                        count+=len(block);assert count<=20*2**20,'Retrieval bound exceeded';stream.write(block)
        except Exception as error:
            raise RuntimeError('Small output retrieval failed: '+destination.name+'; '+type(error).__name__) from None
        if length:assert destination.stat().st_size==length,'Truncated output'
        result['downloads'].append(dict(file=entry.file_name,bytes=destination.stat().st_size,sha256=digest(destination)))
    write(directory,'retrieval.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['precheck','identity','status','retrieve']);parser.add_argument('directory')
    args=parser.parse_args();directory=Path(args.directory)
    try:globals()[args.mode](directory)
    except Exception as error:
        failure=dict(mode=args.mode,observed_utc=datetime.now(timezone.utc).isoformat(),error_type=type(error).__name__,error=str(error))
        write(directory,args.mode+'_failure_'+datetime.now(timezone.utc).strftime('%H%M%S%f')+'.json',failure)
        print(json.dumps(failure));sys.exit(1)
