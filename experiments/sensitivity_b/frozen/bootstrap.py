"""Package-relative loader. Frozen numerical files remain byte-identical on disk."""
from pathlib import Path
import hashlib, importlib.util, json, sys, types
BASE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def exact_sources():
    lineage=json.loads((BASE/'lineage.json').read_text(encoding='utf-8-sig'))
    for relative,info in lineage.items():
        if sha(BASE/relative)!=info['sha256']:raise RuntimeError('Changed algorithm source '+relative)
    return lineage

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def body_rebind(path,name,anchor,header):
    text=Path(path).read_text(encoding='utf-8-sig');pos=text.index(anchor)
    m=types.ModuleType(name);m.__file__=str(Path(path).resolve());sys.modules[name]=m
    exec(compile(header+text[pos:],str(path),'exec'),m.__dict__)
    return m

def activate():
    exact_sources();sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'encoder_leaf'))
    import torch,numpy as np
    import experiment as encoder
    import portable_support as support
    backend=types.ModuleType('portable_original_backend');backend.E=encoder
    # Exact state functions from production backend are retained separately.
    helpers=json.loads((BASE/'backend_state_functions.json').read_text(encoding='utf-8-sig'))
    backend.__dict__.update(torch=torch,np=np,hashlib=hashlib)
    exec(compile(helpers['source'],'portable_backend_state','exec'),backend.__dict__)
    sys.modules['portable_original_backend']=backend
    adapter=body_rebind(BASE/'geometric_argmax_adapter.py','portable_geometric_adapter','Model=Hard=None',
        'from pathlib import Path\nimport hashlib,importlib.util,sys,time\nimport numpy as np\nimport torch\nimport portable_original_backend as OLD\nE=OLD.E\nstate,state_digest=OLD.state,OLD.state_digest\n')
    adapter.configure(dict(layerwise_model_path=str(BASE/'layer_geometric_single_gw.py'),layerwise_model_sha256=sha(BASE/'layer_geometric_single_gw.py'),layerwise_model_class='LayerGeometricSingleGW',hard_score_path=str(BASE/'layer_geometric_hard_score.py'),hard_score_sha256=sha(BASE/'layer_geometric_hard_score.py')))
    training=body_rebind(BASE/'budget600_training_STATIC.py','portable_budget600_training','def verify_manifest(manifest):',
        'from pathlib import Path\nimport importlib.util,sys\nimport portable_support as E\nsha,read,save,state_hash=E.sha,E.read,E.save,E.state_hash\nBASE=Path(__file__).resolve().parent\nREG=BASE/"training_config.json"\nOUTPUT=BASE\n')
    guard=load(BASE/'budget600_raw_guard_STATIC.py','portable_budget600_guard')
    wrapper=load(BASE/'s100_knownK_MAP_wrapper_STATIC_v2.py','portable_final_knownK_wrapper')
    decoder=load(BASE/'surjective_map_decoder.py','portable_exact_MAP')
    from recovery_step import DirectionSchedule
    def Schedule(**kwargs):return DirectionSchedule(**dict(dict(budget=600),**kwargs))
    cfg=json.loads((BASE/'training_config.json').read_text(encoding='utf-8-sig'))
    return adapter,training,guard,Schedule,wrapper,decoder,torch,np,cfg,support
