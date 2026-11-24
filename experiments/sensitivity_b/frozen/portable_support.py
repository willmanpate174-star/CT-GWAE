from pathlib import Path
import hashlib,json,os

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def state_hash(value):
    def canonical(v):
        if hasattr(v,'detach'):
            t=v.detach().cpu().contiguous()
            return dict(dtype=str(t.dtype),shape=list(t.shape),bytes=t.numpy().tobytes().hex())
        if isinstance(v,dict):return {str(k):canonical(w) for k,w in sorted(v.items(),key=lambda x:str(x[0]))}
        if isinstance(v,(list,tuple)):return [canonical(w) for w in v]
        return v
    return hashlib.sha256(json.dumps(canonical(value),sort_keys=True,allow_nan=False).encode()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    assert read(path)==value