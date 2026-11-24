"""Verify included frozen sources and inputs without training."""
from pathlib import Path
import argparse,sys,json
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from reproduction_common import read,sha,catalog,graph

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--load',action='store_true')
    p.add_argument('--require-all-inputs',action='store_true')
    a=p.parse_args()
    manifest=read(ROOT/'checksums.json')
    for name,digest in manifest.items():
        f=(ROOT/name).resolve()
        if not f.is_relative_to(ROOT) or sha(f)!=digest:raise ValueError('Changed package file: '+name)
    items=catalog();available=[];missing=[]
    assert len(items)==360 and len({d['id'] for d in items})==360
    for d in items:
        absent=[n for n in d['sha256'] if not (ROOT/d['path']/n).is_file()]
        if absent:missing.append(d['id']);continue
        if a.load:graph(d)
        available.append(d['id'])
    if a.require_all_inputs and missing:raise SystemExit('Acquire processed inputs for: '+', '.join(missing))
    if a.load:
        sys.path.insert(0,str(ROOT/'src/model'))
        from bootstrap import activate
        activate()
    print(json.dumps(dict(status='passed',package_files=len(manifest),available_inputs=len(available),separately_acquired_inputs=missing,inputs_loaded=a.load,model_imported=a.load,training=False),indent=2))
if __name__=='__main__':main()
