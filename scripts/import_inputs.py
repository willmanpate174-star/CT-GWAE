"""Import an author's exact processed graph inputs after checking their hashes."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True,choices=['dkpol','mnist70000']);p.add_argument('--from-folder',required=True,type=Path);a=p.parse_args()
 items=json.loads((ROOT/'configs/catalog.json').read_text(encoding='utf-8'))['datasets'];item=next(d for d in items if d['id']==a.dataset)
 source=a.from_folder.resolve();dest=(ROOT/item['path']).resolve()
 for name,digest in item['sha256'].items():
  path=source/name
  if not path.is_file() or sha(path)!=digest:raise SystemExit('Input does not match the frozen experiment: '+name)
 for name in item['sha256']:
  dest.mkdir(parents=True,exist_ok=True)
  if (dest/name).exists() and sha(dest/name)!=item['sha256'][name]:raise SystemExit('Refusing to overwrite a different existing input: '+str(dest/name))
  if source/name!=dest/name:shutil.copy2(source/name,dest/name)
 if a.dataset=='dkpol':
  for data_dir in (ROOT/'experiments').rglob('data'):
   target=data_dir/'dkpol'
   if target.is_dir():
    for f in source.iterdir():
     if f.is_file() and f.suffix in ['.npz','.npy']:shutil.copy2(f,target/f.name)
 print('Imported matching processed inputs: '+a.dataset)
if __name__=='__main__':main()
