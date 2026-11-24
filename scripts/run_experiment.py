"""List or execute one preserved experimental task; never auto-submit jobs."""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
GROUPS={'ablation':('ablation_a','tasks_A.json'),
        'sensitivity':('sensitivity_b','tasks_B.json'),
        'shared-template':('shared_template','tasks_A.json'),
        'node-measure':('node_measure','tasks.json')}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',choices=tuple(GROUPS),required=True)
    p.add_argument('--list',action='store_true')
    p.add_argument('--index',type=int)
    p.add_argument('--output',type=Path)
    a=p.parse_args();folder,config=GROUPS[a.group];base=ROOT/'experiments'/folder
    tasks=json.loads((base/'config'/config).read_text(encoding='utf-8-sig')) if a.group!='shared-template' else json.loads((base/'protocol.json').read_text(encoding='utf-8-sig'))['tasks']
    if a.group=='node-measure':tasks=tasks[::2]
    if a.list:
        for i,t in enumerate(tasks):print(i,json.dumps(t,ensure_ascii=False))
        return
    if a.index is None or not 0<=a.index<len(tasks):p.error('Choose a valid --index; use --list first.')
    if a.group in ('shared-template','node-measure') and a.output is not None:p.error('This native package uses its own results/ directory; --output is unsupported.')
    if a.group=='shared-template':
        cmd=[sys.executable,'code/school.py','pool','--index',str(a.index)]
    elif a.group=='node-measure':
        # The native pair operation trains both measures for the same initialization.
        cmd=[sys.executable,'run.py','pair','--index',str(a.index)]
    else:
        out=a.output.resolve() if a.output else ROOT/'runs'/a.group
        cmd=[sys.executable,'code/run.py','task','--index',str(a.index),'--output',str(out)]
    subprocess.run(cmd,cwd=base,check=True)
if __name__=='__main__':main()
