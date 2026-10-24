"""Generate explicit stage configs. Default only writes configs; --execute trains."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml
from cxr_project.config import load_config
from cxr_project.grid import build_configs, RECIPES, EVALUATION_RECIPES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--experiments',nargs='+',required=True,choices=sorted(RECIPES))
    p.add_argument('--seeds',nargs='+',type=int,default=[42,2027,3407])
    p.add_argument('--stage',default='grid')
    p.add_argument('--config-dir',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--project-root',type=Path,help='Root for relative baseline paths; defaults to baseline parent.parent')
    p.add_argument('--weights',action='append',default=[],metavar='B1=PATH')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--execute',action='store_true',help='Explicitly start generated training jobs sequentially')
    args=p.parse_args()
    baseline=load_config(args.baseline)
    root=args.project_root.resolve() if args.project_root else baseline['_project_root']
    weights={}
    for item in args.weights:
        if '=' not in item: p.error('--weights must be ID=PATH')
        key,value=item.split('=',1);weights[key]=value
    configs=build_configs(baseline,args.experiments,args.seeds,root,args.output_root,stage=args.stage,weights=weights)
    directory=args.config_dir.resolve()
    if directory.exists() and any(directory.iterdir()): raise FileExistsError(f'config directory is not empty: {directory}')
    for name,config in configs.items():
        out=Path(config['output_dir'])
        if out.exists(): raise FileExistsError(f'output directory already exists: {out}')
    directory.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for name,config in configs.items():
        path=directory/f'{name}.yaml';path.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
        load_config(path)
        command=[sys.executable,'-m','cxr_project.train','--config',str(path),'--device',args.device]
        jobs.append({'name':name,'config':str(path),'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'command':command})
    record={'stage':args.stage,'baseline':str(args.baseline.resolve()),'baseline_sha256':hashlib.sha256(args.baseline.read_bytes()).hexdigest(),'jobs':jobs,'evaluation_commands':EVALUATION_RECIPES,'execution_requested':args.execute}
    (directory/'grid_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False,indent=2))
    if args.execute:
        for job in jobs: subprocess.run(job['command'],cwd=root,check=True)

if __name__=='__main__': main()
