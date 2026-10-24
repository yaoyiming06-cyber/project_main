"""Offline validation, grouped folds, fixed fusion, and final/export preparation."""
import argparse
import json
from pathlib import Path
import yaml
from cxr_project import evaluation_tools as ev

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('folds');p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--folds',type=int,default=3);p.add_argument('--seed',type=int,default=42)
    p=sub.add_parser('bootstrap');p.add_argument('--first',type=Path,required=True);p.add_argument('--second',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--iterations',type=int,default=2000);p.add_argument('--seed',type=int,default=42)
    p=sub.add_parser('fuse');p.add_argument('--predictions',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--labels',nargs='+',type=int,help='Only fuse selected labels; others retain the first prediction')
    p=sub.add_parser('route');p.add_argument('--base',type=Path,required=True);p.add_argument('--specialist',action='append',required=True,metavar='AP=FILE');p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--labels',nargs='+',type=int,help='Only replace these labels; other labels keep base predictions')
    p=sub.add_parser('compare');p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p=sub.add_parser('export-fp16');p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--weights',choices=['auto','raw','ema'],default='auto')
    p=sub.add_parser('final-manifest');p.add_argument('--train',type=Path,required=True);p.add_argument('--valid',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--final',action='store_true');p.add_argument('--baseline',type=Path,required=True);p.add_argument('--epochs',type=int,required=True);p.add_argument('--config-output',type=Path,required=True);p.add_argument('--run-output',type=Path,required=True)
    for p in sub.choices.values(): p.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.report and args.report.exists(): raise FileExistsError(args.report)
    if args.command=='folds': result=ev.make_subject_folds(args.manifest,args.output_dir,args.folds,args.seed)
    elif args.command=='bootstrap': result=ev.paired_bootstrap(args.first,args.second,args.manifest,iterations=args.iterations,seed=args.seed)
    elif args.command=='fuse': result=ev.fuse_predictions(args.predictions,args.output,label_indices=args.labels)
    elif args.command=='route':
        specialists={}
        for item in args.specialist:
            view,path=item.split('=',1)
            if view in specialists: parser.error('duplicate specialist view')
            specialists[view]=path
        result=ev.route_predictions(args.base,specialists,args.manifest,args.output,label_indices=args.labels)
    elif args.command=='compare': result=ev.compare_predictions(args.reference,args.candidate)
    elif args.command=='export-fp16': result=ev.export_fp16(args.checkpoint,args.output,weights=args.weights)
    else:
        from cxr_project.config import load_config, resolve_path
        if args.config_output.exists() or args.run_output.exists(): raise FileExistsError('final config/run path already exists')
        base=load_config(args.baseline)
        if args.epochs<=int(base['warmup_epochs']): raise ValueError('fixed final epochs must exceed warmup')
        config={k:v for k,v in base.items() if not k.startswith('_')}
        for key in ('image_root','pretrained_checkpoint','test_manifest'): config[key]=str(resolve_path(base,key))
        config.update(project_root=str(base['_project_root']),final_training=True,train_manifest=str(args.output.resolve()),valid_manifest=None,epochs=args.epochs,early_stopping_patience=0,run_name=args.run_output.name,output_dir=str(args.run_output.resolve()))
        result=ev.merge_final_manifests(args.train,args.valid,args.output,final=args.final)
        args.config_output.parent.mkdir(parents=True,exist_ok=True)
        args.config_output.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
        result['config']=str(args.config_output.resolve())
    payload=json.dumps(result,ensure_ascii=False,indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(payload,encoding='utf-8')
    print(payload)

if __name__=='__main__': main()
