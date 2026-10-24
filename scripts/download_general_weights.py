"""Download an explicitly selected general timm weight and record immutable provenance."""
import argparse
import json
from pathlib import Path
import shutil
from cxr_project.general_weights import SOURCES, weight_source
from cxr_project.evaluation_tools import sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=sorted(SOURCES))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--revision',default='main',help='Resolved to an immutable HF commit before downloading')
    args=p.parse_args()
    source=weight_source(args.model)
    output=args.output.resolve();metadata=output.with_suffix(output.suffix+'.source.json')
    if output.exists() or metadata.exists(): raise FileExistsError(output)
    if output.suffix!='.safetensors': raise ValueError('output must end in .safetensors')
    from huggingface_hub import HfApi, hf_hub_download
    info=HfApi().model_info(source['repo_id'],revision=args.revision,files_metadata=True)
    cached=Path(hf_hub_download(source['repo_id'],source['filename'],revision=info.sha))
    digest=sha256(cached)
    file_info=next((x for x in info.siblings if x.rfilename==source['filename']),None)
    lfs=getattr(file_info,'lfs',None)
    expected=getattr(lfs,'sha256',None) if lfs is not None else None
    if expected and expected!=digest: raise ValueError('Hugging Face LFS SHA256 mismatch')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.part')
    shutil.copyfile(cached,temporary);temporary.replace(output)
    report={**source,'model':args.model,'requested_revision':args.revision,'revision':info.sha,'sha256':digest,'bytes':output.stat().st_size,'path':str(output),'url':f'https://huggingface.co/{source["repo_id"]}/blob/{info.sha}/{source["filename"]}'}
    metadata.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
