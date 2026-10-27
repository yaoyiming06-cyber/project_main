"""Study inference with explicit raw/EMA weights, fixed TTA and view coverage."""
from __future__ import annotations
import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from cxr_project.config import load_config, resolve_path
from cxr_project.dataset import StudyDataset, build_transform, collate_studies
from cxr_project.labels import LABEL_COLUMNS
from cxr_project.losses import combine_main_and_expert_logits
from cxr_project.model import build_model, file_sha256
from cxr_project.runtime import select_inference_state

INFERENCE_DEFAULTS = {
    'experts_enabled': True, 'study_pooling': 'attention', 'use_view_embedding': True,
    'hard_label_indices': [0,3,7], 'head_dropout': .2, 'expert_dropout': .2,
    'adaptformer': False, 'adapter_dim': 64, 'adapter_dropout': .1, 'adapter_scale': 1.,
    'normalization': 'imagenet1k', 'clahe': False, 'clahe_clip_limit': 2.,
    'clahe_grid_size': [8,8], 'view_filter': None,
}

def validate_checkpoint_config(config, checkpoint):
    saved = checkpoint.get('config')
    if not isinstance(saved, dict):
        raise ValueError('checkpoint must contain its training config for safe inference')
    mismatches = []
    for key in ('model', 'input_size', *INFERENCE_DEFAULTS):
        default = INFERENCE_DEFAULTS.get(key)
        current, previous = config.get(key, default), saved.get(key, default)
        if isinstance(current, (list,tuple)): current = list(current)
        if isinstance(previous, (list,tuple)): previous = list(previous)
        if current != previous: mismatches.append(f'{key}: config={current!r}, checkpoint={previous!r}')
    if mismatches:
        raise ValueError('inference configuration mismatch; use --tta-sizes for deliberate resizing: ' + '; '.join(mismatches))

def inference_weights(checkpoint, selection='auto'):
    if checkpoint.get('format') == 'cxr_fp16_inference_v1':
        origin = checkpoint.get('weight_source', checkpoint.get('selected_weights', 'raw'))
        if selection not in {'auto', origin}:
            raise ValueError(f'compact checkpoint contains only {origin} weights, cannot select {selection}')
        state, source = select_inference_state(checkpoint, 'auto')
    else:
        state, source = select_inference_state(checkpoint, selection)
    if not isinstance(state, dict) or not state:
        raise ValueError('selected model state is empty')
    return {key: value.float() if value.is_floating_point() else value for key,value in state.items()}, source

def predict_batch(model, batch, device, use_amp=False, horizontal_flip=False):
    images = batch['images'].to(device, non_blocking=True)
    view_ids = batch['view_ids'].to(device, non_blocking=True)
    probabilities = []
    for flipped in ([False,True] if horizontal_flip else [False]):
        context = torch.autocast('cuda', dtype=torch.float16) if use_amp and device.type == 'cuda' else nullcontext()
        with context:
            main, experts = model(images.flip(-1) if flipped else images, batch['offsets'], view_ids)
        probabilities.append(combine_main_and_expert_logits(main.float(), experts.float(), model.hard_label_indices).sigmoid())
    return torch.stack(probabilities).mean(0).cpu()

def set_inference_size(model, config, size, source_state=None):
    if str(config['model']).startswith('swin_'):
        if source_state is not None:
            native = int(config['input_size'])
            model.backbone.set_input_size(img_size=(native,native), window_size=(7,7))
            model.load_state_dict(source_state, strict=True)
        # Keep the pretrained 7x7 windows; default set_input_size would change window size.
        model.backbone.set_input_size(img_size=(size,size), window_size=(7,7))
    elif str(config['model']).startswith('vit_') and size % 16:
        raise ValueError('ViT inference sizes must be multiples of 16')

def predict_dataset(model, dataset, config, device, horizontal_flip=False):
    workers = int(config['num_workers'])
    loader = DataLoader(dataset, batch_size=int(config['batch_size_studies']), shuffle=False,
                        num_workers=workers, pin_memory=bool(config['pin_memory'] and device.type=='cuda'),
                        collate_fn=collate_studies, persistent_workers=workers>0,
                        **({'prefetch_factor':2} if workers>0 else {}))
    ids, predictions, targets = [], [], []
    with torch.inference_mode():
        for index,batch in enumerate(loader,1):
            ids.extend(batch['study_ids'])
            predictions.append(predict_batch(model,batch,device,bool(config['amp']),horizontal_flip))
            if 'targets' in batch: targets.append(batch['targets'])
            print(f'batch {index}/{len(loader)}',flush=True)
    return ids, torch.cat(predictions), torch.cat(targets) if targets else None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','checkpoint','data-root','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    parser.add_argument('--weights', choices=('auto','raw','ema'), default='auto')
    parser.add_argument('--manifest', type=Path, help='Override test manifest, e.g. validation manifest')
    parser.add_argument('--labeled', action='store_true', help='Export target_i and probability_i columns')
    parser.add_argument('--tta', choices=('none','horizontal-flip'), default='none')
    parser.add_argument('--tta-sizes', type=int, nargs='+', help='Fixed inference sizes, equal probability average')
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint_path, output = args.checkpoint.resolve(), args.output.resolve()
    sidecar = output.with_suffix('.json')
    if output.exists() or sidecar.exists(): raise FileExistsError(f'output or metadata already exists: {output}')
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    validate_checkpoint_config(config,checkpoint)
    state, weight_source = inference_weights(checkpoint,args.weights)
    model = build_model(config,pretrained=False).float()
    model.load_state_dict(state,strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    normalization = checkpoint.get('normalization_stats') or checkpoint.get('normalization')
    if str(config['normalization']).lower()=='dataset' and not isinstance(normalization,dict):
        stats_path = checkpoint_path.parent/'dataset_normalization.json'
        if not stats_path.is_file(): raise FileNotFoundError(f'dataset normalization stats not found: {stats_path}')
        normalization = json.loads(stats_path.read_text(encoding='utf-8'))
    sizes = args.tta_sizes or [int(config['input_size'])]
    if not sizes or any(size<32 for size in sizes) or len(set(sizes))!=len(sizes):
        raise ValueError('TTA sizes must be distinct integers >=32')
    manifest = args.manifest.resolve() if args.manifest else resolve_path(config,'test_manifest')
    final_ids = final_targets = total = coverage = None
    for size in sizes:
        set_inference_size(model,config,size,state)
        # Re-open original files for each scale. Never resize a previously normalized tensor.
        dataset = StudyDataset(manifest,args.data_root,build_transform(False,size,config,normalization),
                               labeled=args.labeled,view_filter=config.get('view_filter'),
                               use_view_embedding=config.get('use_view_embedding',True))
        ids, probabilities, targets = predict_dataset(model,dataset,config,device,args.tta=='horizontal-flip')
        if final_ids is None:
            final_ids,final_targets,total,coverage = ids,targets,probabilities,dataset.coverage
        else:
            if ids!=final_ids or (targets is not None and not torch.equal(targets,final_targets)):
                raise ValueError('Study order or labels changed between TTA passes')
            total += probabilities
    total /= len(sizes)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8',newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Study_id',*([f'target_{i}' for i in range(10)]+[f'probability_{i}' for i in range(10)] if args.labeled else LABEL_COLUMNS)])
        for i,study_id in enumerate(final_ids):
            writer.writerow([study_id,*(final_targets[i].tolist() if args.labeled else []),*total[i].tolist()])
    metadata = {'checkpoint':str(checkpoint_path),'checkpoint_sha256':file_sha256(checkpoint_path),
                'manifest':str(manifest),'manifest_sha256':file_sha256(manifest),'config':str(args.config.resolve()),
                'weight_source':weight_source,'weights_requested':args.weights,'labeled':args.labeled,
                'tta':args.tta,'tta_sizes':sizes,'tta_average':'equal probabilities over sizes and flips',
                'coverage':coverage,'output':str(output),'output_sha256':file_sha256(output)}
    sidecar.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Wrote {len(final_ids)} Study predictions to {output}')

if __name__=='__main__': main()
