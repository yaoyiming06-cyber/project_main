"""Auditable evaluation utilities operating only on explicit manifests/predictions."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

def _read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    if not fields or not rows: raise ValueError(f'empty CSV: {path}')
    return fields, rows

def _write_csv(path, fields, rows):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def read_predictions(path):
    fields, rows = _read_csv(path)
    prefix = 'probability' if all(f'probability_{j}' in fields for j in range(10)) else 'label'
    if not {'Study_id', *[f'{prefix}_{j}' for j in range(10)]}.issubset(fields):
        raise ValueError('prediction CSV requires Study_id and probability_0..9')
    ids = [r['Study_id'] for r in rows]
    if len(set(ids)) != len(ids) or any(not s for s in ids): raise ValueError('duplicate/empty Study_id')
    probabilities = np.array([[float(r[f'{prefix}_{j}']) for j in range(10)] for r in rows])
    if not np.isfinite(probabilities).all() or (probabilities<0).any() or (probabilities>1).any(): raise ValueError('invalid probabilities')
    present = [f'target_{j}' in fields for j in range(10)]
    if any(present) and not all(present): raise ValueError('partial target columns')
    targets = np.array([[float(r[f'target_{j}']) for j in range(10)] for r in rows]) if all(present) else None
    if targets is not None and not np.isin(targets,[-1,0,1]).all(): raise ValueError('targets must be -1/0/1')
    return {'ids':ids, 'probabilities':probabilities, 'targets':targets, 'source':str(Path(path).resolve()), 'sha256':sha256(path)}

def aligned_predictions(predictions):
    if not predictions: raise ValueError('at least one prediction file required')
    ids = predictions[0]['ids']; result=[]; target=predictions[0]['targets']
    for p in predictions:
        if set(ids)!=set(p['ids']): raise ValueError('Study_id coverage mismatch')
        lookup={s:i for i,s in enumerate(p['ids'])}; order=[lookup[s] for s in ids]
        t = None if p['targets'] is None else p['targets'][order]
        if (t is None)!=(target is None) or (t is not None and not np.array_equal(target,t)): raise ValueError('target mismatch')
        result.append(p['probabilities'][order])
    return ids,target,result

def _metrics(targets, probabilities):
    if targets is None: return {}
    from .metrics import compute_macro_auc
    return compute_macro_auc(targets, probabilities)

def write_predictions(path, ids, targets, probabilities):
    prefix='probability' if targets is not None else 'label'
    fields=['Study_id'] + ([f'target_{j}' for j in range(10)] if targets is not None else []) + [f'{prefix}_{j}' for j in range(10)]
    rows=[]
    for i,s in enumerate(ids):
        r={'Study_id':s, **{f'{prefix}_{j}':float(probabilities[i,j]) for j in range(10)}}
        if targets is not None: r.update({f'target_{j}':float(targets[i,j]) for j in range(10)})
        rows.append(r)
    _write_csv(path,fields,rows)

def fuse_predictions(paths, output, label_indices=None):
    predictions=[read_predictions(p) for p in paths]
    ids,targets,arrays=aligned_predictions(predictions)
    probabilities=arrays[0].copy()
    labels=list(range(10)) if label_indices is None else list(label_indices)
    if not labels or len(set(labels))!=len(labels) or any(j not in range(10) for j in labels): raise ValueError('label_indices must be unique indices 0..9')
    probabilities[:,labels]=np.mean(arrays,axis=0)[:,labels]
    write_predictions(output,ids,targets,probabilities)
    return {'method':'fixed equal probability mean','sources':[{'path':p['source'],'sha256':p['sha256']} for p in predictions], **_metrics(targets,probabilities)}

def compare_predictions(reference, candidate):
    ids,targets,arrays=aligned_predictions([read_predictions(reference),read_predictions(candidate)])
    a,b=arrays
    ma,mb=_metrics(targets,a),_metrics(targets,b)
    return {'studies':len(ids),'maximum_absolute_probability_error':float(np.abs(a-b).max()),'mean_absolute_probability_error':float(np.abs(a-b).mean()),'reference':ma,'candidate':mb,'delta_macro_auc':None if not ma or ma['macro_auc'] is None else mb['macro_auc']-ma['macro_auc']}

def paired_bootstrap(first, second, manifest, *, iterations=2000, seed=42):
    if iterations<1: raise ValueError('iterations must be positive')
    ids,targets,arrays=aligned_predictions([read_predictions(first),read_predictions(second)])
    if targets is None: raise ValueError('bootstrap requires labeled validation predictions')
    _,rows=_read_csv(manifest); subjects={}
    for r in rows:
        study,subject=r['Study_id'],r['Subject_id']
        if not subject or (study in subjects and subjects[study]!=subject): raise ValueError('invalid Study/Subject mapping')
        subjects[study]=subject
    if not set(ids).issubset(subjects): raise ValueError('manifest missing predictions Study_id')
    groups={}
    for i,s in enumerate(ids): groups.setdefault(subjects[s],[]).append(i)
    keys=sorted(groups); rng=np.random.default_rng(seed); deltas=[]; skipped=0
    eligible=[j for j in range(10) if len(np.unique(targets[np.isin(targets[:,j],[0,1]),j]))==2]
    if not eligible: raise ValueError('no label has both positive and negative targets')
    for _ in range(iterations):
        idx=[i for k in rng.choice(keys,len(keys),replace=True) for i in groups[k]]
        t=targets[idx]
        if any(len(np.unique(t[np.isin(t[:,j],[0,1]),j]))<2 for j in eligible): skipped+=1; continue
        a,b=(_metrics(t,p[idx]) for p in arrays)
        deltas.append(b['macro_auc']-a['macro_auc'])
    if not deltas: raise ValueError('all bootstrap draws invalid; more patients required')
    return {'direction':'second minus first','iterations':iterations,'valid_draws':len(deltas),'skipped_draws':skipped,'invalid_draw_policy':'skip entire paired draw if any originally eligible label loses a class','patients':len(keys),'seed':seed,'eligible_label_indices':eligible,'observed_delta':_metrics(targets,arrays[1])['macro_auc']-_metrics(targets,arrays[0])['macro_auc'],'delta_ci95':[float(x) for x in np.quantile(deltas,[.025,.975])],'probability_delta_positive':float(np.mean(np.array(deltas)>0))}

def make_subject_folds(manifest, output_dir, folds=3, seed=42):
    fields,rows=_read_csv(manifest); mapping={}; groups={}
    for r in rows:
        subject,study=r['Subject_id'],r['Study_id']
        if not subject or not study: raise ValueError('empty subject/study')
        if study in mapping and mapping[study]!=subject: raise ValueError('Study has multiple subjects')
        mapping[study]=subject; groups.setdefault(subject,[]).append(r)
    if folds<2 or len(groups)<folds: raise ValueError('not enough subjects for folds')
    rng=np.random.default_rng(seed); subjects=list(groups); rng.shuffle(subjects)
    subjects.sort(key=lambda s:len({r['Study_id'] for r in groups[s]}),reverse=True)
    assigned=[[] for _ in range(folds)]; sizes=[0]*folds
    for s in subjects:
        f=int(np.argmin(sizes));assigned[f].append(s);sizes[f]+=len({r['Study_id'] for r in groups[s]})
    out=Path(output_dir).resolve(); result={'source':str(Path(manifest).resolve()),'source_sha256':sha256(manifest),'seed':seed,'method':'subject grouped, greedily study-count balanced (not label stratified)','folds':[]}
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    for f,valid_subjects in enumerate(assigned):
        valid_subjects=set(valid_subjects); train=out/f'fold{f+1}_train.csv';valid=out/f'fold{f+1}_valid.csv'
        _write_csv(train,fields,[r for r in rows if r['Subject_id'] not in valid_subjects])
        _write_csv(valid,fields,[r for r in rows if r['Subject_id'] in valid_subjects])
        result['folds'].append({'fold':f+1,'train_manifest':str(train),'valid_manifest':str(valid),'valid_subjects':len(valid_subjects)})
    (out/'folds.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

def merge_final_manifests(train, valid, output, *, final=False):
    if not final: raise ValueError('requires explicit --final after freezing all hyperparameters')
    fields,a=_read_csv(train);other,b=_read_csv(valid)
    if fields!=other: raise ValueError('manifest columns differ')
    for key in ['Subject_id','Study_id','image_path']:
        if {r[key] for r in a}&{r[key] for r in b}: raise ValueError(f'train/valid overlap: {key}')
    _write_csv(output,fields,a+b)
    return {'train_sha256':sha256(train),'valid_sha256':sha256(valid),'merged_sha256':sha256(output),'rows':len(a+b),'selection_policy':'fixed epochs; no validation or checkpoint selection on merged valid'}

def route_predictions(base_path, specialists, manifest, output, label_indices=None):
    """Fixed equal blend of base and every applicable available view specialist."""
    base=read_predictions(base_path); ids=base['ids']; index={s:i for i,s in enumerate(ids)}
    labels=list(range(10)) if label_indices is None else list(label_indices)
    if not labels or len(set(labels))!=len(labels) or any(j not in range(10) for j in labels): raise ValueError('label_indices must be unique indices 0..9')
    _,rows=_read_csv(manifest); views={}
    for r in rows: views.setdefault(r['Study_id'],set()).add(r['ViewPosition'].strip().upper())
    if not set(ids).issubset(views): raise ValueError('manifest missing base Study_id')
    sums=base['probabilities'].copy();counts=np.ones(len(ids));coverage={}
    for view,path in specialists.items():
        p=read_predictions(path); n=0
        for j,s in enumerate(p['ids']):
            if s not in index: raise ValueError('specialist contains unknown Study_id')
            i=index[s]
            if view.upper() not in views[s]: raise ValueError('specialist row has no matching view')
            if (base['targets'] is None)!=(p['targets'] is None) or (base['targets'] is not None and not np.array_equal(base['targets'][i],p['targets'][j])): raise ValueError('specialist target mismatch')
            sums[i]+=p['probabilities'][j];counts[i]+=1;n+=1
        coverage[view]=n
    probs=base['probabilities'].copy();probs[:,labels]=(sums/counts[:,None])[:,labels];write_predictions(output,ids,base['targets'],probs)
    return {'coverage':coverage,'fallback_studies':int((counts==1).sum()),'method':'equal mean of base plus available matching specialists; missing expert falls back to base',**_metrics(base['targets'],probs)}

def export_fp16(checkpoint_path, output, *, weights='auto'):
    import torch
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    from .runtime import select_inference_state
    state, weights = select_inference_state(checkpoint, weights)
    if 'module' in state and isinstance(state['module'],dict): state=state['module']
    if 'state_dict' in state and isinstance(state['state_dict'],dict): state=state['state_dict']
    compact={k:(v.half() if v.is_floating_point() else v) for k,v in state.items()}
    normalization=checkpoint.get('normalization_stats') or checkpoint.get('normalization')
    if (checkpoint.get('config') or {}).get('normalization')=='dataset' and normalization is None:
        stats=Path(checkpoint_path).parent/'dataset_normalization.json'
        if not stats.is_file(): raise FileNotFoundError('dataset normalization stats required for self-contained export')
        normalization=json.loads(stats.read_text(encoding='utf-8'))
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':compact,'config':checkpoint.get('config'), 'normalization':normalization,'normalization_stats':normalization,'selected_weights':weights,'weight_source':weights,'ema_active':False,'format':'cxr_fp16_inference_v1'},output)
    return {'output':str(output.resolve()),'bytes':output.stat().st_size,'selected_weights':weights,'source_sha256':sha256(checkpoint_path),'export_sha256':sha256(output),'verification':'run identical validation inference on source/export then compare; export alone does not verify accuracy'}
