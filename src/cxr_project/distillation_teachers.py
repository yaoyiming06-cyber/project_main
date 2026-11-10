"""Sequential native-resolution teacher inference on the training split only."""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .config import load_config, resolve_path
from .dataset import StudyDataset, build_transform
from .distillation import TeacherCache, validate_splits, validate_teacher_records, write_teacher_cache
from .model import build_model, file_sha256
from .runtime import select_device
from .train import set_seed
from scripts.predict_test import inference_weights, predict_dataset, validate_checkpoint_config


def _resolve(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _predict_teacher(job, train_manifest, image_root, device):
    config = job['runtime_config']
    checkpoint = torch.load(job['checkpoint'], map_location='cpu', weights_only=False)
    validate_checkpoint_config(config, checkpoint)
    saved_data = checkpoint.get('data') or {}
    for key in ('train_manifest_sha256', 'valid_manifest_sha256'):
        if saved_data.get(key) is not None and saved_data[key] != job[key]:
            raise ValueError(f'teacher checkpoint training source differs: {job["name"]} / {key}')
    state, source = inference_weights(checkpoint, job['weights'])
    if any('view_embedding' in key for key in state):
        raise ValueError('distillation teacher state must be image-only')
    normalization = checkpoint.get('normalization_stats') or checkpoint.get('normalization')
    if config['normalization'] == 'dataset' and not isinstance(normalization, dict):
        path = Path(job['checkpoint']).parent / 'dataset_normalization.json'
        normalization = json.loads(path.read_text(encoding='utf-8'))
    model = build_model(config, pretrained=False).float()
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    model.to(device).eval()
    dataset = StudyDataset(
        train_manifest, image_root, build_transform(False, int(config['input_size']), config, normalization),
        labeled=False, use_view_embedding=False, view_filter=None,
    )
    ids, probabilities, _ = predict_dataset(model, dataset, config, device)
    if source != job['weights']:
        raise ValueError('teacher weight source changed')
    return ids, probabilities.numpy()


def build_teacher_cache(student_config, teachers_path, output, *, device_name='cuda', batch_size=None, num_workers=None):
    """Preflight every pinned teacher, then infer/cache one at a time.

    Per-teacher caches are reusable after interruption. They must match the
    current recipe and train/valid hashes exactly. No test manifest is opened.
    """
    config = load_config(student_config)
    if (config['final_training'] or config['use_view_embedding'] or config['view_filter'] is not None
            or config['train_label_indices'] is not None):
        raise ValueError('distillation requires image-only full-label train/valid splits')
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError('teacher batch_size must be positive')
    if num_workers is not None and (type(num_workers) is not int or num_workers < 0):
        raise ValueError('teacher num_workers must be nonnegative')
    output = Path(output).expanduser().resolve()
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(f'teacher output or sidecar already exists: {output}')
    if output.suffix != '.npz':
        raise ValueError('teacher output must end in .npz')
    root = config['_project_root']
    train_manifest, valid_manifest = resolve_path(config, 'train_manifest'), resolve_path(config, 'valid_manifest')
    train, _ = validate_splits(train_manifest, valid_manifest)
    ids = [s.study_id for s in train]
    hashes = {key: file_sha256(path) for key, path in (
        ('train_manifest_sha256', train_manifest), ('valid_manifest_sha256', valid_manifest))}
    recipe = json.loads(Path(teachers_path).read_text(encoding='utf-8'))
    if not isinstance(recipe, dict) or recipe.get('format') != 'cxr_distillation_teachers_v1':
        raise ValueError('invalid distillation teachers recipe')
    if any(recipe.get(key) != digest for key, digest in hashes.items()):
        raise ValueError('teacher recipe train/valid manifest hash mismatch')
    models = recipe.get('models')
    if not isinstance(models, list) or not models:
        raise ValueError('teacher recipe needs at least one model')
    device = select_device(device_name)
    jobs, records = [], []
    for spec in models:
        teacher_config_path = _resolve(root, spec['config'])
        checkpoint_path = _resolve(root, spec['checkpoint'])
        for path, key in ((teacher_config_path, 'config_sha256'), (checkpoint_path, 'checkpoint_sha256')):
            if not path.is_file():
                raise FileNotFoundError(f'teacher file is missing: {path}')
            if file_sha256(path) != spec.get(key):
                raise ValueError(f'teacher {key} hash mismatch: {path}')
        teacher_config = load_config(teacher_config_path)
        if teacher_config['use_view_embedding'] or teacher_config['view_filter'] is not None:
            raise ValueError(f'teacher must be image-only: {spec.get("name")}')
        if teacher_config['train_label_indices'] is not None:
            raise ValueError('teacher must predict all ten labels')
        if spec.get('weights') not in {'ema', 'raw'}:
            raise ValueError('teacher weight selection must be explicitly ema or raw')
        weight = spec.get('fusion_weight')
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError('teacher fusion_weight must be finite and positive')
        teacher_config['num_workers'] = config['num_workers'] if num_workers is None else num_workers
        teacher_config['batch_size_studies'] = config['batch_size_studies'] if batch_size is None else batch_size
        record = dict(name=spec['name'], checkpoint_sha256=spec['checkpoint_sha256'],
                      config_sha256=spec['config_sha256'], weight_source=spec['weights'],
                      fusion_weight=float(weight), input_size=teacher_config['input_size'],
                      inference_device=str(device), amp=bool(teacher_config['amp'] and device.type == 'cuda'),
                      batch_size_studies=teacher_config['batch_size_studies'],
                      torch_version=str(torch.__version__))
        records.append(record)
        jobs.append({**spec, **hashes, 'checkpoint': checkpoint_path, 'runtime_config': teacher_config})
    validate_teacher_records(records)
    progress = output.with_suffix('.teachers')
    progress.mkdir(parents=True, exist_ok=True)
    total = np.zeros((len(ids), 10), dtype=np.float64)
    weight_sum = sum(r['fusion_weight'] for r in records)
    for i, (job, record) in enumerate(zip(jobs, records)):
        partial = progress / f'teacher_{i:02d}.npz'
        started = time.monotonic()
        if partial.exists() or partial.with_suffix('.json').exists():
            cached = TeacherCache(partial, train_manifest=train_manifest, valid_manifest=valid_manifest, study_ids=ids)
            if cached.provenance['teachers'] != [record]:
                raise ValueError('per-teacher progress differs from current recipe/runtime; use a fresh output')
            print(f'Reusing verified teacher {i+1}/{len(jobs)}: {record["name"]}', flush=True)
        else:
            print(f'Predicting teacher {i+1}/{len(jobs)}: {record["name"]} @ {record["input_size"]}', flush=True)
            set_seed(int(config['seed']))
            predicted_ids, probabilities = _predict_teacher(job, train_manifest, resolve_path(config, 'image_root'), device)
            write_teacher_cache(partial, predicted_ids, probabilities, train_manifest=train_manifest,
                                valid_manifest=valid_manifest, teachers=[record])
            cached = TeacherCache(partial, train_manifest=train_manifest, valid_manifest=valid_manifest, study_ids=ids)
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()
        total += cached.lookup(ids, 'cpu').numpy().astype(np.float64) * (record['fusion_weight'] / weight_sum)
        print(f'Teacher {i+1}/{len(jobs)} finished in {(time.monotonic()-started)/60:.2f} min', flush=True)
    return write_teacher_cache(output, ids, total, train_manifest=train_manifest,
                               valid_manifest=valid_manifest, teachers=records)
