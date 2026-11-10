"""Train-only, immutable Study probability caches and multilabel distillation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .labels import LABEL_NAMES
from .manifest import read_study_manifest
from .model import file_sha256


CACHE_FORMAT = 'cxr_teacher_probabilities_v1'


def validate_splits(train_manifest, valid_manifest):
    """Read only train/valid; require disjoint Studies and patients."""
    train = read_study_manifest(train_manifest, include_view_positions=False)
    valid = read_study_manifest(valid_manifest, include_view_positions=False)
    for field in ('study_id', 'subject_id'):
        overlap = {getattr(s, field) for s in train} & {getattr(s, field) for s in valid}
        if overlap:
            raise ValueError(f'distillation train/valid {field} overlap: {sorted(overlap)[:5]}')
    return train, valid


def validate_teacher_records(teachers):
    if not isinstance(teachers, list) or not teachers:
        raise ValueError('distillation cache requires teacher provenance')
    names = set()
    for teacher in teachers:
        if not isinstance(teacher, dict) or not teacher.get('name') or teacher['name'] in names:
            raise ValueError('teacher names must be nonempty and unique')
        names.add(teacher['name'])
        for key in ('checkpoint_sha256', 'config_sha256'):
            digest = teacher.get(key)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError(f'teacher requires a valid {key}')
        weight = teacher.get('fusion_weight')
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError('teacher fusion_weight must be finite and positive')
        if teacher.get('weight_source') not in {'ema', 'raw'}:
            raise ValueError('teacher must record its selected weight_source')


def _validate_arrays(ids, probabilities, expected_ids):
    ids = list(ids)
    if any(not isinstance(s, str) or not s for s in ids) or len(set(ids)) != len(ids):
        raise ValueError('teacher cache must contain unique nonempty Study IDs')
    if len(ids) != len(expected_ids) or set(ids) != set(expected_ids):
        raise ValueError('teacher cache Study coverage does not match the training manifest')
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != (len(ids), 10):
        raise ValueError('teacher probabilities must have shape [training Studies, 10]')
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('teacher probabilities must be finite and within [0, 1]')
    return ids, probabilities


def write_teacher_cache(output, study_ids, probabilities, *, train_manifest, valid_manifest, teachers):
    """Write FP32 targets with source hashes; never overwrite existing artifacts."""
    path = Path(output).expanduser().resolve()
    sidecar = path.with_suffix('.json')
    if path.suffix != '.npz':
        raise ValueError('teacher cache output must end in .npz')
    if path.exists() or sidecar.exists():
        raise FileExistsError(f'teacher cache or sidecar already exists: {path}')
    train, _ = validate_splits(train_manifest, valid_manifest)
    ids, values = _validate_arrays(study_ids, probabilities, [s.study_id for s in train])
    validate_teacher_records(teachers)
    metadata = {
        'format': CACHE_FORMAT, 'split': 'train', 'labels': list(LABEL_NAMES),
        'train_manifest_sha256': file_sha256(Path(train_manifest)),
        'valid_manifest_sha256': file_sha256(Path(valid_manifest)),
        'studies': len(ids), 'teachers': teachers,
        'fusion': 'weighted_probability_mean', 'temperature': 1.0,
        'tta': 'none', 'patient_smoothing_alpha': 0.0,
        'probability_dtype': 'float32',
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.npz.tmp')
    metadata_tmp = sidecar.with_suffix('.json.tmp')
    if temporary.exists() or metadata_tmp.exists():
        raise FileExistsError('incomplete teacher cache exists; use a fresh output path')
    try:
        with temporary.open('xb') as handle:
            np.savez(handle, study_ids=np.asarray(ids), probabilities=values)
        metadata['cache_sha256'] = file_sha256(temporary)
        metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
        metadata_tmp.replace(sidecar)
    finally:
        temporary.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
    return metadata


class TeacherCache:
    """Small CPU-resident table; shuffled and repeated Study lookups are safe."""

    def __init__(self, path, *, train_manifest, valid_manifest, study_ids):
        path = Path(path).expanduser().resolve()
        sidecar = path.with_suffix('.json')
        try:
            metadata = json.loads(sidecar.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f'invalid teacher cache sidecar: {sidecar}') from error
        if not isinstance(metadata, dict) or metadata.get('format') != CACHE_FORMAT or metadata.get('split') != 'train':
            raise ValueError('teacher cache must be a versioned train-only cache')
        if metadata.get('labels') != list(LABEL_NAMES):
            raise ValueError('teacher cache label order mismatch')
        for key, source in (('train_manifest_sha256', train_manifest), ('valid_manifest_sha256', valid_manifest)):
            if metadata.get(key) != file_sha256(Path(source)):
                raise ValueError(f'teacher cache manifest hash mismatch: {key}')
        if metadata.get('cache_sha256') != file_sha256(path):
            raise ValueError('teacher cache payload hash mismatch')
        if (metadata.get('temperature') != 1.0 or metadata.get('tta') != 'none'
                or metadata.get('patient_smoothing_alpha') != 0.0
                or metadata.get('fusion') != 'weighted_probability_mean'):
            raise ValueError('teacher cache protocol mismatch')
        validate_teacher_records(metadata.get('teachers'))
        train, _ = validate_splits(train_manifest, valid_manifest)
        expected = [s.study_id for s in train]
        if len(study_ids) != len(expected) or set(study_ids) != set(expected):
            raise ValueError('student Study coverage must match the complete training split')
        with np.load(path, allow_pickle=False) as cache:
            if set(cache.files) != {'study_ids', 'probabilities'}:
                raise ValueError('invalid teacher cache arrays')
            ids, values = _validate_arrays(cache['study_ids'].tolist(), cache['probabilities'], expected)
        if metadata.get('studies') != len(ids):
            raise ValueError('teacher cache Study count mismatch')
        self._index = {study: i for i, study in enumerate(ids)}
        self._probabilities = torch.from_numpy(values.copy())
        self.provenance = {**metadata, 'metadata_sha256': file_sha256(sidecar)}

    def lookup(self, study_ids, device):
        try:
            indices = [self._index[study] for study in study_ids]
        except KeyError as error:
            raise ValueError(f'teacher cache is missing Study {error.args[0]}') from error
        return self._probabilities[indices].to(device=device, dtype=torch.float32)


def distillation_loss(logits, teacher_probabilities, targets, temperature=1.0):
    """Independent Bernoulli soft BCE; preserve hard-label uncertainty masks."""
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('distillation temperature must be finite and positive')
    if logits.shape != teacher_probabilities.shape or logits.shape != targets.shape:
        raise ValueError('distillation logits, probabilities and targets must have matching shapes')
    valid = targets >= 0
    if not valid.any():
        return logits.sum() * 0.0
    probabilities = teacher_probabilities.detach().float().clamp(1e-6, 1 - 1e-6)
    soft_targets = (torch.logit(probabilities) / temperature).sigmoid()
    per_label = F.binary_cross_entropy_with_logits(logits.float() / temperature, soft_targets, reduction='none')
    return per_label[valid].mean() * temperature ** 2
