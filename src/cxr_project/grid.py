"""Explicit single-factor experiment recipes; never infer YAML switches from prose."""
from copy import deepcopy
from pathlib import Path

PATH_KEYS = ('image_root', 'train_manifest', 'valid_manifest', 'test_manifest', 'pretrained_checkpoint')
RECIPES = {
    'A0': {'freeze_backbone': False, 'unfreeze_last_stages': None, 'validation_weights': 'auto', 'final_training': False, 'epochs': 15, 'warmup_epochs': 5, 'lr': .0005, 'ema': True, 'ema_decay': .9998, 'ema_start_epoch': 1, 'clip_grad': 1., 'early_stopping_patience': 0, 'evaluate_raw_and_ema': True},
    'C1': {'ema': False, 'clip_grad': None}, 'C2': {'ema': False, 'clip_grad': 1.},
    'C3': {'ema': True, 'clip_grad': None}, 'C4': {'ema': True, 'evaluate_raw_and_ema': True},
    'C5': {'ema_decay': .999}, 'C6': {'ema_decay': .9995},
    'O1': {'warmup_epochs': 1}, 'O2': {'warmup_epochs': 2}, 'O3': {'lr': .0002},
    'O4': {'lr': .0003}, 'O5': {'lr': .0001}, 'O6': {'epochs': 10}, 'O7': {'epochs': 20},
    'O8': {'layer_decay': .65}, 'O9': {'layer_decay': .9}, 'O10': {'_effective_batch': 32},
    'I1': {'input_size': 320}, 'I2': {'input_size': 384}, 'I3': {'clahe': True},
    'I4': {'input_size': 192}, 'I5': {'input_size': 224}, 'I6': {'input_size': 256},
    'R1': {'weight_decay': .02}, 'R2': {'weight_decay': .1}, 'R3': {'drop_path': 0.},
    'R4': {'drop_path': .2}, 'R5': {'head_dropout': 0., 'expert_dropout': 0.},
    'R6': {'head_dropout': .4, 'expert_dropout': .4},
    'G0': {'augmentation': 'G0', 'horizontal_flip': False, 'horizontal_flip_prob': 0., 'gamma_prob': 0., 'random_erasing_prob': 0., 'mixup_alpha': 0.},
    'G2': {'augmentation': 'G1', 'augmentation_degrees': 5., 'augmentation_translate': .04, 'augmentation_scale': [.95, 1.05]},
    'G3': {'horizontal_flip': True, 'horizontal_flip_prob': .5},
    'G4': {'gamma_prob': .3, 'gamma_range': [.8,1.2]},
    'G5': {'random_erasing_prob': .1, 'random_erasing_scale': [.005,.02]},
    'G6': {'mixup_alpha': .2}, 'G7': {'augmentation': 'AutoAug'}, 'G8': {'normalization': 'dataset'},
    'L1': {'pos_weight_mode': 'sqrt'}, 'L2': {'pos_weight_mode': 'unit'}, 'L3': {'max_pos_weight': 10.},
    'L4': {'loss': 'focal', 'focal_gamma': 1.}, 'L5': {'loss': 'focal', 'focal_gamma': 2.},
    'L6': {'loss': 'asl', 'asl_gamma_neg': 4., 'asl_gamma_pos': 0., 'asl_clip': .05},
    'L7': {'sampling_strategy': 'weighted', 'sampling_power': .5, 'pos_weight_mode': 'unit', 'sampling_max_weight': 3.},
    'M1': {'experts_enabled': False, 'hard_label_indices': [], 'expert_loss_weight': 0.},
    'M2': {'expert_loss_weight': .25}, 'M3': {'study_pooling': 'mean'},
    'M4': {'use_view_embedding': False}, 'M5': {'freeze_backbone': False, 'unfreeze_last_stages': 2},
    'M6': {'view_filter': ['AP']}, 'M7': {'view_filter': ['PA']}, 'M8': {'view_filter': ['LATERAL']}, 'M9': {'view_filter': ['LL']},
    'B1': {'model': 'swin_base_patch4_window7_224.ms_in22k'}, 'B2': {'model': 'convnext_small.fb_in22k'},
    'B3': {},
    'AF0': {'model': 'vit_base_patch16_224.augreg_in21k', 'freeze_backbone': True, 'adaptformer': False, 'experts_enabled': False, 'hard_label_indices': [], 'expert_loss_weight': 0., 'train_label_indices': [0,3,7], 'normalization': 'pretrained', 'unfreeze_last_stages': None},
    'AF1': {'model': 'vit_base_patch16_224.augreg_in21k', 'freeze_backbone': True, 'adaptformer': True, 'experts_enabled': False, 'hard_label_indices': [], 'expert_loss_weight': 0., 'train_label_indices': [0,3,7], 'normalization': 'pretrained', 'unfreeze_last_stages': None},
}
EVALUATION_RECIPES = {'V1': 'folds', 'V2': 'bootstrap', 'V3': 'predict_test.py --tta horizontal-flip', 'V4': 'predict_test.py --tta-sizes', 'V5': 'fuse', 'V6': 'export-fp16 / compare', 'V7': 'final-manifest'}

def experiment_overrides(experiment):
    if experiment not in RECIPES:
        raise ValueError(f'{experiment} is not a training recipe; evaluation commands: {EVALUATION_RECIPES}')
    return deepcopy(RECIPES[experiment])

def build_configs(baseline, experiments, seeds, project_root, output_root, *, stage='grid', weights=None):
    if not seeds or any(int(s) < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be unique non-negative integers')
    if not experiments or len(set(experiments)) != len(experiments):
        raise ValueError('experiments must be non-empty and unique')
    if not stage or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in stage):
        raise ValueError('stage must contain letters, numbers, underscores or hyphens')
    project_root, output_root = Path(project_root).resolve(), Path(output_root).resolve()
    result = {}
    for experiment in experiments:
        changes = experiment_overrides(experiment)
        for seed in seeds:
            config = {k: deepcopy(v) for k,v in baseline.items() if not k.startswith('_')}
            config.update(changes)
            effective = config.pop('_effective_batch', None)
            if effective is not None:
                micro = int(config['batch_size_studies'])
                if effective % micro: raise ValueError('effective batch must be divisible by micro batch')
                config['accum_steps'] = effective // micro
            if experiment in ('B1','B2','AF0','AF1'):
                if not weights or experiment not in weights:
                    raise ValueError(f'{experiment} requires an explicit matching --weights {experiment}=PATH')
                config['pretrained_checkpoint'] = str(Path(weights[experiment]).resolve())
                from .evaluation_tools import sha256
                config['pretrained_sha256'] = sha256(config['pretrained_checkpoint'])
            for key in PATH_KEYS:
                if config.get(key) is not None:
                    value = Path(config[key]).expanduser()
                    config[key] = str(value.resolve() if value.is_absolute() else (project_root / value).resolve())
            name = f'{stage}_{experiment}_seed{seed}'
            config.update(seed=int(seed), run_name=name, output_dir=str(output_root / name), project_root=str(project_root))
            result[name] = config
    return result
