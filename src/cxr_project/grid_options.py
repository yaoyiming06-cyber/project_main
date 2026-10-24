"""Strict optional grid configuration; defaults preserve historical behavior."""
import math
DEFAULTS = {
 "ema": False, "ema_decay": .9998, "ema_start_epoch": 1,
 "early_stopping_patience": 0, "early_stopping_min_delta": 0.,
 "freeze_backbone": False, "unfreeze_last_stages": None,
 "experts_enabled": True, "head_dropout": .2, "expert_dropout": .2,
 "study_pooling": "attention", "use_view_embedding": True,
 "adaptformer": False, "adapter_dim": 64, "adapter_dropout": .1, "adapter_scale": 1.,
 "pretrained_sha256": None, "view_filter": None,
 "gamma_prob": 0., "gamma_range": [.8,1.2], "random_erasing_prob": 0.,
 "random_erasing_scale": [.005,.02], "horizontal_flip": False,
 "clahe": False, "clahe_clip_limit": 2., "clahe_grid_size": [8,8],
 "mixup_alpha": 0., "sampling_strategy": "shuffle", "sampling_power": .5,
 "sampling_max_weight": 3., "sampling_warmup_epochs": 0,
 "asl_gamma_neg": 4., "asl_gamma_pos": 0., "asl_clip": .05,
 "evaluate_raw_and_ema": False, "validation_weights": "auto",
 "final_training": False, "project_root": None, "train_label_indices": None,
}
OPTIONAL_KEYS = set(DEFAULTS) | {
 "horizontal_flip_prob", "augmentation_degrees", "augmentation_translate", "augmentation_scale",
 "color_jitter_brightness", "color_jitter_contrast", "normalization_mean", "normalization_std",
}
def validate_options(c):
 for key in ('ema','freeze_backbone','experts_enabled','use_view_embedding','adaptformer','clahe',
             'horizontal_flip','evaluate_raw_and_ema','final_training','amp','pretrained','pin_memory','gradient_checkpointing'):
  if type(c[key]) is not bool: raise ValueError(f'{key} must be a YAML boolean')
 for key in ('seed','warmup_epochs','num_workers','early_stopping_patience','sampling_warmup_epochs'):
  if type(c[key]) is not int or c[key]<0: raise ValueError(f'{key} must be a nonnegative integer')
 for key in ('adapter_dim','ema_start_epoch'):
  if type(c[key]) is not int or c[key]<1: raise ValueError(f'{key} must be a positive integer')
 for key in ('head_dropout','expert_dropout','adapter_dropout','gamma_prob','random_erasing_prob','horizontal_flip_prob','sampling_power'):
  if key in c and (not math.isfinite(float(c[key])) or not 0<=float(c[key])<=1): raise ValueError(f'{key} must be in [0,1]')
 for key in ('mixup_alpha','asl_gamma_neg','asl_gamma_pos','early_stopping_min_delta'):
  if not math.isfinite(float(c[key])) or float(c[key])<0: raise ValueError(f'{key} must be finite and nonnegative')
 if not 0<float(c['ema_decay'])<1: raise ValueError('ema_decay must be in (0,1)')
 if not 0<=float(c['asl_clip'])<1: raise ValueError('asl_clip must be in [0,1)')
 if float(c['sampling_max_weight'])<1: raise ValueError('sampling_max_weight must be >=1')
 if float(c['clahe_clip_limit'])<=0: raise ValueError('clahe_clip_limit must be positive')
 if not math.isfinite(float(c['adapter_scale'])): raise ValueError('adapter_scale must be finite')
 for key in ('gamma_range','random_erasing_scale'):
  v=c[key]
  if not isinstance(v,(list,tuple)) or len(v)!=2 or not 0<float(v[0])<=float(v[1]): raise ValueError(f'{key} must be an increasing positive pair')
 if float(c['random_erasing_scale'][1])>1: raise ValueError('random_erasing_scale must be <=1')
 grid=c['clahe_grid_size']
 if not isinstance(grid,(list,tuple)) or len(grid)!=2 or any(type(v) is not int or v<1 for v in grid): raise ValueError('clahe_grid_size must contain two positive integers')
 if c['study_pooling'] not in {'attention','mean'}: raise ValueError('study_pooling must be attention or mean')
 if c['sampling_strategy'] not in {'shuffle','weighted'}: raise ValueError('sampling_strategy must be shuffle or weighted')
 if c['sampling_strategy']=='weighted' and c['pos_weight_mode']!='unit': raise ValueError('weighted sampling requires pos_weight_mode unit')
 if c['validation_weights'] not in {'auto','raw','ema'}: raise ValueError('validation_weights must be auto, raw, or ema')
 if c['validation_weights']=='ema' and not c['ema']: raise ValueError('EMA validation requires ema true')
 if c['unfreeze_last_stages'] is not None:
  if type(c['unfreeze_last_stages']) is not int or c['unfreeze_last_stages']<1: raise ValueError('unfreeze_last_stages must be a positive integer or null')
  if c['freeze_backbone']: raise ValueError('freeze_backbone and unfreeze_last_stages are mutually exclusive')
 if c['view_filter'] is not None:
  from .labels import VIEW_NAMES
  if not isinstance(c['view_filter'],list) or not c['view_filter'] or any(v not in VIEW_NAMES for v in c['view_filter']): raise ValueError('view_filter must be a nonempty known view list')
 for key in ('hard_label_indices','train_label_indices'):
  v=c[key]
  if v is None and key=='train_label_indices': continue
  if not isinstance(v,list) or any(type(i) is not int or i not in range(10) for i in v) or len(set(v))!=len(v): raise ValueError(f'{key} must contain unique indices within 0..9')
  if key=='train_label_indices' and not v: raise ValueError('train_label_indices must be nonempty or null')
 if c['final_training'] and (c['valid_manifest'] is not None or c['early_stopping_patience']): raise ValueError('final_training requires valid_manifest null and early_stopping_patience 0')
 if not c['final_training'] and c['valid_manifest'] is None: raise ValueError('valid_manifest required except for final_training')

