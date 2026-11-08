"""Explicit allowlist of general-purpose pretrained weight sources."""
DINOV2_MODEL = 'vit_base_patch14_dinov2'
DINOV2_URL = (
    'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/'
    'dinov2_vitb14_pretrain.pth'
)
SOURCES = {
 'convnext_base.fb_in22k': 'ImageNet-22K',
 'convnext_small.fb_in22k': 'ImageNet-22K',
 'swin_base_patch4_window7_224.ms_in22k': 'ImageNet-22K',
 'vit_base_patch16_224.augreg_in21k': 'ImageNet-21K',
}

def weight_source(model):
    if model == DINOV2_MODEL:
        return {
            'model': DINOV2_MODEL,
            'dataset': 'LVD-142M general-purpose pretraining',
            'filename': 'dinov2_vitb14_pretrain.pth',
            'url': DINOV2_URL,
            'download': 'official-url',
        }
    if model not in SOURCES: raise ValueError(f'unsupported general-pretrained model: {model}')
    return {'repo_id':f'timm/{model}','dataset':SOURCES[model],'filename':'model.safetensors'}
