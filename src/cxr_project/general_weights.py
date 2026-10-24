"""Explicit allowlist of general-purpose timm ImageNet weight sources."""
SOURCES = {
 'convnext_base.fb_in22k': 'ImageNet-22K',
 'convnext_small.fb_in22k': 'ImageNet-22K',
 'swin_base_patch4_window7_224.ms_in22k': 'ImageNet-22K',
 'vit_base_patch16_224.augreg_in21k': 'ImageNet-21K',
}

def weight_source(model):
    if model not in SOURCES: raise ValueError(f'unsupported general-pretrained model: {model}')
    return {'repo_id':f'timm/{model}','dataset':SOURCES[model],'filename':'model.safetensors'}
