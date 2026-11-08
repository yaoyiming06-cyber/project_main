# DINOv2 ViT-B/14 Fine-Tuning Grid

## Goal

Compare three learning rates and then three patch-aligned input resolutions for the original, general-purpose DINOv2 ViT-B/14 checkpoint. Retain a compact FP16 EMA inference export for each run. This is supervised fine-tuning, not DINO self-supervised continuation.

## Experiment Grid

Use the official `vit_base_patch14_dinov2` model and its LVD-142M general pretraining checkpoint. Run the grid in two stages, with seeds 42, 2027, and 3407:

| Stage | Input size | Learning rate | Seeds | Runs |
| --- | ---: | --- | --- | ---: |
| LR selection | 252 | 0.00005, 0.00010, 0.00020 | 42, 2027, 3407 | 9 |
| Resolution comparison | 224, 280 | Best LR from stage 1 | 42, 2027, 3407 | 6 |

This produces fifteen independent runs, each initialized from the same pinned DINOv2 weights. Select the stage-1 learning rate by the mean of its three fixed validation macro AUC scores. Then compare 224, 252, and 280 at that learning rate by their three-seed validation means. Report each seed score, sample standard deviation, per-label AUC, and best EMA epoch. The held-out competition test data is not read or used for selection. All three resolutions are divisible by the 14-pixel patch size.

## Fixed Training Protocol

Train on the A_exclude2 train manifest and select EMA checkpoints on its fixed valid manifest. The current config schema requires a `test_manifest` path, but the training path must not open or read that manifest. Keep the verified S5-L5 training settings unchanged except for the backbone, input size, and learning rate:

- 15 epochs, 5 warmup epochs, AdamW, linear warmup followed by cosine decay to minimum LR `1e-6`.
- Weight decay `0.10`, layer decay `0.65`, drop path `0.10`.
- Focal loss, gamma `2`, ratio positive weights capped at `20`.
- G1 augmentation, ImageNet normalization, no view embedding or view filtering.
- EMA `0.9998`, updated from epoch 1; EMA-only validation selection, no gradient clipping, early stopping disabled.
- Study batch `16`, accumulation `1`, image chunk size `2`, CUDA AMP enabled, gradient checkpointing enabled; keep the remaining S5-L5 data and optimizer settings unchanged.

Only learning rate changes within stage 1. In stage 2, only resolution changes. The model is trained with the normal CUDA AMP path and full-precision optimizer state. After each run, export the selected best EMA model-only weights in FP16 using the existing export format. Keep full training checkpoints and validation predictions for reproducibility; do not treat the FP16 export as FP16 training.

## Weight and Model Loading

Add an explicit DINOv2 entry for timm's `vit_base_patch14_dinov2` and support the official local `.pth` file without network fallback during model construction. The downloader records the exact source URL, byte size, and SHA-256; loading rejects a missing or mismatched file. Use the timm model's pinned DINOv2 normalization metadata. Do not accept XRay-DINO, Cell-DINO, or other medical variants.

Official source: [DINOv2 ViT-B/14 checkpoint](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth). The project research record identifies the official code and model weights as Apache-2.0 and the pretraining corpus as general LVD-142M data.

## Implementation and Validation

1. Add strict model construction and provenance validation for the official checkpoint.
2. Add a DINOv2 baseline config and staged recipes that generate all fifteen run configs without starting training by default.
3. Reuse the existing train, EMA, validation, and FP16 export paths.
4. Add focused tests for checkpoint verification, DINOv2 model construction and output shapes, config generation, and FP16 export compatibility.
5. Run synthetic CPU/MPS checks locally. Run the fifteen full training jobs on Windows CUDA; this Mac host has no CUDA device.

## Constraints

Do not read competition test labels, ViewPosition, or clinical fields. Do not use external medical data. Do not commit or push training outputs. The grid is exploratory; no performance improvement is assumed before the three-seed validation results are available.
