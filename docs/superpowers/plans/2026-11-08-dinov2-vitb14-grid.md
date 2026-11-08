# DINOv2 ViT-B/14 Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add verified official DINOv2 ViT-B/14 supervised fine-tuning support and generate the approved 21-run staged LR/resolution grid with FP16 EMA inference exports.

**Architecture:** Reuse the existing timm VisionTransformer backbone wrapper, Study-level classifier, training loop, EMA, validation and export path. Add a DINOv2-specific official-checkpoint downloader/provenance validator and a small grid generator with separate LR-selection and resolution-comparison stages. Config generation and training execution are separate CLI commands.

**Tech Stack:** Python 3.11-3.12, PyTorch, timm 1.0.30, PyYAML, unittest, existing CUDA AMP training and evaluation tools.

---

### Task 1: Pin and verify the official DINOv2 checkpoint

**Files:**
- Create: `src/cxr_project/dinov2_weights.py`
- Modify: `src/cxr_project/general_weights.py`
- Create: `scripts/download_dinov2_weights.py`
- Create: `tests/test_dinov2_weights.py`

- [x] **Step 1: Add failing tests for checkpoint source and integrity checks**

Test that the only DINOv2 source is the official `dinov2_vitb14_pretrain.pth` URL; a file without its `.source.json` record, a wrong URL, a wrong byte count, a digest mismatch, and an invalid configured SHA-256 must each fail. A valid temporary checkpoint plus a matching provenance record must pass.

- [x] **Step 2: Run the focused tests and confirm they fail**

Run: `.venv-mac/bin/python -m unittest discover -s tests -p 'test_dinov2_weights.py' -v`

Expected: FAIL because the validator and official source entry do not exist yet.

- [x] **Step 3: Implement atomic download and local provenance validation**

Define the official URL as `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth`. Download to a sibling `.part` file, compute byte size and SHA-256, require a non-empty PyTorch checkpoint, atomically replace the destination, and write `<checkpoint>.source.json` with model name, source URL, size, SHA-256, and resolved path. Validation must require the sidecar and configured SHA-256 and verify all values against the current file; never download while constructing a model.

- [x] **Step 4: Re-run the focused tests**

Run: `.venv-mac/bin/python -m unittest tests.test_dinov2_weights -v`

Expected: PASS, including all missing/mismatched provenance cases.

### Task 2: Build DINOv2 through the existing Study classifier

**Files:**
- Modify: `src/cxr_project/model.py`
- Modify: `src/cxr_project/config.py`
- Modify: `src/cxr_project/general_weights.py`
- Create: `tests/test_dinov2_model.py`

- [x] **Step 1: Add failing model/config tests**

Test that `vit_base_patch14_dinov2` is allowlisted, that pretrained construction refuses missing or invalid local provenance before loading model weights, that non-pretrained construction builds the expected 768-feature backbone and ten-label Study head, and that DINOv2 input sizes not divisible by 14 are rejected. Test that input sizes 224, 252, and 280 load and produce finite `[batch, 10]` logits at the classifier boundary.

- [x] **Step 2: Run the focused tests and confirm they fail**

Run: `.venv-mac/bin/python -m unittest discover -s tests -p 'test_dinov2_model.py' -v`

Expected: FAIL because the model is not registered and DINOv2 config validation is absent.

- [x] **Step 3: Implement strict local pretrained construction**

Register `vit_base_patch14_dinov2` as the official general-purpose LVD-142M model. Validate its checkpoint with `dinov2_weights.py`, then call `timm.create_model` with `pretrained=True`, `num_classes=0`, `dynamic_img_size=True`, and `pretrained_cfg_overlay={'file': local_path}` so timm 1.0.30 applies its DINOv2 checkpoint filter. Pass the backbone through `ConvNeXtStudyClassifier` with `feature_dim=backbone.num_features`; do not change the existing classifier state format. Require the explicit checkpoint SHA-256 in config and keep the loader network-free.

- [x] **Step 4: Enforce patch-aligned DINOv2 sizes and preserve pretrained normalization**

In `load_config`, when model is `vit_base_patch14_dinov2`, reject `input_size % 14 != 0`. Set DINOv2 grid configs to `normalization: pretrained`; retain the existing timm normalization lookup, which resolves to mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.

- [x] **Step 5: Re-run focused tests**

Run: `.venv-mac/bin/python -m unittest discover -s tests -p 'test_dinov2_*.py' -v`

Expected: PASS for model construction, patch-size validation, logits, and checkpoint provenance.

### Task 3: Generate the staged 21-run grid

**Files:**
- Create: `configs/dinov2_vitb14_baseline.yaml`
- Create: `src/cxr_project/dinov2_grid.py`
- Create: `scripts/dinov2_grid.py`
- Create: `tests/test_dinov2_grid.py`

- [x] **Step 1: Add failing tests for both grid stages**

Test stage 1 produces exactly 15 unique configs for 5 learning rates (`2.5e-5`, `5e-5`, `1e-4`, `2e-4`, `4e-4`) × seeds (42, 2027, 3407), all at size 252. Test stage 2 produces exactly 6 configs for sizes 224 and 280 × the same seeds at one explicitly selected stage-1 learning rate. Assert every generated config preserves the baseline's 15 epochs, 5 warmup epochs, `min_lr=1e-6`, EMA `0.9998`, EMA validation selection, batch 16, accumulation 1, image chunk 2, G1, Focal gamma 2, ratio/cap20, no view input/filter, no clipping, and no early stopping. Assert unique output paths and embedded checkpoint SHA-256.

- [x] **Step 2: Run the focused tests and confirm they fail**

Run: `.venv-mac/bin/python -m unittest discover -s tests -p 'test_dinov2_grid.py' -v`

Expected: FAIL because the baseline and DINOv2 grid commands do not exist.

- [x] **Step 3: Implement an explicit two-stage config generator**

Create a `stage1` command that writes the 15 LR configs and manifest. Create a `stage2` command that reads all fifteen completed run manifests, picks the best LR by three-seed mean, then writes the 6 resolution configs. Both commands must resolve the checkpoint and data paths, verify the local weight provenance, avoid overwriting non-empty config/output directories, validate every YAML using `load_config`, and never start training. A separate `run` command verifies generated config hashes and executes the stage sequentially, exporting best EMA weights as FP16 after each successful run. Record each config path, SHA-256, grid factors, checkpoint provenance and exact commands in `grid_manifest.json`.

- [x] **Step 4: Add the baseline and run generation tests**

Copy the verified S5-L5 experiment factors into the DINOv2 baseline, changing only model/checkpoint, LR placeholder, patch-aligned input size, and pretrained normalization. Keep effective Study batch 16 (`batch_size_studies: 16`, `accum_steps: 1`), image chunk 2, 15 epochs and 5 warmup epochs. Then run `.venv-mac/bin/python -m unittest discover -s tests -p 'test_dinov2_grid.py' -v` and expect PASS with 15 and 6 configs.

### Task 4: Verify training and FP16 export compatibility

**Files:**
- Create: `tests/test_dinov2_export.py`
- Modify: `README.md`

- [x] **Step 1: Add failing runtime/export coverage**

Test DINOv2 config normalization reaches the training transform, a synthetic validation forward yields ten logits, and `export_fp16` stores the selected EMA state as model-only FP16 weights while retaining config and normalization metadata and omitting optimizer state.

- [x] **Step 2: Run focused runtime/export tests**

Run: `.venv-mac/bin/python -m unittest discover -s tests -p 'test_grid_*.py' -v`

Expected: the new DINOv2-specific assertions fail before the documentation and test updates.

- [x] **Step 3: Document Windows CUDA setup and staged commands**

Add the official DINOv2 checkpoint URL, the downloader command, stage-1 generation command, the rule for choosing best LR from three-seed validation macro AUC, stage-2 generation command, sequential training command, and per-run FP16 export command. State clearly that FP16 is export-only and that competition test labels, ViewPosition and clinical fields are not read or used for selection.

- [x] **Step 4: Run focused and full regression tests**

Run: `.venv-mac/bin/python -m unittest discover -s tests -p 'test_dinov2_*.py' -v`, then `.venv-mac/bin/python -m unittest discover -s tests -v`.

Expected: all focused tests and the existing suite pass. Run a synthetic forward on MPS if available; do not start the 15 full training runs on this Mac. Full training is intended for Windows/CUDA.
