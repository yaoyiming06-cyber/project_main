# Grid interfaces implementation plan

Goal: make each current grid experiment executable through validated configuration or a dedicated evaluation command.

Architecture: preserve old ConvNeXt defaults and checkpoints, add explicit opt-in model/data interfaces, and generate independent per-seed YAML from a frozen stage baseline. Evaluation tools consume aligned predictions and never fit on test.

Tech stack: PyTorch, timm, YAML, NumPy, sklearn, Pillow, OpenCV.

- [x] Model: optional experts/dropout/pooling/view embedding, partial freezing, nonmedical alternate backbones, Transformer adapters. Test gradients, state compatibility, and shape behavior.
- [x] Data/loss: deterministic G0, gamma/erasing/CLAHE, masked Study feature MixUp, ASL, train-only weighted sampling, view coverage. Test missing-label and mixed-view cases.
- [x] Training: validate all fields, skip EMA/schedule on skipped AMP updates, raw/EMA evaluation, checkpoint selection, resume state, final fixed-budget training. Test minimal synthetic training paths.
- [x] Tools: stage/seed configuration generation, folds, bootstrap, fusion/routing, compact export, fixed TTA, final manifest merge. Test alignment and leakage guards.
- [x] Integration: run existing and new tests, verify generated YAML, update README and interface reference. Keep local changes without commit/push; do not launch competition training.

验证记录：73项 unittest 全部通过；58个训练配方 × 3 seeds 共174份配置通过正式 load_config；src/scripts/tests compileall 通过；未启动真实比赛训练，未提交或推送 Git。
