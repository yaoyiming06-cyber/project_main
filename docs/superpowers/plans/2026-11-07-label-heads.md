# 标签注意力与空间查询头实施计划

**目标：** 在现有 Windows/CUDA、Mac/MPS 工程中加入第一批 C0/C1/H1/H2/H3 配置和推理兼容，不启动正式训练。

**执行方式：** 本地逐项执行；所有 Git 提交和推送均留待用户明确要求。已有未提交修改保留；实施前涉及的原文件快照位于 `/private/tmp/cxr_label_heads_20261107_before`。

**设计：** 见 `docs/superpowers/specs/2026-11-07-label-heads-design.md`。PyTorch 原生模块，不增加第三方依赖。

## 1. 测试先行

- [x] 新建 `tests/test_label_heads.py`：验证独立标签图像权重、单图/多图顺序不变性、查询使用空间分布、梯度和 checkpoint round trip。
- [x] 新建配置/配方/推理错误检查：不允许视角、专家、MixUp 与新结构混用；query参数检查；旧缺省兼容。
- [x] 运行 `.venv-mac/bin/python -m unittest discover -s tests -p 'test_label_heads.py' -v`，确认新增功能未实现的预期失败。

## 2. 模型与配置

- [x] `src/cxr_project/pooling.py`：独立标签注意力，只在每个 Study 内归一化。
- [x] `src/cxr_project/heads.py`：逐标签分类与空间 cross-attention 查询；只用图像 tokens。
- [x] `src/cxr_project/model.py`：保留旧模块布局，新增两种特征路径与对应聚合。
- [x] `grid_options.py`、`config.py`、`grid.py`：默认/严格校验、五组配方、统一训练预算；新空间头只允许 ConvNeXt。

## 3. 恢复和推理

- [x] `train.py`：恢复校验补齐新字段缺省，同时拒绝改变新结构后恢复。
- [x] `scripts/predict_test.py`：检查新结构字段；competition_inference/predict/cost沿用同一模型构建，不复制模型逻辑。
- [x] 真实合成Study训练一步、保存EMA/恢复、预测CSV；不使用比赛图像或测试标签。

## 4. 交付

- [x] 新增完整跨平台 `configs/head_baseline_l5.yaml`，以现有 `grid_search.py` 生成独立输出路径。
- [x] README与接口文档补充配方、命令、规则与官方权重链接；明确DINOv2尚未接入。
- [x] 运行相关旧测试、新测试、全套回归；CPU/MPS验证及旧模型严格加载。
- [x] 独立代码审查、修复发现的问题、报告验证边界。只保留本地文件。

完成记录（2026-11-08）：108项完整回归通过；另完成真实本地预训练Base前向/反向、Windows/Mac各15份CLI配置校验及独立代码审查。详细边界见 `docs/experiments/标签头接口验收_20261108.md`。未正式训练、未提交或推送Git。
