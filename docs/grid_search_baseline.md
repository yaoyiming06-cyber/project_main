> 状态说明（2026-10-24）：本文保留2026-10-21的历史实验说明。新的可执行配方、接口状态和Windows运行命令以 [docs/grid_interfaces.md](grid_interfaces.md) 与 [当前实验总表](experiments/imagenet_grid_draft/ImageNet训练网格汇总.csv) 为准。

# 训练网格基础文档

更新时间：2026-10-21  
适用工程：`project_main`，ConvNeXt-Base + 纯 ImageNet-22K 初始化  
历史网格来源：2026-10-19「修复版训练实验网格」，旧代码版本 `3b11ecc`，旧实验骨干 EVA-X-S

## 用途与比较边界

本文把旧版训练网格的结果和状态归档下来，并给出新 `project_main` 的实验顺序。旧实验与新工程使用不同骨干、预训练权重和训练实现，旧 AUC 只用来决定哪些方向值得验证；不能把旧 AUC 与新模型 AUC 直接排序，也不应从旧模型断点续训新架构。新工程每个实验都从 ConvNeXt-Base 的 ImageNet-22K 权重和新分类头初始化。

统一主指标为固定验证集上的十标签 macro AUC。表中 `*` 表示单病种 AUC，不能与十标签 macro AUC 比较。测试集仅在方案冻结后用于最终评测，不能用来筛选实验或阈值。验证集和数据清单应固定为 `project_main/data/manifests/` 当前版本；数据版本变化时要作为新实验批次记录。工程现已提供 `scripts/predict_test.py` 和 `scripts/export_competition_submission.py`；测试集只能在方案冻结后使用，不能用于筛选实验或阈值。

## 旧版实验结果与网格状态

旧版网格协议为 EVA-X-S、输入 224、AdamW、E0 标签策略、有效 batch 16；图中的三列 seed 为 42、2027、3407。下表照录旧 seed42 数值及截图标注的新项目状态。“无旧分数”表示图上没有结果；“待筛选补做”表示先由 seed42 筛选，再决定是否用另外两个 seed 复核。

| 实验 / 变化 | LR | 轮数 / warmup | posweight | 旧 seed42 AUC | 新工程状态（42 / 2027 / 3407） |
|---|---:|---:|---|---:|---|
| E0 原基础增强 | 1e-3 | 10 / 5 | 自动 ratio | 0.81556 | 建议重训 / 受控补做 / 待补做 |
| 仅 warmup 1 | 1e-3 | 10 / 1 | 自动 ratio | — | 待补做 / 筛选后补 / 筛选后补 |
| 仅 warmup 2 | 1e-3 | 10 / 2 | 自动 ratio | — | 待补做 / 筛选后补 / 筛选后补 |
| 仅 LR 2e-4 | 2e-4 | 10 / 5 | 自动 ratio | 0.81027 | 建议重训 / 筛选后补 / 筛选后补 |
| 仅 LR 5e-4 | 5e-4 | 10 / 5 | 自动 ratio | — | 待补做 / 筛选后补 / 筛选后补 |
| 仅 LR 8e-4 | 8e-4 | 10 / 5 | 自动 ratio | 0.80381 | 建议重训 / 筛选后补 / 筛选后补 |
| 5e-4 + 15 轮 + warmup3 | 5e-4 | 15 / 3 | 自动 ratio | 0.79798 | 建议重训 / 筛选后补 / 筛选后补 |
| E0 · 15 轮 | 1e-3 | 15 / 5 | 自动 ratio | 0.78969 | 建议重训 / 筛选后补 / 筛选后补 |
| G0 真正无随机增强 | 1e-3 | 10 / 5 | 自动 ratio | — | 先补配置 / 筛选后补 / 筛选后补 |
| G1 明确轻增强 | 1e-3 | 10 / 5 | 自动 ratio | 0.80385 | 先明确协议 / 筛选后补 / 筛选后补 |
| G2 affine | 1e-3 | 10 / 5 | 自动 ratio | 0.80878 | 建议重训 / 筛选后补 / 筛选后补 |
| G3 水平翻转 | 1e-3 | 10 / 5 | 自动 ratio | 0.80293 | 建议重训 / 筛选后补 / 筛选后补 |
| CheXpert crop | 1e-3 | 10 / 5 | 自动 ratio | 0.80384 | 建议重训 / 筛选后补 / 筛选后补 |
| 关闭 posweight（全 1） | 1e-3 | 10 / 5 | 全 1 | 0.80187 | 建议重训 / 筛选后补 / 筛选后补 |
| sqrt posweight | 1e-3 | 10 / 5 | 自动 sqrt | 0.80230 | 建议重训 / 筛选后补 / 筛选后补 |
| 三个弱标签手动加权 | 1e-3 | 10 / 5 | 手动 2/1 | 0.80375 | 建议重训 / 筛选后补 / 筛选后补 |
| E5 热启动 / 自训练 | 1e-3 | 10 / 5 | ratio / E5 | 0.80957 | 先明确协议 / 筛选后补 / 筛选后补 |
| AdaptFormer · 纵隔增宽 | 1e-3 | 10 / 5 | 单标签 BCE | 0.73229* | 必须重训 / 筛选后补 / 筛选后补 |
| AdaptFormer · 肺实变 | 1e-3 | 10 / 5 | 单标签 BCE | 0.81467* | 必须重训 / 筛选后补 / 筛选后补 |
| AdaptFormer · 肺炎 | 1e-3 | 10 / 5 | 单标签 BCE | 0.75305* | 必须重训 / 筛选后补 / 筛选后补 |
| AdaptFormer · 肺部阴影 | 1e-3 | 10 / 5 | 单标签 BCE | 0.73598* | 必须重训 / 筛选后补 / 筛选后补 |
| AP 十标签视角专家（A） | 1e-3 | 10 / 5 | 分视角 ratio | — | 待正式训 / 筛选后补 / 筛选后补 |
| PA 十标签视角专家（B） | 1e-3 | 10 / 5 | 分视角 ratio | — | 待正式训 / 筛选后补 / 筛选后补 |
| LATERAL 十标签视角专家（C） | 1e-3 | 10 / 5 | 分视角 ratio | — | 待正式训 / 筛选后补 / 筛选后补 |
| LL 十标签视角专家（A） | 1e-3 | 10 / 5 | 分视角 ratio | — | 待正式训 / 筛选后补 / 筛选后补 |
| Frontal-first | 1e-3 | 10 / 5 | 自动 ratio | 0.79517 | 暂缓 / 暂缓 / 暂缓 |
| BLR=5e-4（旧日志） | 待核实 | 10 / 5 | 自动 ratio | — | 暂缓 / 暂缓 / 暂缓 |
| Head probe（仅线性头） | 1e-3 | 5 / 1 | 待核实 | — | 暂缓 / 暂缓 / 暂缓 |

截图中的额外记录：旧 Mac seed2027 基线 AUC 为 0.80372，可保留作旧实验记录；旧版三种子完成数为 18 / 1 / 0，新工程三种子完成数为 0 / 0 / 0。该网格是计划和历史汇总，不代表新工程已经开始训练。

## 旧结果怎么解释

- 在旧版单因素 LR 对照中，1e-3 基线高于 2e-4 与 8e-4；这不足以证明新 ConvNeXt 的最优 LR 仍为 1e-3。
- 旧版 G2、G3 和 crop 的分数都低于旧版 G1 基线，但旧版增强实现和代码修复可能影响可比性；新工程先定义好变换，再做 G0/G1 对照。
- 旧版关闭 / sqrt / 手工 posweight 的结果低于自动 ratio 基线。新工程先保留自动 posweight，不把损失和增强、warmup 混在同一次比较里。
- 单标签专家的 `*` 分数只描述该病种。它们不能说明十标签整体 AUC 是否提升，也不能直接决定专家是否应融合。
- 15 轮行同时改变训练预算或 warmup，不能当作单一因素对照；新轮次实验应单独固定其它条件。

## 新工程当前基线

当前主配置为 `configs/g1_seed42.yaml`：

| 配置项 | 当前值 |
|---|---|
| 骨干 / 权重 | ConvNeXt-Base `convnext_base.fb_in22k`，纯 ImageNet-22K |
| 输入 | 224 × 224 |
| 轮数 / warmup | 10 / 5 |
| Study batch / 累积 | 2 / 8，每次更新名义上 16 个 Study |
| LR / 最小 LR | 1e-3 / 1e-6 |
| 优化器 / weight decay | AdamW / 0.05 |
| 标签策略 | E0，`-1` 忽略，空标签按 0 |
| posweight | 当前训练清单按唯一 Study 自动计算，截断至 20 |
| 计划增强 | G1；轻微仿射与亮度/对比度变化，不水平翻转 |
| 数据版本 | A_exclude2 清单，排除两张确认损坏训练图 |

按当前训练清单的 18,513 个 train Study、Study batch 2、累积 8 估算，每轮约 1,158 次优化器更新，10 轮约 11,580 次；warmup 5 / 2 / 1 对应约 5,790 / 2,316 / 1,158 次更新。学习率计划已按梯度累积后的优化器更新步数计算。

## 新工程需要的最小改造

### 1. 让增强策略由配置实际控制

`configs/g1_seed42.yaml` 的 `augmentation`、仿射、色彩变化、`horizontal_flip` 和 `normalization` 字段已经接入 `build_transform()`；训练入口会按配置构建训练/验证变换，未知策略直接报错。

训练集支持 G0、G1 和 torchvision `AutoAugmentPolicy.IMAGENET`；`normalization: dataset` 会在 resize/pad 后计算并缓存训练集 mean/std，验证始终使用确定性变换。G0 关闭所有随机增强。

相关文件：

- `src/cxr_project/dataset.py`：变换策略与配置参数。
- `src/cxr_project/train.py`：把已加载配置传给 train/valid transform 构造。
- `tests/`：验证 YAML 中的 G1 参数确实体现在实际 transform；G0 没有随机变换；可额外验证 G1 打开水平翻转开关后才出现翻转操作。
- `configs/`：增加 G0 配置，或采用明确的基线配置覆盖；实验名和输出目录必须唯一。

G0 与 G1 的对照采用相同骨干权重、数据清单、标签策略、posweight、优化器、seed、epoch、warmup、batch 和 LR，只改变训练随机增强。

### 2. 增加受控实验配置，而不是一次性穷举全网格

当前保留 `configs/g1_seed42.yaml` 和 `scripts/run_g1_seed42.ps1` 作为基线入口。三组新增消融（dataset norm、AutoAugment、Focal）和三组 posweight 对照（unit、sqrt、manual）应复制基线 YAML 后单独启动，每次使用不同的 `run_name` 和 `output_dir`。

建议配置顺序：

1. **修复并冻结 G1 定义**：确认 `G1` 使用当前 YAML 中的 `degrees=3`、`translate=0.02`、`scale=0.98–1.02`、brightness/contrast `0.08`、不翻转。
2. **新架构基线**：G1、LR 1e-3、10 轮、warmup 5、E0、自动 posweight，分别训练 seed42、2027、3407。网格图所记完成状态均为待补做；现有单配置仅是 seed42 模板，不是已经完成的模型结果。
3. **G0 vs G1**：先以 seed42 做同条件对照。若差异值得追踪，再以 seed2027、3407 重复胜出方案；不要把多个 seed 与增强因素同时改变。
4. **warmup 单因素**：固定 G1、LR 1e-3、10 轮和其它所有配置，用 seed42 分别比较 warmup1、warmup2 与 warmup5 基线。选择有一致收益且训练稳定的方案后，再补另两种 seed。
5. **再评估其它方向**：只有新基线和上述筛选完成后，才决定是否在 ConvNeXt 上重做 LR、G2、G3 或损失策略。不要把旧 EVA-X 的 AUC 当作新模型实验数据。

建议配置文件名示例：`g0_seed42.yaml`、`g1_seed2027.yaml`、`g1_seed3407.yaml`、`g1_warmup1_seed42.yaml`、`g1_warmup2_seed42.yaml`。可以复制现有 YAML 后只修改实验因素、seed、`run_name` 和 `output_dir`；后续若配置数量增加，再考虑写参数化 launcher。

### 3. 记录规则与实验判定

每次 run 应保存完整配置副本和 manifest；当前代码还记录训练/验证清单 SHA-256、预训练权重 SHA-256、Git revision、损失、posweight 和归一化信息。不得复用输出目录。重跑实验要换新的 run 名称，并注明重跑原因。

筛选时先看十标签 macro AUC，再看三种子均值和标准差、各标签变化与训练稳定性。单个 seed 的小幅提升只作候选，不称为稳定提升；需要 seed42 筛选的候选，使用 2027 和 3407 做受控复核。相同验证集上的很多轮筛选会逐渐过拟合验证集，所以先按上述少量单因素对照推进。

新模型 valid AUC 必须在新工程相同代码、数据和指标口径下比较。最终配置冻结后再评估比赛 test；不要根据 test 分数挑配置。旧版本基线、旧 Mac seed2027 基线、单病种 AUC 与新工程十标签 macro AUC 分开归档。

## 开始实验前检查

- [ ] `project_main` 中 G0/G1 实际变换已由配置控制，G0 测试确认无随机操作。
- [ ] G1 的运行 transform 与配置字段一致；水平翻转默认关闭。
- [ ] train/valid/test 使用同一份锁定的数据清单版本，test 不进入训练或筛选。
- [ ] ConvNeXt-Base 权重文件 SHA-256 校验通过，模型标识为 `convnext_base.fb_in22k`。
- [ ] 每个 seed、增强、warmup 组合使用唯一输出目录。
- [ ] 开跑前记录 run 配置、数据版本、代码版本和预期 seed。
