# CXR 比赛训练工程（Windows / CUDA）

这是独立的 Windows 训练工程，使用**纯 ImageNet-22K 通用预训练权重**初始化 ConvNeXt-Base，再训练比赛的 10 标签 Study 分类头和困难标签专家头。骨干为 `timm` 模型 `convnext_base.fb_in22k`；没有使用医学模型或医学预训练权重，也没有采用 ImageNet-22K 之后再经 ImageNet-1K 微调的 `fb_in22k_ft_in1k` 版本。

当前工程只包含训练和数据清单校验流程。它尚未在 Windows CUDA 机器上完成正式训练；训练完成后会产生验证结果和 checkpoint。比赛测试集预测与 submission 导出不在本工程当前入口内。

## 工程结构

```text
project_main/
├── configs/g1_seed42.yaml          # 本次完整训练配置
├── data/manifests/                 # train / valid / test CSV 清单
├── scripts/
│   ├── setup_windows_cuda.ps1      # 创建虚拟环境并安装 CUDA PyTorch
│   ├── validate_data.py            # 校验图片路径、清单和 split 泄漏
│   ├── download_weights.py          # 下载并校验 ImageNet-22K 权重
│   └── run_g1_seed42.ps1            # 启动完整训练
├── src/cxr_project/                # 数据、模型、损失、训练代码
├── tests/                           # 工程合同与回归测试
└── weights/                         # 下载后的本地权重（不纳入 Git）
```

图片没有复制到工程。Windows 磁盘上请把 `project_main` 与原始比赛图片目录 `cxr_first` 放在同一父目录：

```text
<比赛目录>/
├── cxr_first/       # 包含 train/、valid/、test/ 图片
└── project_main/    # 本工程
```

如果图片目录位置不同，运行数据校验或训练时用 `--data-root` 指定图片根目录。

## 数据清单

`data/manifests/` 中的 CSV 来自质量复核后的 **A_exclude2** 数据版本，清单里的 `image_path` 相对 `cxr_first`。训练使用全部 Study 下的全部视角图像，并按 Study 分组计算标签。

| Split | 图片 | Study | 用途 |
| --- | ---: | ---: | --- |
| train | 30,342 | 18,513 | 训练 |
| valid | 1,422 | 859 | 每轮验证和选择最佳 checkpoint |
| test | 2,218 | 1,414 | 留作后续预测；训练入口不会读取 |

训练清单已排除两张确认损坏图片：

- `train/subject_01922/study_02467/pic_004044.jpg`
- `train/subject_05239/study_13172/pic_021690.jpg`

启动训练前，校验脚本会确认清单里的图片都存在、排除图片没有进入训练集，以及 train/valid/test 之间没有 Subject、Study 或图像交叉。

## 模型与训练配置

ConvNeXt-Base 从 ImageNet-22K 预训练权重初始化。该分类 checkpoint 原有 21,841 类分类头会被移除，换成比赛任务的结构：10 标签主头、视角嵌入注意力 Study 聚合，以及为纵隔增宽、肺炎、肺部阴影设置的三个困难标签专家头。主头与专家头的对应 logit 在预测时取平均，专家训练同时使用辅助损失。

配置沿用 G1 seed42 训练节奏：

| 项目 | 设置 |
| --- | --- |
| 输入 | 224 × 224 |
| Epoch / warmup | 10 / 5 |
| Study batch / 梯度累积 | 2 / 8（名义上每次更新 16 个 Study） |
| 优化器 | AdamW；LR `1e-3`，min LR `1e-6`，weight decay `0.05` |
| Backbone layer decay / drop path | `0.75` / `0.1` |
| 标签策略 | E0：显式 `-1` 忽略，空标签按 0 |
| 正样本权重 | 按清洗后训练集的唯一 Study 自动计算，最大值截断为 20 |
| 增强 | G1：轻微旋转、平移、缩放、亮度和对比度变化；不水平翻转 |

Warmup 和余弦学习率按梯度累积后的优化器更新步数计算。正样本权重会从当前训练清单重新计算，不沿用其它数据版本上的旧数值。

## Windows 环境准备

需要 64 位 Python 3.11 或 3.12、可用的 NVIDIA GPU 和兼容的 NVIDIA 驱动。请在 PowerShell 中进入工程目录并运行：

```powershell
cd <比赛目录>\project_main
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_cuda.ps1
```

安装脚本会在工程内创建 `.venv`，安装 PyTorch 2.8.0 / torchvision 0.23.0 的 CUDA 12.8 wheels、工程依赖，并检查 CUDA 是否可用。CUDA 运行时随 PyTorch wheels 提供，通常无需另装完整 CUDA Toolkit；NVIDIA 驱动仍需正常工作。

如果环境检查提示 CUDA 不可用，先运行 `nvidia-smi` 确认驱动和显卡可见，再重新运行检查：

```powershell
.\.venv\Scripts\python.exe .\scripts\check_environment.py
```

## 校验数据并准备权重

从 `project_main` 目录运行：

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_data.py --data-root ..\cxr_first
.\.venv\Scripts\python.exe .\scripts\download_weights.py
```

权重来自 [timm 的 `convnext_base.fb_in22k` 模型卡](https://huggingface.co/timm/convnext_base.fb_in22k)，下载地址固定到模型 revision `9afd4146adabf1b6e861d0f1ac2e7baa7c8dd1b7`。文件保存为 `weights/convnext_base.fb_in22k.safetensors`，大小为 439,848,696 字节。下载脚本和训练加载过程都会校验 SHA-256：

```text
0f09df05894c7fd1e78569daaae9e839a7b3bc5666587a08ba1d8e8cea63a8a4
```

首次下载需要网络并会下载约 440 MB；中断后再次运行下载命令即可重新下载。权重文件已加入 `.gitignore`，需要在训练用的 Windows 磁盘上运行该命令获取。

## 启动完整训练

先确保数据校验和权重下载成功，再运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_g1_seed42.ps1
```

当前图片目录若不在 `project_main` 同级位置，可传入实际路径：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_g1_seed42.ps1 `
  -DataRoot 'D:\project\race_dataset\cxr_first\cxr_first'
```

也可以直接调用训练模块：

```powershell
.\.venv\Scripts\python.exe -m cxr_project.train `
  --config .\configs\g1_seed42.yaml `
  --data-root ..\cxr_first `
  --device cuda
```

训练会运行完整 10 个 epoch，逐轮计算 valid AUC，并把结果写入 `runs/G1_seed42_convnext_base_in22k/`：

- `checkpoint-last.pth`：最近一轮 checkpoint，用于续训。
- `checkpoint-best.pth`：valid macro AUC 最佳 checkpoint。
- `epoch_metrics.csv`：逐轮总 AUC 和每个标签的 AUC。
- `valid_best_predictions.csv`：最佳 checkpoint 对验证 Study 的预测。
- `run_manifest.json`：解析后的训练配置、数据统计、权重 SHA-256 和训练状态。

训练结束时，`run_manifest.json` 的 `status` 应为 `completed`。终端会逐轮打印 `train_loss`、`val_macro_auc` 和当前最佳分数。

### 断点续训

若进程中断，使用最近的 `checkpoint-last.pth`，并显式指定原输出目录：

```powershell
.\.venv\Scripts\python.exe -m cxr_project.train `
  --config .\configs\g1_seed42.yaml `
  --data-root ..\cxr_first `
  --device cuda `
  --resume .\runs\G1_seed42_convnext_base_in22k\checkpoint-last.pth `
  --output-dir .\runs\G1_seed42_convnext_base_in22k
```

续训 checkpoint 会恢复模型、优化器、GradScaler、epoch 和更新步数；使用相同输出目录可以延续原实验记录。checkpoint 每个 epoch 结束时保存，因此中途退出会从最近完成的 epoch 继续，未完成的 epoch 会重跑。当前 checkpoint 不保存随机数生成器状态，所以续训不是逐 batch 的精确复现。

### 显存不足时

默认每个优化器更新对应约 16 个 Study（batch 2 × 累积 8）。如果需要降低单个 micro-batch 的显存占用，在 `configs/g1_seed42.yaml` 中把 `batch_size_studies` 改为 `1`，并把 `accum_steps` 改为 `16`，以尽量保持每次更新的有效 Study 数和每轮更新次数。只改 batch 而不相应增加累积步数，会改变有效 batch 和训练更新数。

## 本机代码检查

在已安装工程依赖的环境中运行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
```

Mac 可用于代码、权重文件和清单检查，但本工程的正式训练命令要求 Windows 上可见的 CUDA GPU。

---

## 网格搜索历史总结与新项目计划

更新时间：2026-10-21  
适用工程：`project_main`，ConvNeXt-Base + 纯 ImageNet-22K 初始化  
历史网格来源：2026-10-19「修复版训练实验网格」，旧代码版本 `3b11ecc`，旧实验骨干 EVA-X-S

## 用途与比较边界

本文把旧版训练网格的结果和状态归档下来，并给出新 `project_main` 的实验顺序。旧实验与新工程使用不同骨干、预训练权重和训练实现，旧 AUC 只用来决定哪些方向值得验证；不能把旧 AUC 与新模型 AUC 直接排序，也不应从旧模型断点续训新架构。新工程每个实验都从 ConvNeXt-Base 的 ImageNet-22K 权重和新分类头初始化。

统一主指标为固定验证集上的十标签 macro AUC。表中 `*` 表示单病种 AUC，不能与十标签 macro AUC 比较。测试集仅在方案冻结后用于最终评测，不能用来筛选实验或阈值。验证集和数据清单应固定为 `project_main/data/manifests/` 当前版本；数据版本变化时要作为新实验批次记录。本工程目前没有比赛测试集预测或 submission 导出入口，因此测试集规则是后续评测流程要求，不代表当前训练入口已能执行测试预测。

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
| posweight | 当前训练清单按唯一 Study 自动计算 ratio，截断至 20 |
| 损失 | 加权 BCE；可切换 Focal，`gamma=2.0` |
| 计划增强 | G1；轻微仿射与亮度/对比度变化，不水平翻转；可切换 AutoAugment |
| 归一化 | 默认 ImageNet-1K 统计；可切换为训练清单统计值 |
| 数据版本 | A_exclude2 清单，排除两张确认损坏训练图 |

按当前训练清单的 18,513 个 train Study、Study batch 2、累积 8 估算，每轮约 1,158 次优化器更新，10 轮约 11,580 次；warmup 5 / 2 / 1 对应约 5,790 / 2,316 / 1,158 次更新。学习率计划已按梯度累积后的优化器更新步数计算。

## 新增消融与对照实验

下面七组实验都使用同一份 `A_exclude2` 清单、ConvNeXt-Base ImageNet-22K 权重、seed42、输入 224、10 epoch、warmup 5、Study batch 2、累积 8、AdamW、LR `1e-3`、weight decay `0.05`、layer decay `0.75`、drop path `0.1` 和 E0 标签口径。每组都必须使用唯一的 `run_name` 和 `output_dir`，从同一份预训练权重独立初始化；只有表中加粗因素改变。

| 组别 | 实验名 | 归一化 | 增强 | 损失 | posweight | 目的 |
|---|---|---|---|---|---|---|
| 基线 | `G1_seed42_convnext_base_in22k` | ImageNet-1K | G1 | 加权 BCE | ratio | 新 ConvNeXt 工程基准 |
| 消融 1 | `E0_dataset_norm_seed42` | **训练集 mean/std** | G1 | 加权 BCE | ratio | 检查数据集统计是否优于预训练统计 |
| 消融 2 | `E0_autoaug_seed42` | ImageNet-1K | **AutoAugment / ImageNet policy** | 加权 BCE | ratio | 检查更强增强的泛化影响 |
| 消融 3 | `E0_focal_seed42` | ImageNet-1K | G1 | **Focal，gamma=2.0** | ratio | 检查难例聚焦是否改善弱标签表现 |
| 对照 1 | `E0_unit_posweight_seed42` | ImageNet-1K | G1 | 加权 BCE | **全 1** | 去除类别不平衡加权 |
| 对照 2 | `E0_sqrt_posweight_seed42` | ImageNet-1K | G1 | 加权 BCE | **sqrt(ratio)** | 减弱自动 ratio 的幅度 |
| 对照 3 | `E0_manual_posweight_seed42` | ImageNet-1K | G1 | 加权 BCE | **2,1,1,2,1,1,1,2,1,1** | 复核三个困难标签的手动弱加权 |

这里的四组“损失/标签策略”包含 E0 ratio 基线、全 1、sqrt(ratio) 和手动权重四组；前三个新增损失/归一化/增强实验都只改变一项。`AutoAugment` 使用 torchvision 的 `AutoAugmentPolicy.IMAGENET`，它是探索性对照，不能直接当作医学图像的默认增强。数据集归一化统计在训练清单图像经过 `ResizePadToSquare(224)` 和 0 填充后计算，保存在该 run 的 `dataset_normalization.json`；缓存记录训练清单 SHA-256、图片根目录和输入尺寸，续训会检查来源一致后复用。旧版本无来源记录的缓存不能安全续用，应该作为新 run 重新计算归一化并从预训练权重开始。

每组的配置覆盖如下；未列出的字段沿用 `configs/g1_seed42.yaml`：

```yaml
# E0_dataset_norm_seed42
run_name: E0_dataset_norm_seed42
output_dir: runs/E0_dataset_norm_seed42
normalization: dataset

# E0_autoaug_seed42
run_name: E0_autoaug_seed42
output_dir: runs/E0_autoaug_seed42
augmentation: AutoAug

# E0_focal_seed42
run_name: E0_focal_seed42
output_dir: runs/E0_focal_seed42
loss: focal
focal_gamma: 2.0

# E0_unit_posweight_seed42
run_name: E0_unit_posweight_seed42
output_dir: runs/E0_unit_posweight_seed42
pos_weight_mode: unit
manual_pos_weights: null

# E0_sqrt_posweight_seed42
run_name: E0_sqrt_posweight_seed42
output_dir: runs/E0_sqrt_posweight_seed42
pos_weight_mode: sqrt
manual_pos_weights: null

# E0_manual_posweight_seed42
run_name: E0_manual_posweight_seed42
output_dir: runs/E0_manual_posweight_seed42
pos_weight_mode: manual
manual_pos_weights: [2, 1, 1, 2, 1, 1, 1, 2, 1, 1]
```

运行顺序建议先完成 `E0_seed42`，再依次执行三组消融，最后执行三组 posweight 对照。主指标仍是固定 valid 上的十标签 macro AUC；只有 seed42 筛选出候选后，才用 seed2027 和 3407 做复核。不要用 test 结果挑选配置，也不要把旧 EVA-X-S 的 AUC 与此 ConvNeXt 工程直接排序。

## 参数复核结论

- `AdamW`、layer decay `0.75`、drop path `0.1`、有效 Study batch `16` 和 5 epoch warmup 是合理的首轮迁移学习起点，适合先冻结协议做基线。
- 全量微调 ConvNeXt-Base 时 LR `1e-3` 偏激进，它沿用了旧 EVA-X-S 网格，不能预先视为新骨干最优值。若基线出现 loss 爆炸、AUC 早期下降或梯度频繁裁切，应优先补做 LR `2e-4` 或 `5e-4`，并保持其它条件不变。
- 当前 `clip_grad: null` 表示不裁切；这适合先观察原始训练稳定性。若出现非有限 loss 或梯度尖峰，可单独设 `clip_grad: 1.0` 做稳定性对照，但不能把裁切和其它消融同时改变。
- `batch_size_studies: 2`、`accum_steps: 8` 在显存不足时可以改成 `1/16`，尽量保持有效 batch 和每轮更新次数；`image_chunk_size: 2` 与 gradient checkpointing 已经承担主要显存控制。
- ImageNet-1K 统计是当前预训练权重的匹配输入，数据集 mean/std 和 AutoAugment 都是待验证的消融项；它们不应默认替换基线。

## 当前实现状态

`build_transform()` 已由配置驱动：G0 关闭随机增强，G1 读取 YAML 中的仿射、色彩和翻转参数，AutoAugment 使用 ImageNet policy；验证始终不使用随机增强。训练入口支持 `normalization: dataset`、`loss: focal`、`pos_weight_mode: ratio/unit/sqrt/manual`，并把清单 SHA-256、Git revision、工作区是否有改动、已跟踪 diff SHA-256、实际权重、损失和归一化信息写入 `run_manifest.json`。未跟踪文件路径也会被记录。缓存来源变化、未知策略、缺失手动权重、NaN/Inf 和其它非法数值会在启动时直接报错。

## 后续实验与记录清单

### 1. 配置入口已冻结

上述配置入口已经接入训练；后续改造重点是完成七组实验并保存每组的 run manifest，而不是继续扩大一次性网格。

`build_transform` 接收已加载配置，训练集按 `augmentation` 选择 G0、G1 或 AutoAugment，并按 `normalization` 选择 ImageNet 或缓存的数据集统计；验证变换始终确定性。遇到未知策略会直接报错。

相关文件：

- `src/cxr_project/dataset.py`：变换策略与配置参数。
- `src/cxr_project/train.py`：计算/缓存数据集统计，解析损失和 posweight 策略，并写入实验指纹。
- `tests/`：覆盖 G0/G1/AutoAugment、数据集统计、Focal mask 和四种 posweight 策略。
- `configs/`：复制基线后只修改单个实验因素；实验名和输出目录必须唯一。

G0 与 G1 的对照采用相同骨干权重、数据清单、标签策略、posweight、优化器、seed、epoch、warmup、batch 和 LR，只改变训练随机增强。

### 2. 增加受控实验配置，而不是一次性穷举全网格

当前保留 `configs/g1_seed42.yaml` 和 `scripts/run_g1_seed42.ps1` 作为基线入口。七组实验先复制基线 YAML 后单独启动，每次设置不同的 `run_name` 和 `output_dir`，防止结果覆盖。

建议配置顺序：

1. **冻结 G1 定义**：`degrees=3`、`translate=0.02`、`scale=0.98–1.02`、brightness/contrast `0.08`、不翻转。
2. **新架构基线**：G1、LR 1e-3、10 轮、warmup 5、E0、自动 posweight，分别训练 seed42、2027、3407。网格图所记完成状态均为待补做；现有单配置仅是 seed42 模板，不是已经完成的模型结果。
3. **G0 vs G1**：先以 seed42 做同条件对照。若差异值得追踪，再以 seed2027、3407 重复胜出方案；不要把多个 seed 与增强因素同时改变。
4. **warmup 单因素**：固定 G1、LR 1e-3、10 轮和其它所有配置，用 seed42 分别比较 warmup1、warmup2 与 warmup5 基线。选择有一致收益且训练稳定的方案后，再补另两种 seed。
5. **完成本页七组实验后再扩展**：只有新基线、三组消融和三组 posweight 对照完成后，才决定是否在 ConvNeXt 上重做 LR、G2、G3、其它 Focal gamma 或视角专家。不要把旧 EVA-X 的 AUC 当作新模型实验数据。

建议配置文件名：`e0_dataset_norm_seed42.yaml`、`e0_autoaug_seed42.yaml`、`e0_focal_seed42.yaml`、`e0_unit_posweight_seed42.yaml`、`e0_sqrt_posweight_seed42.yaml`、`e0_manual_posweight_seed42.yaml`。可以复制现有 YAML 后只修改实验因素、`run_name` 和 `output_dir`；后续若配置数量增加，再考虑写参数化 launcher。

### 3. 记录规则与实验判定

每次 run 应保存完整配置副本和 manifest；当前代码会保存逐 epoch 验证 AUC、最佳 epoch、逐标签 AUC、数据清单路径与 SHA-256、预训练权重 SHA-256、Git revision 和本地工作区状态（已跟踪 diff 的 SHA-256 与未跟踪文件路径）、损失、posweight 和归一化信息。不得复用输出目录。重跑实验要换新的 run 名称，并注明重跑原因。

筛选时先看十标签 macro AUC，再看三种子均值和标准差、各标签变化与训练稳定性。单个 seed 的小幅提升只作候选，不称为稳定提升；需要 seed42 筛选的候选，使用 2027 和 3407 做受控复核。相同验证集上的很多轮筛选会逐渐过拟合验证集，所以先按上述少量单因素对照推进。

新模型 valid AUC 必须在新工程相同代码、数据和指标口径下比较。最终配置冻结后再评估比赛 test；不要根据 test 分数挑配置。旧版本基线、旧 Mac seed2027 基线、单病种 AUC 与新工程十标签 macro AUC 分开归档。

## 开始实验前检查

- [x] `project_main` 中 G0/G1/AutoAugment 实际变换已由配置控制，G0 测试确认无随机操作。
- [x] G1 的运行 transform 与配置字段一致；水平翻转默认关闭。
- [x] dataset normalization、Focal loss 和四种 posweight 策略已经有配置入口和合同测试。
- [ ] train/valid/test 使用同一份锁定的数据清单版本，test 不进入训练或筛选。
- [ ] ConvNeXt-Base 权重文件 SHA-256 校验通过，模型标识为 `convnext_base.fb_in22k`。
- [ ] 每个 seed、增强、warmup 组合使用唯一输出目录。
- [x] 开跑前记录 run 配置、数据版本、代码版本和预期 seed。
