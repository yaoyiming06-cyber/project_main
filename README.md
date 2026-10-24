# CXR 比赛训练工程（Windows / CUDA）

这是独立的 Windows 训练工程，使用**纯 ImageNet-22K 通用预训练权重**初始化 ConvNeXt-Base，再训练比赛的 10 标签 Study 分类头和困难标签专家头。骨干为 `timm` 模型 `convnext_base.fb_in22k`；没有使用医学模型或医学预训练权重，也没有采用 ImageNet-22K 之后再经 ImageNet-1K 微调的 `fb_in22k_ft_in1k` 版本。

工程支持配置校验、ConvNeXt/ViT/Swin 训练、验证集推理、测试集预测、网格配置生成及融合/导出评估。当前新增网格方案均为可运行接口；这不代表它们已经训练或提升成绩。正式比赛训练仍需在 Windows CUDA 机器上执行。本地改动仅保留在工作区，没有提交或推送。

## 工程结构

```text
project_main/
├── configs/g1_seed42.yaml          # 本次完整训练配置
├── data/manifests/                 # train / valid / test CSV 清单
├── scripts/                         # 环境、数据、训练、推理、网格和评估命令
│   ├── grid_search.py              # 生成/执行单因素网格配置
│   ├── evaluate_grid.py            # folds、bootstrap、融合、导出、最终清单
│   ├── predict_test.py             # raw/EMA、TTA、多尺寸预测
│   └── export_competition_submission.py # 转比赛长表
├── src/cxr_project/                # 数据、模型、损失、训练、实验接口
├── docs/grid_interfaces.md         # 网格开关和 Windows 命令
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
- `checkpoint-best-raw.pth` / `checkpoint-best-ema.pth`：启用 `evaluate_raw_and_ema` 后分别保存的最佳权重。
- `epoch_metrics.csv`：逐轮总 AUC 和每个标签的 AUC。
- `valid_best_predictions.csv` 及 `valid_best_raw_predictions.csv` / `valid_best_ema_predictions.csv`：对应最佳模型对验证 Study 的预测。
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

续训 checkpoint 会恢复模型、优化器、GradScaler、EMA、epoch、更新步数、数据加载器/采样器及 Python、NumPy、PyTorch 随机状态；会先比较训练配置和 manifest 哈希，不匹配时拒绝续训。checkpoint 每个 epoch 结束时保存，中断的未完成 epoch 会从最近完成的 epoch 继续重跑。

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

## 网格接口与实验清单

网格条目现已接入可执行配置或评估接口。逐项开关、Windows 命令、阶段规则和已知限制见 [docs/grid_interfaces.md](docs/grid_interfaces.md)。实验计划、历史 AUC、三种子训练状态和接口状态见 [ImageNet训练网格汇总.csv](docs/experiments/imagenet_grid_draft/ImageNet训练网格汇总.csv)。接口就绪表示可生成或运行实验，不表示它已经训练，也不代表性能提升；旧实验结果和代码状态以 CSV 的当前记录为准。

网格脚本默认只生成并校验配置，不启动训练：

```powershell
python -m pip install -e .
python scripts/grid_search.py `
  --baseline configs/g1_seed42.yaml `
  --experiments G0 G2 G3 O1 O2 `
  --seeds 42 2027 3407 `
  --stage S0 `
  --config-dir configs/generated/S0 `
  --output-root runs/grid/S0
```

请在 Windows 工程目录内生成配置，以写入本机绝对路径。检查 `grid_manifest.json` 后，加入 `--execute` 才会按顺序启动完整训练。每个阶段传入已经确认的完整基线；评估/融合命令与运行限制见接口文档。

新增训练接口包括 G0 与几何/光度增强、MixUp、ASL、温和采样、CLAHE、冻结或部分解冻、通用 ConvNeXt-Small/Swin/ViT、AdaptFormer、标签子集训练、raw/EMA 双评估及 fixed-budget 最终训练。AF0/AF1 是 ImageNet 通用 ViT 的困难标签候选接口，尚未训练；AP/PA/LATERAL/LL 视角专家和独立小样本架构也没有新工程训练成绩。

最终训练模式通过 `final_training: true` 禁用验证和最佳轮次选择，固定轮数结束输出 `checkpoint-final.pth`。test 仅用于方案冻结后的最终评测。FP16 与 AMP 是不同设置；实际交付仍须汇总模型及必要文件大小核对 500MB 限制。

完整自动化测试在本地 CPU/合成数据环境通过。Windows CUDA 下的大规模训练、新骨干权重下载与加载、真实数据 TTA 以及最终部署包体大小仍需在目标机器验证。
