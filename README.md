# CXR 比赛训练工程（Windows CUDA / Apple Silicon MPS）

这是独立训练工程，使用**纯 ImageNet-22K 通用预训练权重**初始化 ConvNeXt-Base，再训练比赛的 10 标签 Study 分类头和困难标签专家头。骨干为 `timm` 模型 `convnext_base.fb_in22k`；没有使用医学模型或医学预训练权重，也没有采用 ImageNet-22K 之后再经 ImageNet-1K 微调的 `fb_in22k_ft_in1k` 版本。

工程支持配置校验、ConvNeXt/ViT/Swin 训练、验证集推理、测试集预测、网格配置生成及融合/导出评估。训练可运行在 Windows CUDA、Apple Silicon MPS 或 CPU 上。新增接口通过运行检查，不等于已经完成正式训练或提高比赛成绩。

## 现阶段目标与实验安排（2026-11-08）

当前目标是提高纯图像模型在比赛数据上的泛化能力，重点检验“不同疾病标签分别选择图像和空间区域”是否比现有共享特征方案有效。历史单模型或融合的验证AUC较高，并未稳定转化成更高的比赛成绩，因此本阶段先以固定数据、骨干、损失和训练预算做结构对照，再根据重复实验结果决定后续方向。

| 顺序 | 实验 | 需要回答的问题 | 当前状态 |
| --- | --- | --- | --- |
| 1 | `HEAD_C0` vs `HEAD_C1`，seed42 | 原0/3/7专家头是否有收益？建立同协议对照 | 代码/配置已验收，未启动正式训练 |
| 2 | `HEAD_H1` vs `HEAD_C1`，seed42 | 每种疾病分别选择Study内图像是否有效？ | 代码/配置已验收，未启动正式训练 |
| 3 | `HEAD_H2` vs `HEAD_C1`，seed42 | 保留空间特征、由标签query提取区域表示是否有效？ | 代码/配置已验收，未启动正式训练 |
| 4 | 候选及对应对照补seed2027/3407 | 收益能否跨种子复现，哪些标签受益或退步？ | 等首轮结果后确定候选 |
| 5 | `HEAD_H3` | 逐标签图像聚合与空间query组合是否互补？ | 接口已验收，按H1/H2结果决定是否训练 |
| 6 | 原版通用DINOv2 ViT-B/14学习率与分辨率网格 | 通用自监督权重能否提升比赛验证集表现？ | 代码与分阶段配置入口已实现；21次Windows/CUDA正式训练待跑 |

首轮C0/C1/H1/H2都是独立实验，可以分配给不同训练机并行运行；它们各自从相同ImageNet权重重新初始化，不继承上一实验的比赛checkpoint。首轮共4次训练；若只保留一个结构候选，再补该候选与C1的两种子共4次，合计8次。保留多个候选或新增H3时另计。

主指标是固定官方验证集的10标签macro AUC，同时记录逐标签AUC和训练曲线；F1在预先固定阈值后通过导出的验证预测另行计算，作为辅助指标。复核时报告42/2027/3407配对差值、均值、样本标准差和患者级bootstrap。不要仅用某个seed的最高值宣布结构胜出。原L5结果只有在数据、停止规则和EMA选权重协议一致时才可复用；否则重跑C0。外部CheXpert仅作冻结方案后的泛化审计，比赛测试集不用于本阶段调参。

所有HEAD实验使用去除两张破损图的A_exclude2清单、256分辨率、15轮/warmup5、LR5e-4、WD.10、layer_decay.65、G1、Focal gamma2、自动ratio posweight/cap20、EMA.9998、有效Study batch16；关闭视角输入/过滤、早停、梯度裁切。新结构不接收测试ViewPosition或临床字段。推荐配置是`configs/head_baseline_l5.yaml`（Windows）和`configs/head_baseline_l5_mac.yaml`（Mac）；下方保留的旧G1/S0–S7说明属于历史参考。完整HEAD连续命令及权重链接在文末。

当前交付包括结构代码、跨平台配置、恢复/推理兼容、DINOv2模型及网格入口；还没有新结构的真实训练AUC。正式提交前另行核对所有模型权重与必要文件的500MB总量和官方推理耗时。

## 工程结构

```text
project_main/
├── configs/head_baseline_l5.yaml  # 当前HEAD实验Windows完整基线
├── configs/head_baseline_l5_mac.yaml # 当前HEAD实验Mac运行配置
├── configs/g1_seed42.yaml          # 历史G1配置，非本轮默认基线
├── data/manifests/                 # train / valid / test CSV 清单
├── scripts/                         # 环境、数据、训练、推理、网格和评估命令
│   ├── grid_search.py              # 生成/执行单因素网格配置
│   ├── evaluate_grid.py            # folds、bootstrap、融合、导出、最终清单
│   ├── predict_test.py             # raw/EMA、TTA、多尺寸预测
│   └── export_competition_submission.py # 转比赛长表
├── src/cxr_project/                # 数据、模型、损失、训练、实验接口
├── docs/grid_interfaces.md         # 网格开关和运行命令
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

以下是历史 `g1_seed42.yaml` 的结构和参数。新一批纯图像标签头实验使用独立的 `head_baseline_l5*.yaml`，配置与运行方式见文末“标签注意力与空间查询头实验”；不要用下面的旧参数表替代新实验配置。

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

## Mac（Apple Silicon / MPS）环境准备

同一份 `src/cxr_project` 训练代码支持 Apple Silicon 的 MPS。Mac 使用单独的 `.venv-mac` 环境，Windows CUDA 环境和安装脚本保持独立。MPS 训练与预测固定使用 FP32，自动关闭 CUDA AMP 和 GradScaler；Apple 芯片无需安装 CUDA、Apex、xformers 或 Triton。

需要 Apple Silicon 和 Python 3.11 或 3.12。在工程目录运行：

```bash
brew install python@3.12  # 若尚未安装 Python 3.12
chmod +x scripts/setup_mac_mps.sh scripts/run_g1_seed42_mac.sh
./scripts/setup_mac_mps.sh
```

安装脚本会在 `.venv-mac` 创建环境，安装 PyTorch 2.8.0 / torchvision 0.23.0 与项目依赖，然后用 MPS 做设备和矩阵运算检查。随后下载通用 ImageNet-22K 权重并检查比赛数据：

```bash
.venv-mac/bin/python scripts/download_weights.py
.venv-mac/bin/python scripts/validate_data.py --data-root ../cxr_first
```

Mac 配置沿用 G1 的 ConvNeXt-Base、224 分辨率、10 轮、5 轮 warmup、LR `1e-3`、E0 标签、正样本权重、损失和增强。为适配 16GB Apple Silicon，配置使用 Study batch 1、累积 16 步、单图块、0 个 DataLoader worker、关闭 pin memory 和 AMP；每次优化器更新仍累计最多 16 个 Study。配置文件为 [configs/g1_seed42_mac.yaml](configs/g1_seed42_mac.yaml)。运行完整训练：

```bash
./scripts/run_g1_seed42_mac.sh
```

若图片目录不在工程同级目录，把实际路径作为第一个参数：

```bash
./scripts/run_g1_seed42_mac.sh '/Volumes/External SSD/cxr_first'
```

也可以直接调用入口：

```bash
.venv-mac/bin/python -m cxr_project.train \
  --config configs/g1_seed42_mac.yaml \
  --data-root ../cxr_first \
  --device mps
```

训练会完整运行配置中的 10 轮，并将 checkpoint、逐轮验证 AUC 和运行记录写入 `runs/G1_seed42_convnext_base_in22k_mac/`。MPS 与 CUDA 的浮点执行不同，结果会有小幅差异；Mac 适配本身不代表性能提升。设备可用性检查命令为：

```bash
.venv-mac/bin/python scripts/check_environment.py --device mps
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

Mac 和 Windows 可使用同一训练入口；在 Mac 上传入 `--device mps`，在 Windows 上传入 `--device cuda`。自动选择设备也可用 `--device auto`（优先 CUDA，其次 MPS，最后 CPU）。

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

在 Mac 上执行已生成的网格配置时加 `--device mps`，并使用 Mac 环境中的 Python：

```bash
.venv-mac/bin/python scripts/grid_search.py \
  --baseline configs/g1_seed42_mac.yaml \
  --experiments G0 G2 G3 O1 O2 \
  --seeds 42 2027 3407 \
  --stage S0 \
  --config-dir configs/generated/S0_mac \
  --output-root runs/grid/S0_mac \
  --device mps
```

请在 Windows 工程目录内生成配置，以写入本机绝对路径。检查 `grid_manifest.json` 后，加入 `--execute` 才会按顺序启动完整训练。每个阶段传入已经确认的完整基线；评估/融合命令与运行限制见接口文档。

新增训练接口包括 G0 与几何/光度增强、MixUp、ASL、温和采样、CLAHE、冻结或部分解冻、通用 ConvNeXt-Small/Swin/ViT、AdaptFormer、标签子集训练、raw/EMA 双评估及 fixed-budget 最终训练。AF0/AF1 是 ImageNet 通用 ViT 的困难标签候选接口，尚未训练。

2026-10-28 更新：取消后续 AP/PA/LATERAL/LL 视角专家及依赖 ViewPosition 的模型路由。保留按疾病标签分工的 MLP 专家和 AF0/AF1 候选。比赛实验必须在完整基线 YAML 中显式设置以下两项；不要直接沿用旧模板的默认值：

```yaml
use_view_embedding: false
view_filter: null
```

最新网格已更新至 **2026-11-04 S4 收尾版**，见 [网格图](docs/experiments/imagenet_grid_draft/ImageNet训练网格汇总.png)、[合并 CSV](docs/experiments/imagenet_grid_draft/ImageNet训练网格汇总.csv) 和 [S4/S5 分图](docs/experiments/imagenet_grid_draft/02_正则增强与损失.png)。S3 参考基准仍为 I6（256）；S4 选择 **R2 作为 S5 基线**：WD=.10、Drop Path=.10、头 Dropout=.20、普通 G1，三种子 AUC 0.811005784 / 0.813180464 / 0.810071988，均值 **0.811419412**、样本标准差 0.001594983。R4 保留作稳定性与融合候选。缺少精确日志的种子留空，不虚填三种子均值；实际中断及未闭合状态保留。

S5 先做 **L1–L3 三组新增 posweight 实验**（42/2027/3407，共9次），与已有 ratio/cap20 基线比较；选定后再从实际胜出的完整 BCE 配置生成 L4/L5，比较 Focal gamma=1/2，继承选定 posweight。不强制新增 L0；若改变停止/验证协议，旧 R2 只能作历史参考，必要的同协议对照另计。所有实验从通用 ImageNet 权重重新初始化，关闭视角输入、过滤与裁切。详细来源见 [S4 复核记录](outputs/analysis/s4_review_20261104/S4复核与S5基线.md)，[S3 逐标签表](docs/experiments/imagenet_grid_draft/S3_逐标签分辨率对照.csv) 保留原记录。

最终训练模式通过 `final_training: true` 禁用验证和最佳轮次选择，固定轮数结束输出 `checkpoint-final.pth`。test 仅用于方案冻结后的最终评测。FP16 与 AMP 是不同设置；实际交付仍须汇总模型及必要文件大小核对 500MB 限制。

完整自动化测试在本地 CPU/合成数据环境通过。Windows CUDA 下的大规模训练、新骨干权重下载与加载、真实数据 TTA 以及最终部署包体大小仍需在目标机器验证。

## 比赛推理与耗时

scripts/predict.py 和 scripts/cost.py 共用 scripts/competition_inference.py。部署清单格式见 configs/competition_deployment.example.json；先复制为 configs/competition_deployment.json，再填入已确认模型的配置和checkpoint路径。示例只展示字段，不代表当前推荐的融合或平滑参数。历史本机部署清单和权重不随代码上传；HEAD实验尚未正式训练，不可直接当部署模型。输入只从 subject_*/study_*/图片目录读取 ID，兼容字段 ViewPosition 固定为 UNKNOWN，不读取测试集临床或视角信息。

Windows 正式预测与官方100张样本测速：

```powershell
python scripts/predict.py --bundle configs/competition_deployment.json --data-dir D:\path\to\pictures --output submission.csv --device cuda
python scripts/cost.py --bundle configs/competition_deployment.json --data-dir D:\path\to\cost_data_test\pictures_100 --warmup-dir D:\path\to\cost_data_test\pictures_5 --output-dir outputs\cost_run --device cuda
```

测速默认要求100张输入、5张独立预热图片（优先各取自不同Study）、连续3轮；计时包含图片读取、预处理、三个模型串行推理、融合、患者平滑和最终CSV写盘，模型加载、扫描目录及预热不计时。Study内的图像特征也逐张编码（image batch=1），再执行完整Study聚合。输出报告记录模型哈希、权重来源、精度、样本数和计时边界。比赛提供的本机样例不足100张时，仅可用 --allow-short-sample 联调，该结果也只作诊断。

患者平滑只使用本次 data-dir 中同一 Subject 的 Study。若要与全量提交逐Study核对，输入目录必须包含参与平滑的完整测试集；用抽样目录与全量提交比较会因患者上下文不同而产生预期差异。

## 标签注意力与空间查询头实验（2026-11-08）

第一批代码已实现，仍使用 ConvNeXt-Base 的 ImageNet-22K 通用权重。现有工程本来就是 Study 级训练；这批实验让不同标签分别选择 Study 中的图像，或从每张图的空间特征中提取标签相关表示。新增模块使用 PyTorch 原生运算，不需要额外安装 Query2Label、ML-Decoder、xformers 或其他注意力库。

| 配方 | 图像表示与 Study 聚合 | 旧 0/3/7 专家 | 对照用途 |
| --- | --- | --- | --- |
| `HEAD_C0` | 全局图像向量＋共享 Study attention＋10标签线性头 | 开启 | 同协议重跑原结构 |
| `HEAD_C1` | 全局图像向量＋共享 Study attention＋10标签线性头 | 关闭 | 新结构的直接对照 |
| `HEAD_H1` | 全局图像向量＋逐标签 Study attention＋逐标签分类 | 关闭 | 检验不同标签选择不同图像 |
| `HEAD_H2` | 空间标签 query＋共享 Study attention＋逐标签分类 | 关闭 | 检验标签相关空间特征 |
| `HEAD_H3` | 空间标签 query＋逐标签 Study attention＋逐标签分类 | 关闭 | H1/H2 有收益后再做组合 |

`HEAD_` 前缀用于区分历史网格中的 C1/H1 等名称。H1/H2 首先与 C1 比较；C0/C1 检验原专家头的价值。这是参考标签 query 思路实现的轻量结构，不是完整复现 Query2Label 或 ML-Decoder，也没有照搬 TransMIL。空间头从骨干的 NCHW 特征图读取 tokens，使用10个可学习 query、256维、1层 cross-attention、4头、dropout 0.2；查询头随机初始化，无需额外预训练权重。

本批配置固定：256分辨率、15轮、warmup 5轮、AdamW LR `5e-4` / min LR `1e-6`、WD `0.10`、layer decay `0.65`、drop path `0.10`、头 dropout `0.20`、G1轻增强、Focal gamma=2、Study级自动 ratio posweight/cap=20、EMA `0.9998` 从第1轮更新、只用 EMA 验证选优、关闭早停和梯度裁切、有效 Study batch=16。数据继续使用 A_exclude2 清单，去除的仍是两张损坏图片。所有 HEAD 配方均关闭视角输入和过滤，只训练10个目标标签；H1/H2/H3 还会拒绝专家头和 MixUp 混用。

原 L5 部分结果采用过不同早停或选权重方式，不能直接当作这批统一协议的重复实验。先做 seed42 的 C0/C1/H1/H2；筛出候选后，给候选及其对照补齐2027和3407，按配对差值、三种子均值/样本标准差、逐标签 AUC 和患者级 bootstrap 判断。H3 接口已提供，暂不列入首轮连续训练。验证集结果用来比较新结构；比赛测试集与外部医学评估集不参与训练、超参数或 checkpoint 选择。

Windows/CUDA 使用 [configs/head_baseline_l5.yaml](configs/head_baseline_l5.yaml)，默认 Study batch=2、累积8、chunk=2、workers=4、AMP开启。以下命令只生成首轮4份配置：

```powershell
.\.venv\Scripts\python.exe .\scripts\grid_search.py --baseline configs/head_baseline_l5.yaml --experiments HEAD_C0 HEAD_C1 HEAD_H1 HEAD_H2 --seeds 42 --stage pilot --config-dir configs/generated/heads_seed42 --output-root runs/heads/pilot --device cuda
```

检查生成 YAML 后，连续运行这4个实验：

```powershell
$headJobs = (Get-Content -Raw .\configs\generated\heads_seed42\grid_manifest.json | ConvertFrom-Json).jobs
foreach ($headJob in $headJobs) {
    & .\.venv\Scripts\python.exe -m cxr_project.train --config $headJob.config --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $($headJob.name)" }
}
```

Mac 使用 [configs/head_baseline_l5_mac.yaml](configs/head_baseline_l5_mac.yaml)：Study batch=1、累积16、chunk=1、workers=0、关闭AMP和pin memory。其余训练因素相同；MPS与CUDA数值和micro batch不同，不能保证位级一致。16GB机器保留梯度检查点，实际速度和内存仍以目标机器运行情况为准。生成配置并按清单连续运行：

```bash
.venv-mac/bin/python scripts/grid_search.py --baseline configs/head_baseline_l5_mac.yaml --experiments HEAD_C0 HEAD_C1 HEAD_H1 HEAD_H2 --seeds 42 --stage pilot_mac --config-dir configs/generated/heads_seed42_mac --output-root runs/heads/pilot_mac --device mps
.venv-mac/bin/python - <<'PY'
import json
import subprocess
from pathlib import Path
jobs = json.loads(Path('configs/generated/heads_seed42_mac/grid_manifest.json').read_text())['jobs']
for job in jobs:
    subprocess.run(job['command'], check=True)
PY
```

每个配置与输出目录均须未被占用。生成器将本机绝对路径写入 YAML 和清单，请在真正训练的 Windows/Mac 上重新生成，不要互相复制生成后的绝对路径配置。若要一次性生成五配方×三种子的15份配置，将 experiments 补上 `HEAD_H3`、seeds 改为 `42 2027 3407`，并使用新的配置/输出目录；默认不启动训练，只有 `--execute` 或后续显式训练命令会启动。

每轮指标写入 `epoch_metrics.csv` 和 `epoch_metrics_ema.csv`；最佳EMA权重及对应验证预测保存在 `checkpoint-best-ema.pth` 和 `valid_best_ema_predictions.csv`。恢复必须使用该实验原配置及 `checkpoint-last.pth`：不允许把旧模型 checkpoint 改配置后当作新结构续训。旧配置缺少新增字段时按历史线性头处理，旧权重参数名和计算路径保持兼容。推理仍使用共用的 `scripts/predict_test.py`、competition/predict/cost 模型构建流程；新增结构的推理必须保留相同 head/query/pooling 参数。完整参数与恢复/预测命令见 [docs/grid_interfaces.md](docs/grid_interfaces.md#head-组参数与恢复推理)。

**权重下载：** 本批只需要已有的 [ConvNeXt-Base ImageNet-22K 模型卡](https://huggingface.co/timm/convnext_base.fb_in22k)中的权重；[固定版本 safetensors 直链](https://huggingface.co/timm/convnext_base.fb_in22k/resolve/9afd4146adabf1b6e861d0f1ac2e7baa7c8dd1b7/model.safetensors)。下载后保存为 `weights/convnext_base.fb_in22k.safetensors`；现有下载脚本会校验大小和上文 SHA-256。已有文件可直接复用。

## DINOv2 ViT-B/14 网格（2026-11-08）

权重下载地址为 Meta 官方 [DINOv2 ViT-B/14 checkpoint](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth)，属于通用 LVD-142M 自监督预训练，不使用医疗模型或医疗数据；[官方代码和许可](https://github.com/facebookresearch/dinov2)。下载脚本会校验固定字节数 `346378731` 和 SHA-256 `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`。输入边长均为14的倍数。基线文件为 [configs/dinov2_vitb14_baseline.yaml](configs/dinov2_vitb14_baseline.yaml)，保留 S5-L5 训练设置：15轮、warmup5、线性warmup后余弦退火、min LR `1e-6`、EMA `0.9998` 并仅用 EMA 验证选优、Study batch16、累积1、image chunk2、G1、Focal gamma2、自动 ratio posweight/cap20、关闭视角输入/过滤、梯度裁切和早停。训练用 CUDA AMP；FP16 只用于每次训练后的 EMA 推理权重导出。

网格实验名和组合如下，共21次。配置文件、run_name 和输出目录使用相同的生成名称；所有试验使用种子42/2027/3407。学习率按对数间距覆盖较保守到较积极的微调范围。

| 实验 | 配置名 | 分辨率 | 学习率 | 数量 |
| --- | --- | ---: | ---: | ---: |
| D1 | `DINOv2_lr2.5e-5_seed{42,2027,3407}` | 252 | `2.5e-5` | 3 |
| D2 | `DINOv2_lr5e-5_seed{42,2027,3407}` | 252 | `5e-5` | 3 |
| D3 | `DINOv2_lr1e-4_seed{42,2027,3407}` | 252 | `1e-4` | 3 |
| D4 | `DINOv2_lr2e-4_seed{42,2027,3407}` | 252 | `2e-4` | 3 |
| D5 | `DINOv2_lr4e-4_seed{42,2027,3407}` | 252 | `4e-4` | 3 |
| R224 | `DINOv2_res224_lr<best-lr>_seed{42,2027,3407}` | 224 | D1-D5胜出值 | 3 |
| R280 | `DINOv2_res280_lr<best-lr>_seed{42,2027,3407}` | 280 | D1-D5胜出值 | 3 |

先比较 D1-D5 三种子最佳 EMA 验证 macro AUC 均值选学习率，再以该学习率比较 R224、R280 和同学习率的 D1-D5（252）结果。只读固定 train/valid 清单；比赛测试标签、ViewPosition和临床字段不用于训练或选择。

在 Windows/CUDA 项目目录执行以下命令。下载器写入权重及 `.source.json`，记录官方 URL、字节数和 SHA-256；模型创建只加载此本地权重，不会隐式联网：

```powershell
.\.venv\Scripts\python.exe .\scripts\download_dinov2_weights.py
.\.venv\Scripts\python.exe .\scripts\dinov2_grid.py stage1 --config-dir configs\generated\dinov2\stage1 --output-root runs\grid\dinov2\stage1 --device cuda
```

第一条网格命令默认只生成并验证15份配置。检查配置和 `grid_manifest.json` 后，用单独的 `run` 命令连续训练并在每个 run 完成后导出 `checkpoint-best-ema-fp16.pth`。第一阶段结束后汇总三种子验证结果：

```powershell
.\.venv\Scripts\python.exe .\scripts\dinov2_grid.py run --config-dir configs\generated\dinov2\stage1
.\.venv\Scripts\python.exe .\scripts\dinov2_grid.py select-lr --stage1-config-dir configs\generated\dinov2\stage1 --report outputs\dinov2\stage1_lr_selection.json
.\.venv\Scripts\python.exe .\scripts\dinov2_grid.py stage2 --stage1-config-dir configs\generated\dinov2\stage1 --config-dir configs\generated\dinov2\stage2 --output-root runs\grid\dinov2\stage2 --device cuda
```

`stage2` 会从第一阶段完整的 `run_manifest.json` 中自动选均值最高的 LR，并生成6份配置；检查后运行 `.\.venv\Scripts\python.exe .\scripts\dinov2_grid.py run --config-dir configs\generated\dinov2\stage2` 开始第二阶段。`run` 会先核对配置哈希并拒绝已有 run 输出目录。若已有训练由其他方式完成，可按 `grid_manifest.json` 中的 `export_fp16_command` 单独导出 EMA FP16 权重。FP16 文件是部署/存储用导出，不是 FP16 训练权重；完整 checkpoint 与验证预测仍保留。

此工程没有在 Mac 上启动正式训练；在目标 Windows CUDA 机器下载真实权重后，仍应先确认能够加载完整 checkpoint，再运行长训练。原版 MAE 官方权重采用 NC 许可，按当前比赛规则不列入方案。更多规则背景见 [通用自监督路线评估](docs/experiments/多标签头_MIL_通用自监督路线评估_20261107.md)。

本批验收：136项自动测试通过；官方 DINOv2 checkpoint 已验证并真实加载，MPS 下224/252/280三档均完成10标签前向；第一阶段15份配置生成与路径校验通过。HEAD结构仍无正式训练成绩，DINOv2的21次Windows/CUDA正式训练尚未启动。CUDA训练未在本机执行。其他结构接口覆盖与H3少量冗余计算说明见 [接口验收记录](docs/experiments/标签头接口验收_20261108.md)。
