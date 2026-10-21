# CXR 比赛 Windows/CUDA 训练工程

本目录是独立的 Windows 训练项目。它使用 `ConvNeXt-Base + ImageNet-1K 通用预训练权重`，保留 G1 seed42 的训练节奏与标签口径，并使用对 10 个标签进行主预测、对少样本困难标签再加专家分支的结构。医学预训练模型和医学权重没有被引入。

训练按 Study 分组，读取一个 Study 下的全部图像，并用视角嵌入注意力汇总；每个 batch 是 2 个 Study，梯度累积 8 次。困难标签专家覆盖纵隔增宽、肺炎和肺部阴影。专家分支与普通分类头的 logit 取平均，训练时还增加专家辅助损失。

配置沿用已完成 G1 seed42 实验的关键参数：输入 224、10 epoch、warmup 5 epoch、batch 2、累积 8、LR 1e-3、min LR 1e-6、AdamW、weight decay 0.05、layer decay 0.75、drop path 0.1、自动 posweight 上限 20、E0（不确定标签忽略，空白按 0）。warmup 和余弦退火按梯度累积后的优化器更新次数计算。自动 posweight 从这里的清洗版训练 manifest 按唯一 Study 重算，不照搬旧数据产生的数值。G1 轻增强仍为小角度仿射、轻微平移缩放及亮度/对比度变化，无水平翻转；像素归一化改用 ImageNet 统计值以匹配通用权重。

## 文件与数据

- `data/manifests/` 保存 A_exclude2 的 train、valid、test 清单；图片本身没有复制。
- 训练图像根目录预设为 `project_main` 同级的 `cxr_first` 文件夹。清单中的图片路径是相对路径。
- 两张确认损坏图已从训练 manifest 排除：
  - `train/subject_01922/study_02467/pic_004044.jpg`
  - `train/subject_05239/study_13172/pic_021690.jpg`
- 当前清单为训练 30,342 张 / 18,513 Study，验证 1,422 张 / 859 Study，测试 2,218 张 / 1,414 Study。测试清单只保留用于之后预测，本训练入口不会读取它。

Windows 磁盘上应保持类似结构：

```text
<比赛目录>/
├── cxr_first/       # 原图与官方 CSV
└── project_main/    # 本工程
```

## Windows CUDA 环境

先在 Windows 安装 64 位 Python 3.11 或 3.12、NVIDIA 显卡驱动，再在 PowerShell 中运行：

```powershell
cd <比赛目录>\project_main
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_cuda.ps1
```

脚本建立本工程自己的 `.venv`，安装 PyTorch 2.8.0 / torchvision 0.23.0 CUDA 12.8 wheels、依赖和本工程，并检查 CUDA 是否可见。Wheel 自带 CUDA 运行时，系统只需兼容的 NVIDIA 驱动，不需另外安装完整 CUDA Toolkit。

## 检查清单与下载权重

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_data.py --data-root ..\cxr_first
.\.venv\Scripts\python.exe .\scripts\download_weights.py
```

权重来自 torchvision 的公开 ImageNet-1K ConvNeXt-Base checkpoint，已放在 `weights/convnext_base_imagenet1k_v1.pth`。只有该文件缺失时才需要运行下载命令；训练优先加载此文件。

## 开始完整训练

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_g1_seed42.ps1
```

等价命令：

```powershell
.\.venv\Scripts\python.exe -m cxr_project.train `
  --config .\configs\g1_seed42.yaml `
  --data-root ..\cxr_first `
  --device cuda
```

训练覆盖清单里的全部 Study，完整执行 10 个 epoch。输出写入 `runs/G1_seed42_convnext_base/`，包含 `checkpoint-best.pth`、`checkpoint-last.pth`、逐 epoch 验证 AUC、最佳验证集预测和 `run_manifest.json`。断点续训时将 `--resume` 指向已有 checkpoint，并指定相同 `--output-dir`；也可指定新输出目录，历史最佳模型和预测会一起保留。

如果 CUDA 发生 OOM，先把 `configs/g1_seed42.yaml` 的 `batch_size_studies` 从 2 改为 1；其余训练参数保持不变。每个 Study 的所有图像仍会参与训练，`image_chunk_size` 与梯度检查点用于限制显存峰值。

## 本机检查

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

本机 Mac 可以运行代码与清单检查，但无法执行 Windows CUDA 安装脚本或验证 NVIDIA GPU。完整训练应在装好 NVIDIA 驱动的 Windows 机器上启动。
# project_main
