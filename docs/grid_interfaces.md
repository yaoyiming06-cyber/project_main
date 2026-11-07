# 网格配置与评估工具

这份文档对应新接口。CSV 是实验设计与历史记录；可执行参数以 `src/cxr_project/grid.py` 中的明确配方和生成后的 YAML 为准。不会把 CSV 描述中的文字直接当作代码开关。历史 H1/H2/H3/R0/H4/F0 不自动重跑；V 组使用评估工具，不伪装成训练开关。

## 环境与阶段规则

以下命令以 Windows PowerShell 和 CUDA 环境为例。Mac 可用工程内 `.venv-mac` 环境执行相同网格接口，并传入 `--device mps`：

```powershell
python -m pip install -e .
```

每个阶段显式传入上一阶段确认的完整配置；通常重新从同一通用 ImageNet 权重初始化，不从胜出 checkpoint 续训。先筛 seed42，候选以 2027/3407 配对复核。保留固定原始基线，组合胜出因素后再验证交互作用。配置生成器默认只写配置和清单，不启动训练；只有 `--execute` 才逐个执行。

**2026-10-31 比赛计划调整：** 后续基线必须显式配置 `use_view_embedding: false`、`view_filter: null`，训练和测试推理均不使用 ViewPosition，也不按该字段选择图像或模型。以下命令中的 baseline 要先按这两项更新；生成器不会替你自动覆盖旧模板。M6–M9 视角专家和 ViewPosition 路由已从比赛网格取消，历史代码和训练文件仅留档。M4 纯图像 layer_decay=.75 的三个种子均已完成，无需补跑；均值0.804990306，未稳定优于O8 .65的0.806490459。后续S3基准为I6（256），保留layer_decay=.65。

```powershell
python scripts/grid_search.py --baseline configs/g1_seed42.yaml --experiments A0 --seeds 42 2027 3407 --stage S0 --config-dir configs/generated/S0 --output-root runs/grid/S0
python scripts/grid_search.py --baseline configs/generated/S0/S0_A0_seed42.yaml --experiments C1 C2 C3 C4 C5 C6 --seeds 42 --stage S1 --config-dir configs/generated/S1 --output-root runs/grid/S1
python scripts/grid_search.py --baseline configs/confirmed_S2.yaml --experiments I4 I5 I6 I1 I2 --seeds 42 2027 3407 --stage S3 --config-dir configs/generated/S3 --output-root runs/grid/S3
```

`configs/confirmed_S2.yaml` 是人工确认胜出后保存的完整配置，示例不表示这份文件已经存在。I4/I5/I6/I1/I2 分别为192/224/256/320/384。生成器把路径及 project_root 固定为本机绝对路径，避免嵌套 configs 文件夹改变路径含义；请在实际训练的机器上生成配置，Mac 与 Windows 之间不要直接搬运含绝对路径的生成 YAML。若输入是运行目录中的 config_resolved.yaml，传 `--project-root .` 指定工程根目录。

每次生成目录、训练输出目录必须是新的；已存在会报错，防止覆盖。所有生成的 YAML 都经过正式 load_config 校验。`grid_manifest.json` 记录基线、配置 SHA256 和可执行命令。已有生成目录需要直接按清单里的命令执行，不要对同一目录再次生成。

Mac 示例：

```bash
.venv-mac/bin/python scripts/grid_search.py \
  --baseline configs/g1_seed42_mac.yaml \
  --experiments G0 G2 \
  --seeds 42 \
  --stage mac_smoke \
  --config-dir configs/generated/mac_smoke \
  --output-root runs/grid/mac_smoke \
  --device mps
```

## 实验映射说明

- A0 固定15轮、warmup5、LR5e-4、EMA .9998、自第1轮启动、clip1、关闭早停并输出raw/EMA验证。配置接口已就绪，仍需在目标加速器上完成真实训练。EMA尚未启动的早期epoch按raw验证；启动后，validation_weights=auto/ema才使用EMA，checkpoint.selected_weights记录实际来源。
- C1–C6 控制 EMA/裁切；O1–O10 控制训练预算、LR、warmup、层衰减、有效batch；R1–R6 控制WD、drop path、主头与专家头Dropout。
- G0/G2–G8、I3 控制无增强、几何、翻转、Gamma、擦除、Study MixUp、AutoAug、训练统计归一化和CLAHE；尺寸使用I1/I2/I4/I5/I6。
- L1–L6 控制posweight/Focal/ASL。**L7 必须与 unit posweight 的 L2 基线比较**；L7启用温和采样且强制unit，避免强采样与ratio权重叠加。拿ratio基线直接比较L7会同时改变两个因素。
- M1是真正禁用专家及logit混合；M2调整辅助损失；M3均值聚合；M5只解冻最后两个stage。M4已从ImageNet初始化完成42/2027/3407三种子纯图像训练，最佳EMA AUC为0.803137482 / 0.806635473 / 0.805197964；三组均检查use_view_embedding=false和view_filter=null。M4保留为已完成结果，旧C3分数不能替代M4分数。
- B1/B2更换骨干；B3不再改参数，只把传入的已确认组合配置复制到三种子。AF0/AF1为新增ViT三标签普通冻结头/AdaptFormer配对实验，均只优化标签0/3/7，使用预训练模型对应归一化；接口已实现但尚未下载权重或训练。其余标签输出不应参与融合。

## 新骨干通用权重

只有用户运行下载命令才访问模型文件。脚本限制为列出的通用ImageNet来源，先把HF revision解析为不可变commit，再下载safetensors；记录来源URL、commit、大小、SHA256及上游LFS摘要核对结果。训练端不会悄悄改用随机初始化或下载医学权重。

```powershell
python scripts/download_general_weights.py --model swin_base_patch4_window7_224.ms_in22k --output weights/swin_base_patch4_window7_224.ms_in22k.safetensors
python scripts/download_general_weights.py --model convnext_small.fb_in22k --output weights/convnext_small.fb_in22k.safetensors
python scripts/download_general_weights.py --model vit_base_patch16_224.augreg_in21k --output weights/vit_base_patch16_224.augreg_in21k.safetensors
python scripts/grid_search.py --baseline configs/confirmed_S5.yaml --experiments AF0 AF1 --weights AF0=weights/vit_base_patch16_224.augreg_in21k.safetensors --weights AF1=weights/vit_base_patch16_224.augreg_in21k.safetensors --seeds 42 2027 3407 --stage AF --config-dir configs/generated/AF --output-root runs/grid/AF
```

B1/B2同样传 `--weights B1=...` 或 `--weights B2=...`，生成器计算该本地文件SHA256写入YAML。下载来源的合规性仍应按比赛规则最终确认。

## V1：只在train内按患者三折

```powershell
python scripts/evaluate_grid.py folds --manifest data/manifests/train_manifest.csv --output-dir data/manifests/train_subject_3fold --folds 3 --seed 42
```

所有同Subject的Study和图片留在同一折。采用随机打散后按Study数量贪心均衡，**不是多标签分层算法**；使用前检查各折标签阳性数。生成的folds.json和每折train/valid CSV可作为该折基础YAML的manifest；再用grid_search生成三个训练种子。官方valid不参与分折，不能把这种分数视为官方valid同协议成绩。

## V2：患者配对Bootstrap

```powershell
python scripts/evaluate_grid.py bootstrap --first runs/baseline/valid_best_predictions.csv --second runs/candidate/valid_best_predictions.csv --manifest data/manifests/valid_manifest.csv --iterations 2000 --seed 42 --report reports/paired_bootstrap.json
```

按Study_id重排，要求覆盖完全相同且targets一致；按Subject成组、有放回抽样，对两个模型使用同一组抽样。差值方向为candidate减baseline。任何原本可评估标签在一次抽样中丢失正/负类，就跳过整次配对并报告跳过数。Bootstrap不能消除反复调参的选择偏差。

## V3/V4：固定TTA

使用 `scripts/predict_test.py` 的 `--tta horizontal-flip` 和 `--tta-sizes`。在验证集先固定方案，再用于比赛测试；不要用测试分数挑尺寸。详见该脚本 `--help`。

## V5：等权融合

```powershell
python scripts/evaluate_grid.py fuse --predictions runs/a/valid_best_predictions.csv runs/b/valid_best_predictions.csv --output reports/equal_valid.csv --report reports/equal_valid.json
```

融合只做固定等权概率平均，严格对齐Study和标签，不搜索权重。比赛方案不使用ViewPosition路由。`--labels 0 3 7` 只替换指定标签。输入兼容每Study一行的测试 `label_0..9` 和验证 `probability_0..9`，验证文件另含 `target_0..9`。无标签融合输出 `label_0..9`，可直接接提交导出工具。标签专家使用 `fuse --labels 0 3 7`，其余七标签保留第一个基础模型预测。比赛长表submission需继续使用既有提交导出工具。

## V6：FP16导出与复核

```powershell
python scripts/evaluate_grid.py export-fp16 --checkpoint runs/best/checkpoint-best.pth --output exports/model_fp16.pth --weights auto --report exports/model_fp16.json
python scripts/evaluate_grid.py compare --reference reports/source_valid.csv --candidate reports/fp16_valid.csv --report reports/fp16_comparison.json
```

只导出选中的raw或EMA一份权重，去除optimizer/scaler等训练状态，保留配置与归一化元数据。auto优先使用checkpoint记录的已验证权重来源，否则依据EMA活动状态。导出后须用相同验证manifest、尺寸、聚合和TTA分别推理原文件与导出文件，compare报告概率最大/平均绝对误差及AUC差。仅导出成功不等于已验证准确率；500MB要求需加总最终全部权重和必要文件，不能只看单个文件。

## V7：冻结方案后合并最终训练

```powershell
python scripts/evaluate_grid.py final-manifest --train data/manifests/train_manifest.csv --valid data/manifests/valid_manifest.csv --output data/manifests/final_train_valid.csv --baseline configs/confirmed_final.yaml --epochs 15 --config-output configs/final_seed42.yaml --run-output runs/final_seed42 --final --report reports/final_manifest.json
python -m cxr_project.train --config configs/final_seed42.yaml --device cuda
```

`--final`是显式最终模式，必须先确定所有超参数、固定训练轮数和推理方式。合并前检查Subject/Study/image_path不重叠，记录两个输入摘要；配置设 `final_training:true`、`valid_manifest:null`、早停关闭。最终固定轮数结束使用last checkpoint，不得用已经进入训练的旧valid重新选模。工具本身仅生成manifest/config，不自动训练。

## HEAD 组参数与恢复推理

第一批配方为 `HEAD_C0`、`HEAD_C1`、`HEAD_H1`、`HEAD_H2`、`HEAD_H3`，与历史同名的 C/H 实验分开。完整训练因素固定在 `grid.py::_HEAD_PROTOCOL`，不从任意上一阶段模板继承早停、裁切、标签子集或增强。请使用 `configs/head_baseline_l5.yaml`（Windows）或 `configs/head_baseline_l5_mac.yaml`（Mac），两者只在micro batch/累积、worker、chunk、AMP与pin memory上不同。旧骨干的其他权重路径不适合作为HEAD模板。

| 参数 | 默认 | 作用与限制 |
| --- | --- | --- |
| `head_type` | `linear` | `label_query` 启用空间标签表示，当前只支持ConvNeXt |
| `study_pooling` | `attention` | `mean`均值；`label_attention`逐标签选择图像 |
| `query_dim` | 256 | 正整数，必须能被query_heads整除 |
| `query_layers` | 1 | cross-attention块数，正整数 |
| `query_heads` | 4 | 注意力头数，正整数 |
| `query_dropout` | 0.2 | `[0,1)`；只作用于空间查询模块 |

`HEAD_C0`为原主头＋专家；`HEAD_C1`关闭专家；`HEAD_H1`为linear/label_attention；`HEAD_H2`为label_query/attention；`HEAD_H3`为label_query/label_attention。H2的共享图像权重仍由同次骨干提取的全局向量决定，H3按每个标签自己的空间表示打分。空间查询只读取图像特征内容，没有额外坐标或临床输入。单图Study的图像权重自然退化为1。

H1/H2/H3要求 `experts_enabled:false`、`use_view_embedding:false`、`view_filter:null`、`mixup_alpha:0`，否则在加载配置时拒绝。新头不支持用旧的 `[images,channels]` 缓存代替空间特征；查询需要原图前向或含空间标签表示的内部tuple。旧模型默认参数与state_dict兼容，新模块不会给旧模型添加权重键。新增结构与旧结构必须分别从相同通用预训练权重重新初始化；跨结构strict加载或改配置resume会被拒绝。

Windows首轮生成与运行命令见README。补两种子的示例（先确认候选；此处仅以H1为例，不表示它已经胜出）：

```powershell
python scripts/grid_search.py --baseline configs/head_baseline_l5.yaml --experiments HEAD_C1 HEAD_H1 --seeds 2027 3407 --stage confirm --config-dir configs/generated/heads_confirm_h1 --output-root runs/heads/confirm_h1 --device cuda
```

只生成配置，不访问比赛测试集，不自动训练。所有配方固定15轮、warmup5、EMA验证、无早停；用于比较的旧L5结果也必须符合这一协议。保留全部三种子的成绩及逐标签结果，避免只报告胜出的seed或单标签。

断点恢复示例：

```powershell
python -m cxr_project.train --config configs/generated/heads_seed42/pilot_HEAD_H2_seed42.yaml --resume runs/heads/pilot/pilot_HEAD_H2_seed42/checkpoint-last.pth --device cuda
```

验证集重新预测示例（先完成该实验训练）：

```powershell
python scripts/predict_test.py --config configs/generated/heads_seed42/pilot_HEAD_H2_seed42.yaml --checkpoint runs/heads/pilot/pilot_HEAD_H2_seed42/checkpoint-best-ema.pth --data-root ../cxr_first --manifest data/manifests/valid_manifest.csv --labeled --weights ema --device cuda --output reports/heads_h2_seed42_valid.csv
```

预测脚本会创建输出父目录。Mac把python换为`.venv-mac/bin/python`、device换为mps，并使用本机生成的配置和输出路径。不要修改head/query/pooling字段来适配不同checkpoint。测试推理也沿用同一个build_model；没有独立的推理结构副本。

权重仍为原ConvNeXt-Base ImageNet-22K；查询头无单独权重。DINOv2和官方训练集自监督路线属于后续独立任务，当前build_model不接受DINOv2配置，也没有自监督训练命令。下载来源见README和`docs/experiments/多标签头_MIL_通用自监督路线评估_20261107.md`。
