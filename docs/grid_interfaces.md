# 网格配置与评估工具

这份文档对应新接口。CSV 是实验设计与历史记录；可执行参数以 `src/cxr_project/grid.py` 中的明确配方和生成后的 YAML 为准。不会把 CSV 描述中的文字直接当作代码开关。历史 H1/H2/H3/R0/H4/F0 不自动重跑；V 组使用评估工具，不伪装成训练开关。

## 环境与阶段规则

Windows PowerShell 在项目根目录、已配置好 CUDA PyTorch 的虚拟环境内执行：

```powershell
python -m pip install -e .
```

每个阶段显式传入上一阶段确认的完整配置；通常重新从同一通用 ImageNet 权重初始化，不从胜出 checkpoint 续训。先筛 seed42，候选以 2027/3407 配对复核。保留固定原始基线，组合胜出因素后再验证交互作用。配置生成器默认只写配置和清单，不启动训练；只有 `--execute` 才逐个执行。

```powershell
python scripts/grid_search.py --baseline configs/g1_seed42.yaml --experiments A0 --seeds 42 2027 3407 --stage S0 --config-dir configs/generated/S0 --output-root runs/grid/S0
python scripts/grid_search.py --baseline configs/generated/S0/S0_A0_seed42.yaml --experiments C1 C2 C3 C4 C5 C6 --seeds 42 --stage S1 --config-dir configs/generated/S1 --output-root runs/grid/S1
python scripts/grid_search.py --baseline configs/confirmed_S2.yaml --experiments I4 I5 I6 I1 I2 --seeds 42 2027 3407 --stage S3 --config-dir configs/generated/S3 --output-root runs/grid/S3
```

`configs/confirmed_S2.yaml` 是人工确认胜出后保存的完整配置，示例不表示这份文件已经存在。I4/I5/I6/I1/I2 分别为192/224/256/320/384。生成器把路径及 project_root 固定为本机绝对路径，避免嵌套 configs 文件夹改变路径含义；因此请在 Windows 本机生成，不能直接把 Mac 生成的绝对路径 YAML 搬过去。若输入是运行目录中的 config_resolved.yaml，传 `--project-root .` 指定工程根目录。

每次生成目录、训练输出目录必须是新的；已存在会报错，防止覆盖。所有生成的 YAML 都经过正式 load_config 校验。`grid_manifest.json` 记录基线、配置 SHA256 和可执行命令。已有生成目录需要直接按清单里的命令执行，不要对同一目录再次生成。

## 实验映射说明

- A0 固定15轮、warmup5、LR5e-4、EMA .9998、自第1轮启动、clip1、关闭早停并输出raw/EMA验证。配置接口已就绪，仍需在Windows CUDA上训练。EMA尚未启动的早期epoch按raw验证；启动后，validation_weights=auto/ema才使用EMA，checkpoint.selected_weights记录实际来源。
- C1–C6 控制 EMA/裁切；O1–O10 控制训练预算、LR、warmup、层衰减、有效batch；R1–R6 控制WD、drop path、主头与专家头Dropout。
- G0/G2–G8、I3 控制无增强、几何、翻转、Gamma、擦除、Study MixUp、AutoAug、训练统计归一化和CLAHE；尺寸使用I1/I2/I4/I5/I6。
- L1–L6 控制posweight/Focal/ASL。**L7 必须与 unit posweight 的 L2 基线比较**；L7启用温和采样且强制unit，避免强采样与ratio权重叠加。拿ratio基线直接比较L7会同时改变两个因素。
- M1是真正禁用专家及logit混合；M2调整辅助损失；M3均值聚合；M4移除视角embedding；M5只解冻最后两个stage；M6–M9只用对应AP/PA/LATERAL/LL图像训练和验证。视角过滤后的AUC不能与全valid的总AUC直接相比；融合时用下方固定路由回退。
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

## V5：等权融合、视角路由

```powershell
python scripts/evaluate_grid.py fuse --predictions runs/a/valid_best_predictions.csv runs/b/valid_best_predictions.csv --output reports/equal_valid.csv --report reports/equal_valid.json
python scripts/evaluate_grid.py route --base reports/base_valid.csv --specialist AP=reports/ap_valid.csv --specialist PA=reports/pa_valid.csv --manifest data/manifests/valid_manifest.csv --output reports/routed_valid.csv --report reports/routed_valid.json
```

融合只做固定等权概率平均，严格对齐Study和标签，不搜索权重。路由使用基础预测加所有可用且匹配视角的专家等权平均；缺视角或缺专家预测时回退基础模型，输出覆盖数。`--labels 0 3 7` 只替换指定标签。输入兼容每Study一行的测试 `label_0..9` 和验证 `probability_0..9`，验证文件另含 `target_0..9`。无标签融合输出 `label_0..9`，可直接接提交导出工具。标签专家使用 `fuse --labels 0 3 7`，其余七标签保留第一个基础模型预测。比赛长表submission需继续使用既有提交导出工具。

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
