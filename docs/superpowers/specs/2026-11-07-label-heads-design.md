# 标签选择图像与空间区域：第一批设计

用户已批准实施第一批，保留当前工程和旧模型。只修改本地文件，不训练比赛数据，不提交或推送 Git。

## 范围

- C0：当前共享 Study attention、10标签主头、0/3/7专家。
- C1：关闭专家的共享 Study attention 对照。
- H1：全局图像向量、逐标签 Study attention、逐标签线性分类。
- H2：保留 ConvNeXt 空间特征、10个学习 query、共享 Study attention、逐标签线性分类。
- H3：H2 的空间特征提取，改为逐标签 Study attention；接口提供，待 H1/H2 验证后安排训练。
- 新结构关闭视角嵌入、视角过滤、旧专家和 MixUp。默认字段保留旧参数名与计算路径。
- DINOv2 和自监督续训不在本批实现；提供后续原版通用权重链接。

## 数据流与兼容

旧路径保持 Tensor[N,C] -> Tensor[B,C] -> 主/专家 logits。H1 聚合为 Tensor[B,10,C]。查询路径从 NCHW 特征图提取 Tensor[N,10,D]，并保留同次骨干前向的全局向量供共享图像权重使用。查询结构当前只支持 ConvNeXt；不把池化向量当作空间 tokens。

模型始终返回主 logits[B,10] 与专家 logits[B,0或3]，标签顺序、损失 mask、EMA 和 CSV 格式不变。推理与恢复校验覆盖新结构字段，旧配置缺省字段按历史默认补齐。新模块不参与旧模型的 state_dict。

## 训练协议

完整 L5 对照：256、15轮、warmup5、LR5e-4、WD.1、layer_decay.65、drop_path.1、G1、Focal2、ratio/cap20、EMA.9998、clip_grad=null；关闭所有早停，统一 EMA 选权重，种子42/2027/3407，有效Study batch16。按硬件调整 micro batch、worker、chunk、AMP，但记录差异；不声称跨设备位级一致。

## 验收

真实小网络测试覆盖变长 Study、图像顺序不变性、标签独立权重、空间内容敏感性、所有新模块的梯度、断点/EMA/推理一致性；旧配置与旧 state_dict 严格兼容。CPU 与可用 MPS 做合成数据 smoke；CUDA 测试无设备时明确标记未运行。完整比赛训练不在本次运行。
