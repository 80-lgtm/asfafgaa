# 中英文术语对照与语言转换风险

本表服务于后续英文写作，不代表已开始或完成英文终稿。

## 推荐术语

| 中文工作用语 | 推荐英文 | 使用说明 |
|---|---|---|
| 剪切散斑 | shearography / electronic speckle-pattern shearing interferometry (ESPSI) | 首次出现给全称；全文固定一种主称谓。 |
| 剪切散斑图像 | shearographic image / shearogram | 图题可结合数据提供方用法；不要写成普通散斑照片。 |
| 无损检测 | non-destructive testing (NDT) | 不扩写为已验证的工业质量控制效果。 |
| 测量区域 | measurement region | 代码保留 `region`；首次解释是数据元数据中的区域标识，不自行假定等同完整物理试件。 |
| region 不相交划分 | region-disjoint split | 仅指 region 不跨 train/val/test，不称 specimen-independent。 |
| 状态级图像样本 | state-specific image sample | 指 `(region, type)` 的中位时间帧；与唯一 region 数区分。 |
| 全局形变正常样本 | defect-free samples with global deformation | 优先于含糊的 stripe normal；首次对应 `Good_stripes`。 |
| 正常参考库 / 记忆库 | nominal reference bank / memory bank | nominal 表示无缺陷参考；全文避免 normal/nominal 无规则混用。 |
| 正常态覆盖 | coverage of nominal states | 不使用“正常态多样性是决定性因素”。 |
| 轻量 | lightweight | 必须有结构/训练需求语境；不自动等于 low-latency。 |
| 冻结骨干网络 | frozen backbone | 本文未进行端到端训练。 |
| coreset 最近邻检测器 | coreset nearest-neighbour detector | 选择英式或美式拼写后全文统一。 |
| 事后条纹抑制 | post-hoc stripe suppression | 主配置 alpha=0，不能描述成生效的核心模块。 |
| 频域上下文 | frequency-domain context | 仅补充探索，不称已验证物理机制。 |
| 误报率 | false positive rate (FPR) | 明确分母和阈值；8 个 Good_stripes 中 2 个误报即 0.25。 |
| bbox 弱标注命中 | bounding-box hit under weak annotations | 不译为 segmentation accuracy / localization IoU。 |
| 单例定位示例 | qualitative localization example | 不写成代表性或典型性已证实。 |
| CPU-only 原型 | CPU-only prototype | 不等于 deployable / real-time / production-ready。 |
| 配对 bootstrap | paired bootstrap | 重采样单位必须明确；当前主结果尚未完成。 |

## 高风险措辞替换

| 避免 | 建议 | 原因 |
|---|---|---|
| prove / demonstrate causality | was associated with / we observed under this split | 消融同时改变正常状态、状态帧数量与实际 coreset，不能单独归因。 |
| decisive factor | a substantial observed difference in this evaluation | 防止超出单数据集证据。 |
| significant improvement | higher AUROC / lower FPR in this run | `significant` 易被理解为统计显著；目前未给 CI。 |
| robust stripe suppression | exploratory frequency-domain variant | 频域结果伴随召回下降且样本少。 |
| leakage-free | region-disjoint with respect to the recorded region identifier | 绝对“无泄漏”忽略同试件、相邻采集等残余相关。 |
| independent test regions | unique recorded regions / state-specific samples | 19 个 region 有两个正常状态，样本不全是统计独立观测。 |
| outperforms PatchCore/EfficientAD | a simplified coreset baseline | 未运行同协议外部基线。 |
| localization accuracy | bounding-box hit rate | bbox 非像素掩码，单点命中规则较宽松。 |
| suitable for industrial deployment | potentially relevant to resource-constrained evaluation | 没有端到端时延、生产环境或设备验证。 |

## 数字与统计表达检查

- `0.25` 必须同时保留 Good_stripes `n=8`（2/8）；`0.125` 正常总 FPR 对应 2/16。
- 测试集写“115 个状态级图像，105 个唯一 region”；不能只写“115 个独立区域”。
- 不把同一固定划分上的差值写成置信区间、显著性或跨分布可重复性。
- 主配置 alpha=0 时，不得把 Stripe-Aware 抑制模块写成最终主贡献。
- 摘要、讨论、结论与应用价值使用同一强度，不在结尾重新升级为因果或部署结论。
