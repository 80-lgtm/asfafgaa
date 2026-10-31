# 剪切散斑图像异常检测中全局形变正常样本的作用：基于轻量记忆库的 region 不相交评估

> 中文工作稿，非投稿终稿。数值来自 `outputs/region/region_report.json` 与
> `outputs/region/uncertainty_report.json`。四个结果图经用户委托选择 A，允许插入
> 工作稿，但仍需投稿前人工终审；英文稿见 `manuscript_draft_en.md`。

## 摘要

剪切散斑图像中的全局形变条纹可能与局部缺陷响应同时出现。仅用无全局形变的正常
样本构建参考库，可能使异常检测器将无缺陷的形变态判为异常。此外，同一测量区域的
重复采集可跨官方数据划分出现，使按图像报告的指标难以解释为新物理区域上的性能。
本文使用 SADD v1.1.0，在保持各 region 不跨训练、验证和测试集的前提下，评估一种
由冻结 ResNet-18、贪心 k-center coreset 和最近邻距离组成的轻量检测流程。每个
`(region, type, split)` 组合保留中位时间帧；19 个 region 因具有两种正常状态而保留
两个状态级样本。主配置将 Good_clean 与 Good_stripes 均纳入正常参考库，coreset
大小为 2,000。在 115 个测试状态级样本（105 个唯一 region）上，图像级 AUROC 为
0.956，验证集阈值下 faulty 召回为 0.909，正常样本误报率为 0.125；8 个
Good_stripes 测试样本的误报率为 0.25。排除 Good_stripes 后，对应 AUROC 为 0.843，
Good_stripes 误报率为 0.875。该消融同时改变正常状态组成、状态帧数量和实际 coreset
大小，尚不能分离这些因素的影响。事后梯度条纹抑制未带来增益；频域变体仅作为补充探索。本文提供
region 不相交的复评流程、正常参考样本消融及 bbox 弱标注定位分析，不主张跨试件、
跨设备泛化或工业部署有效性。

**关键词：** 剪切散斑；异常检测；正常参考库；coreset；区域不相交划分；弱标注定位

## 1. 引言

剪切散斑（shearography）可通过干涉条纹响应辅助无损检测。SADD 提供受控条件下的
聚合物—橡胶粘接层测量，包含 Faulty、Good_clean 和 Good_stripes 三类图像。其中
Good_stripes 不含粘接不连续，但包含用于形成相位跳变的全局形变[1,2]。因此，正常
参考库是否覆盖这种状态，是应用基于正常样本的异常检测方法时必须处理的设计问题。

已有剪切散斑研究已比较仅含无全局形变正常样本与额外包含全局形变正常样本的训练
子集[2]。因此，本文不把“将 Good_stripes 纳入正常库”宣称为首次提出。本文关注的是：
在更严格地区分重复测量区域的评估协议下，这一参考库设计选择对应怎样的检测与误报
表现，以及一个不需要 GPU 训练的简化流程能否提供可复现的应用基线。

本文的工作包括：

1. 构建 region 不跨 split 的清单，并显式说明同 region 多正常状态带来的统计层级；
2. 在同一轻量 coreset 最近邻流程中比较有、无 Good_stripes 的正常参考库；
3. 如实报告梯度条纹抑制的负结果及频域变体的探索性权衡；
4. 使用专家 bbox 进行弱标注命中分析，但不将其等同于像素级分割精度。

## 2. 相关工作

PatchCore 使用正常 patch 特征记忆库和 coreset 压缩进行工业图像异常检测[3]；PaDiM
使用预训练 CNN 的 patch 嵌入与多元高斯分布表示正常类[4]；EfficientAD 通过轻量
student–teacher 路线追求高效异常检测[5]。这些方法提供了方法参照，但本文未在相同
SADD 划分上实现和运行 PaDiM 或 EfficientAD，故不报告与它们的数值优劣。

本文采用受 PatchCore 启发的简化实现，而非对原方法的完整复现。主要差别包括：仅使用
ResNet-18 `layer3` 网格、以普通欧氏最近邻距离评分、以 95 分位聚合图像分数，并未
复现原论文的全部主干、邻域聚合、空间平滑和定位处理。

SADD 的数据集论文指出，同一物理缺陷区域的重复采集可能跨官方 split，官方划分不是
物理缺陷实例不相交的[1]。本文据此进行 region 原子化复评。已有剪切散斑异常检测研究
则已考察全局形变正常样本的加入[2]；本文将贡献限定为新的复评协议及其下的消融和
弱标注分析，不据此声称普遍优越性。

## 3. 材料与方法

### 3.1 数据、许可与引用

使用 SADD v1.1.0，Zenodo DOI 为 `10.5281/zenodo.18301911`。本地归档包含
10,173 个图像条目，类型为 Faulty、Good_clean 和 Good_stripes；专家标注为 bbox、CSV
和 YOLO 格式，未提供像素级掩码[1]。归档 MD5 为
`3b694539f8fa12bc12af2ee4f939f064`，与 Zenodo 页面一致。数据记录与数据集论文
分别引用，详见 `REFERENCES_VERIFIED_CN.md`。

### 3.2 样本选择与 region 原子化划分

先按 `(region, type, split)` 选取中位时间帧，得到 1,131 个图像条目、1,103 个唯一
region。多出的 28 个条目来自 19 个同时具有 Good_clean 与 Good_stripes 状态的正常
region，不是故障样本的重复复制。对每个 region 保留其所有状态条目，并将整个 region
分入同一 split：Faulty 保留官方 split；跨 split 的正常 region 按 `test > val > train`
的既定优先级归组。修正后 train/val/test 的 region 交集均为 0。

该操作保证的是 **region 不跨 split**，并不保证每个指标中的样本都来自不同 region。
19 个多状态正常 region 仍各自贡献两个状态级样本；AUROC、召回和误报率以状态级图像
样本计算。文中所有样本量据此报告，不把 115 个测试图像误写为 115 个独立 region。

### 3.3 特征与轻量参考库

灰度图像转换为三通道并直接缩放至 224×224。使用冻结的 ImageNet-1K v1 ResNet-18[6]，
取 `layer3` 输出，形成 14×14 网格，每个位置为 256 维特征。训练正常样本的所有网格
特征通过随机起点、贪心 k-center 规则压缩为 2,000 个特征，随机种子为 42。

对于测试图像第 \(i\) 个网格特征 \(f_i\)，局部异常分数定义为

\[
d_i=\min_{m\in\mathcal M}\lVert f_i-m\rVert_2,
\]

其中 \(\mathcal M\) 为 coreset。图像级分数为局部分数的 95 分位数。流程不进行
GPU 训练；这并不自动证明其满足工业实时或在线部署要求。

### 3.4 正常库消融与上下文变体

完整正常库包含训练集中的 Good_clean 与 Good_stripes；受限正常库仅包含 Good_clean。
两种设置均固定 coreset 大小、特征层和评分规则。梯度上下文和频域上下文分别作为探索
性调整项；验证集 AUROC 选择 alpha，相同 alpha 下取较小值。主配置的最优 alpha 为 0，
即不启用事后上下文抑制。阈值仅由验证集的 Youden 指数确定，再固定用于测试集。

### 3.5 弱标注定位

对测试 Faulty 图像，将 14×14 异常图按原图尺寸线性映射到 bbox 坐标。每张图的阈值为
该图 196 个网格分数的 95 分位数；任一达到阈值的网格落入专家 bbox 即记该 bbox
命中。该指标衡量弱标注下的响应重叠，不计算 IoU、Dice 或像素级 AUROC。它对每个
异常图的相对阈值、网格分辨率和“单点即命中”的规则敏感，不能单独证明定位准确。

## 4. 结果

### 4.1 数据规模

| split | Faulty | Good_clean | Good_stripes | 状态级图像合计 |
|---|---:|---:|---:|---:|
| train | 886 | 10 | 9 | 905 |
| val | 98 | 6 | 7 | 111 |
| test | 99 | 8 | 8 | 115 |

测试集含 105 个唯一 region；正常测试集含 6 个唯一 region。测试 Faulty 占多数，
因此总体 AUROC 与误报指标应结合类别构成解释。验证集正常样本仅 13 个状态级图像，
阈值和 alpha 选择可能不稳定。

### 4.2 主配置与正常库消融

| 正常库/上下文 | 验证 AUROC | 测试 AUROC | faulty 召回 | 正常 FPR | Good_stripes FPR |
|---|---:|---:|---:|---:|---:|
| C+S / G（主配置，alpha=0） | 0.980 | 0.956 | 0.909 | 0.125 | 0.250 |
| C+S / F（alpha=1） | 0.981 | 0.970 | 0.838 | 0.063 | 0.125 |
| C / G（alpha=0） | 0.782 | 0.843 | 0.949 | 0.438 | 0.875 |
| C / F（alpha=0） | 0.782 | 0.843 | 0.949 | 0.438 | 0.875 |

注：C=Good_clean，S=Good_stripes，G=梯度上下文，F=频域上下文；所有配置
coreset k=2,000。C/F 与 C/G 均在 alpha=0，因此二者等价。

在本次确定性划分与实现中，纳入 Good_stripes 的配置具有更高的测试 AUROC，并在
验证阈值下减少 Good_stripes 误报；同时，faulty 召回从 0.949 降至 0.909。这体现了
误报与漏报之间的权衡，而不是所有指标的一致改善。

两种正常库均覆盖同样的 10 个训练正常 region。仅 Good_clean 时有 10 个状态级图像、
1,960 个候选 patch，因此实际 coreset 为 1,960；加入同一批 region 中 9 个已有的
Good_stripes 状态后有 19 个状态级图像、3,724 个候选 patch，实际 coreset 为 2,000。
对比未增加训练 region 身份，但同时改变状态组成、状态帧数量和实际 coreset 大小。
等状态帧、固定实际 coreset 的敏感性分析见 4.4；这些补充分析仍不构成跨数据集的
因果证明。

### 4.3 上下文变体

梯度上下文在主正常库下选择 alpha=0，未显示增益。频域上下文选择 alpha=1，测试
AUROC 从 0.956 变为 0.970，Good_stripes FPR 从 0.250 变为 0.125，但 faulty 召回
从 0.909 降至 0.838。由于测试正常样本很少、配置经同一数据流程探索且尚无不确定性
分析，这些数值仅作为补充探索，不构成频域方法更优或条纹鲁棒性已验证的结论。

### 4.4 region 级不确定性与匹配对照（补充分析）

以唯一 region 为统计单元重新聚合测试指标：每 region 取最大图像分数，按 any-positive
规则判定误报与召回。region 级点估计为 AUROC=0.894（状态级为 0.956）、faulty region
召回=0.909、正常 region any-FPR=0.333（2/6）、Good_stripes region any-FPR=0.333。
仅 6 个正常测试 region；10,000 次 cluster bootstrap（种子 20261031）给出 AUROC 95%
区间 [0.764, 0.983]、正常 any-FPR [0, 0.80]、stripe any-FPR [0, 0.80]。区间极宽，
说明正常样本太少，无法得出稳定结论；19 次 bootstrap 因只抽到单类而跳过。

在 10 个状态帧、实际 coreset 1,960 的匹配条件下（clean-only 对比 5 clean + 5 stripes，
10 个随机种子，stripes 从 9 个帧有放回抽样），加 stripes 后测试 AUROC 一致高于
clean-only（0.915–0.968 vs 0.830–0.864），stripe FPR 一致更低（0.125–0.625 vs
0.875–1.000）；但 faulty 召回随抽帧和阈值变化。该敏感性分析控制了帧数与 coreset，方向与主
消融一致，但幅度随种子变化，不构成因果证明。

### 4.5 bbox 弱标注分析

主配置在测试 Faulty 图像的 165 个专家 bbox 中命中 134 个，命中率为 0.812。定位
示例见 `outputs/region/figure_heatmap_example.png`；单例仅用于核对坐标和展示响应，不代替
整体评估，也不表示像素级定位精度。

## 5. 讨论

本研究的应用信息不是“提出一种通用最优模型”，而是提示正常参考库构建和重复测量
区域的划分方式会改变观察到的结果及其解释。对于含全局形变正常状态的数据，仅使用
Good_clean 的参考库可能与实际正常状态覆盖不匹配。然而，当前实验无法区分状态覆盖、
额外状态帧、实际 coreset 大小和划分选择的独立影响，也未提供跨设备或跨试件证据。

Region 原子化是减少同源测量跨 split 的必要处理，但不是物理试件级独立性的充分
条件。后续需要以唯一 region 为重采样或聚合单位进行不确定性分析，并明确同 region
不同正常状态是否应作为配对观测。SADD 的受控材料、加载方式和人工界面不连续也不能
直接代表所有真实 kissing bond 或工业构件[1]。

## 6. 局限

- 仅使用 SADD 一个受控数据集，未验证跨数据集、跨材料、跨设备或真实工业部署；
- 测试集仅 16 个正常状态级样本、6 个唯一正常 region；Good_stripes FPR=0.25
  对应 8 个状态级样本中的 2 个误报，估计不稳定；
- 19 个正常 region 具有两种状态，主要指标按状态级图像计算；region 级聚合与
  cluster bootstrap 已作为补充分析，但不是同 region 双状态的配对因果分析；
- 正常库消融虽保持相同 10 个训练正常 region，但未控制状态帧数量（10 或 19）和
  实际 coreset 大小（1,960 或 2,000）；匹配对照证实方向但幅度不稳；
- 正常测试集仅 6 个唯一 region，region 级 any-FPR 置信区间极宽（[0, 0.80]）；
- 未实现同协议的 PaDiM、EfficientAD 等外部基线，也未复现完整 PatchCore；
- bbox 命中为弱标注指标，不能替代像素级定位或缺陷尺寸测量；
- CPU-only 运行不构成端到端时延、内存峰值或在线部署验证。

## 7. 结论

在本文的 SADD region 不相交复评流程中，包含 Good_stripes 的轻量正常参考库与更高
的测试 AUROC 和更低的 Good_stripes 误报相伴，但伴随一定召回损失。该观察支持进一步
研究正常参考状态覆盖与评估协议，而不是宣称新网络结构或普遍性能优势。梯度条纹抑制
的负结果和频域探索的权衡均需保留。未来优先任务是以唯一 region 为统计单位补充
扩充正常 region 统计功效，并增加同协议外部基线。

## 数据与代码可用性

SADD v1.1.0 可通过 Zenodo DOI `10.5281/zenodo.18301911` 获取。本文的 region
清单、代码、报告和图件脚本计划在真实仓库中提供；当前尚无本文项目的公开仓库链接
或 DOI，不得写成“已公开”。完整草稿见 `DATA_AVAILABILITY_DRAFT_CN.md`。

## 伦理、作者与利益声明（待作者完成）

本工作未开展新的人体、动物或光学采集实验，而是分析公开数据集。作者信息已补充：
张永康（独立第一作者兼通讯作者），南开大学，ORCID 0009-0008-6188-1469，
邮箱 2413614@mail.nankai.edu.cn；CRediT 贡献：一人完成全部角色；利益冲突：无；
资助：无。

## 参考文献

采用下列已核验记录，最终格式随目标期刊调整。核验证据见
`REFERENCES_VERIFIED_CN.md`。

[1] Krause, J., Schuler, N., Schuth, M., von Freymann, G. Shearographic anomaly
detection dataset (SADD). *Machine Vision and Applications* 38, 12 (2027; online
26 October 2026). https://doi.org/10.1007/s00138-026-01948-0

[2] Plassmann, J., Schuler, N., von Freymann, G., Schuth, M. Unsupervised Learning
for Industrial Defect Detection: A Case Study on Shearographic Data. *Artificial
Intelligence XLII*, LNAI 16302, 316–329 (2025).
https://doi.org/10.1007/978-3-032-11442-6_22

[3] Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., Gehler, P. Towards
Total Recall in Industrial Anomaly Detection. CVPR 2022.
https://doi.org/10.1109/CVPR52688.2022.01392

[4] Defard, T., Setkov, A., Loesch, A., Audigier, R. PaDiM: A Patch Distribution
Modeling Framework for Anomaly Detection and Localization. *Pattern Recognition.
ICPR International Workshops and Challenges*, LNCS 12664, 475–489 (2021).
https://doi.org/10.1007/978-3-030-68799-1_35

[5] Batzner, K., Heckler, L., König, R. EfficientAD: Accurate Visual Anomaly
Detection at Millisecond-Level Latencies. WACV 2024.
https://doi.org/10.1109/WACV57701.2024.00020

[6] He, K., Zhang, X., Ren, S., Sun, J. Deep Residual Learning for Image Recognition.
CVPR 2016, 770–778. https://doi.org/10.1109/CVPR.2016.90

[7] Plaßmann, J., Schuler, N., von Freymann, G., Schuth, M. Shearographic Anomaly
Detection Dataset (SADD), v1.1.0. Zenodo (2026).
https://doi.org/10.5281/zenodo.18301911
