# 核心参考文献与数据记录核验

核验日期：2026-10-31。以下为工作稿的核心来源，不是完整系统综述；未阅读全文的
条目只用于其官方摘要/元数据支持的定位，不转述全文实验结论。

## 已核实

| 编号 | 来源与用途 | 官方/一手证据 | 当前使用边界 |
|---|---|---|---|
| [1] | Krause, Schuler, Schuth, von Freymann. *Shearographic anomaly detection dataset (SADD)*. Machine Vision and Applications 38, 12 (2027)；2026-10-26 在线。DOI `10.1007/s00138-026-01948-0`。 | [Springer 论文页](https://link.springer.com/article/10.1007/s00138-026-01948-0)：标题、作者、卷/文章号、日期、DOI、官方划分局限及 Data Availability。 | 官方 split 不是物理缺陷实例不相交；论文直接讨论 v1.0.1，不能代替 v1.1.0 数据版本记录。 |
| [2] | Plassmann, Schuler, von Freymann, Schuth. *Unsupervised Learning for Industrial Defect Detection: A Case Study on Shearographic Data*. Artificial Intelligence XLII, LNAI 16302, 316–329 (2025). DOI `10.1007/978-3-032-11442-6_22`。 | [Springer 章节页](https://link.springer.com/chapter/10.1007/978-3-032-11442-6_22)：元数据及摘要明确比较两种正常训练子集。 | 限制“Good_stripes 入库”的新颖性措辞；摘要不足以支持逐项复现其实验数值。 |
| [3] | Roth, Pemula, Zepeda, Schölkopf, Brox, Gehler. *Towards Total Recall in Industrial Anomaly Detection*. CVPR 2022. DOI `10.1109/CVPR52688.2022.01392`。 | [CVF 官方记录](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)核对作者、题名与 CVF 页码 14318–14328；[DOI 记录](https://doi.org/10.1109/CVPR52688.2022.01392)及检索元数据核对 DOI。 | CVF 与 IEEE 页码体系不同，按所选引用样式保留一种并核对；本文不是完整 PatchCore 复现。 |
| [4] | Defard, Setkov, Loesch, Audigier. *PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization*. LNCS 12664, 475–489 (2021). DOI `10.1007/978-3-030-68799-1_35`。 | [Springer 章节页](https://link.springer.com/chapter/10.1007/978-3-030-68799-1_35)：作者、题名、在线日期、页码、书系、摘要。 | 只作方法定位；未在本文 SADD 协议下运行。 |
| [5] | Batzner, Heckler, König. *EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies*. WACV 2024. DOI `10.1109/WACV57701.2024.00020`。 | [CVF 官方记录](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html)核对作者、题名与 CVF 页码 128–138；[IEEE 元数据](https://doi.org/10.1109/WACV57701.2024.00020)核对 DOI。 | 原论文效率硬件与本文 CPU 不同；不得把毫秒延迟套用到本文。 |
| [6] | He, Zhang, Ren, Sun. *Deep Residual Learning for Image Recognition*. CVPR 2016, 770–778. DOI `10.1109/CVPR.2016.90`。 | [CVF 官方记录](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)与 [IEEE DOI 记录](https://doi.org/10.1109/CVPR.2016.90)。 | 用于 ResNet 架构引用，不表示 ImageNet 预训练数据许可已由此条目核实。 |
| [7] | Plaßmann, Schuler, von Freymann, Schuth. *Shearographic Anomaly Detection Dataset (SADD)*, v1.1.0. Zenodo (2026). DOI `10.5281/zenodo.18301911`。 | [Zenodo v1.1.0](https://zenodo.org/records/18301911)：版本、三类样本、bbox 标注、归档 MD5 `3b694539f8fa12bc12af2ee4f939f064`、DOI。 | 与期刊论文分开引用；当前网页文本抓取未显示许可证名称，最终再分发前需人工查看 Rights 字段。 |

## 通用参考数据集（本文未使用）

- [MVTec AD 官方页面](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)：
  超过 5,000 张高分辨率图像、15 类、像素级异常标注，许可 CC BY-NC-SA 4.0，
  明确不允许商业用途。本文没有把其数值或标注混入 SADD 主实验。

## 待核实

- 本文仍需针对“重复测量/实例不相交评估”“正常模态覆盖”“弱标注定位”补充更完整
  文献检索；以上 7 条不能证明研究空白或首次性。
- 目标期刊未定，参考文献数量、作者缩写、页码、在线优先年份和数据引用样式待核对。
- SADD v1.1.0 的 CC BY 4.0 状态在项目前序记录中已核验，但本次网页文本未显示该
  字段；公开衍生清单或图像前需再次人工检查 Zenodo Rights 与署名要求。
- ImageNet 数据与 torchvision 权重的适用条款需分别核验；ResNet 论文不等同于权重或
  数据许可证明。
