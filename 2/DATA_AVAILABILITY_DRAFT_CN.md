# 数据与代码可用性说明（草稿）

状态：公开仓库尚未创建，链接和 DOI 均为待补字段；以下不是已公开声明。

## Data Availability（中文草稿）

本研究分析的原始数据为 Shearographic Anomaly Detection Dataset (SADD) v1.1.0，
可通过 Zenodo 记录 `https://doi.org/10.5281/zenodo.18301911` 获取。本文使用的
原始 ZIP 归档 MD5 为 `3b694539f8fa12bc12af2ee4f939f064`，与该版本页面一致。
数据记录采用 CC BY 4.0【项目前序核验；发布衍生材料前再次人工检查该版本 Rights
字段与署名要求】。SADD 数据集论文与 v1.1.0 数据记录分别引用。

本文派生的 region 清单、region 不相交划分、特征提取与评估代码、消融和定位分析
脚本、图件生成代码，以及对应的数值报告和图件源数据，计划在【公开/机构仓库名称】
提供【真实稳定链接、版本和 DOI 待创建后填入】。派生数据与原始 SADD 的关系、筛选
规则和修改将随仓库说明。当前未声称上述项目材料已经公开。

大型中间特征缓存不默认随仓库分发。代码与运行说明将用于重新生成缓存；当前没有全部
缓存的逐文件校验清单，不能写成“所有缓存均提供校验信息”。公开范围由作者最终确认。

## Data Availability（英文占位草稿）

The raw data analysed in this study are the Shearographic Anomaly Detection Dataset
(SADD), version 1.1.0, available from Zenodo at
https://doi.org/10.5281/zenodo.18301911. The MD5 checksum of the source archive used
in this work is 3b694539f8fa12bc12af2ee4f939f064. The dataset record and the associated
dataset article are cited separately. Any redistribution of derived materials will
retain the applicable attribution, version, licence, and change notices.

## Code Availability（英文占位草稿）

The derived region manifest, region-disjoint splitting code, analysis and evaluation
scripts, numerical reports, and figure-generation source files will be deposited in
[repository name and stable link to be inserted only after deposition]. A version
identifier or DOI will be reported only after it has been issued and verified. Large
intermediate feature caches are not planned for redistribution; the code and documented
workflow can regenerate them. No project repository or DOI is claimed at this draft stage.

## 第三方数据、图像与许可提醒

- SADD 的 CC BY 许可不等于省略署名、版本、许可链接和修改说明；公开派生清单或示例
  图像前核对相应义务。
- 不因关联论文采用 CC BY 4.0，就跳过 v1.1.0 数据记录本身的 Rights 核验。
- ImageNet/ResNet-18 预训练权重是第三方资源；分别保留来源、版本与适用许可，不把
  ResNet 论文引用当成权重或数据许可。
- 本文不默认使用 MVTec AD；若后续加入，须独立遵守其 CC BY-NC-SA 4.0 非商业条款。
