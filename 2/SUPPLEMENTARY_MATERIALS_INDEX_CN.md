# 补充材料索引与证据包（工作版）

本索引说明拟随论文提供的材料，不表示已经公开上传或已满足特定期刊的补充材料格式。

| 编号 | 内容 | 工作区来源 | 状态与边界 |
|---|---|---|---|
| S1 | region 原子化清单与 split 重叠统计 | `data/processed_region/manifest.csv`、`data/processed_region/region_split_summary.json` | 已核实；1,131 状态帧、1,103 region、split 重叠为 0。 |
| S2 | 主配置、消融与 bbox 报告 | `outputs/region/region_report.json` | 已核实；主指标为状态级图像指标。 |
| S3 | 唯一 region bootstrap 与匹配敏感性分析 | `outputs/region/uncertainty_report.json` | 已核实；仅 6 个正常测试 region，区间宽；匹配库各 10 帧、coreset 1,960。 |
| S4 | 图件、图注和输入到像素追溯 | `outputs/region/`、`FIGURE_TRACEABILITY_CN.md` | 4 图已生成并插入英文工作稿；方法图未制作。 |
| S5 | 复现环境、命令与缓存边界 | `REPRODUCIBILITY.md`、`requirements.txt` | 已有；无逐文件缓存校验清单，未声称缓存公开。 |
| S6 | 核心文献核验及语言风险 | `REFERENCES_VERIFIED_CN.md`、`TERMINOLOGY_TRANSLATION_CN.md` | 核心来源已核验；完整综述和期刊样式待补。 |
| S7 | 数据/代码可用性与作者声明 | `DATA_AVAILABILITY_DRAFT_CN.md`、`manuscript_draft_en.md` | 仓库与作者信息为占位，不得视为已完成。 |

## 发布前检查

- 核实 SADD v1.1.0 的 Rights 字段、署名、版本和修改说明，避免无审查重传原始归档。
- 选择项目代码许可，创建真实仓库后补稳定链接和版本/DOI。
- 统一英文稿、补充表和图注的状态级/region 级指标，禁止混用 AUROC 0.956 与 0.894。
- 官方补充材料格式、文件大小和匿名要求待作者人工核对 NDT&E 指南。
