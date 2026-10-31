# M7 图件地图、图注与追溯记录

状态：4 个结果图已用真实输入通过 Python 渲染并目检；用户委托助手选择 A，允许插入
工作稿。该代理决定不替代投稿前人工终审。NDT&E International 为拟定期刊，具体栏宽、
字体和文件规范因官方指南访问受限而仍待核实。

## 图件地图与候选图注

| 临时图号 | 建议位置 | 文件与类型 | 候选图注 | 状态 |
|---|---|---|---|---|
| 方法图（未编号） | 方法 3.2–3.4 之后 | `figure_method_overview`（规划）+ `figure_method_module`（规划）；EXPLANATORY | 总览：状态选择、region 原子化、冻结特征、coreset 与评分；模块：正常库消融和上下文评分。不得画成 alpha 抑制已被验证有效。 | 未渲染，未纳入英文稿。 |
| Fig. 1 | 结果 4.2 | `outputs/region/figure_roc.png/.pdf`；RESULT | region 不相交划分下主配置的 ROC。AUROC=0.956，基于 115 个状态级测试图像（99 faulty、16 normal；105 个唯一 region）。这不是唯一 region 重采样的置信区间。 | 已插入英文工作稿，待投稿前终审。 |
| Fig. 2 | 结果 4.2 | `outputs/region/figure_auc_bars.png/.pdf`；RESULT | 验证与测试 AUROC。C=Good_clean，S=Good_stripes，G=gradient context，F=frequency-domain context。所有配置请求 k=2,000，实际大小见正文。y 轴从 0.7 起。 | 已插入英文工作稿，待投稿前终审。 |
| Fig. 3 | 结果 4.2 | `outputs/region/figure_stripe_fpr.png/.pdf`；RESULT | 验证集阈值下测试正常样本误报率。Good_stripes n=8 个状态级图像，所有正常状态级图像 n=16；正常测试集仅 6 个唯一 region。小样本下不提供显著性含义。缩写同 Fig. 2。 | 已插入英文工作稿，待投稿前终审。 |
| Fig. 4 | 结果 4.5 | `outputs/region/figure_heatmap_example.png`；RESULT | SADD 测试图像 `000161.png`（region `112503`）：原图、14×14 异常图放大显示、专家 bbox。异常图使用主配置 alpha=0；仅作坐标与响应的定性展示，不表示像素级精度或样本代表性。 | 已插入英文工作稿，待投稿前终审。 |

## 真实输入、代码与运行证据

- 数值来源：`outputs/region/region_report.json`；SHA-256
  `e4398c86338d2c150adbf8d2b81a5c1b86889c0e051d8d4015b3531ddfe967ac`。
- 划分来源：`data/processed_region/manifest.csv`；SHA-256
  `3a8b2e3c266bc86194fbcb988a358cea5731b0108afda3d18bb98e1458f7c561`。
- 原始归档：`data/raw/SADD_v1.1.zip`；MD5 与 Zenodo 一致。定位图直接读取归档图像与
  `SADD/labels/faulty/expert_annotations/labels.csv`。
- 特征来源：`outputs/l2/cache/<image>.npz`；这些缓存用于 region 复评，缓存目录名不表示
  采用旧官方 split。主配置由训练正常样本重新建立 k=2,000 coreset（seed=42）。
- 图形代码：`scripts/make_region_figures.py`；ROC 计算来源为
  `scripts/run_region_evaluation.py`。没有手工向绘图脚本填入实验指标。
- 实际执行：2026-10-31 在工作区运行
  `conda run -n perovskite python -m scripts.make_region_figures`，成功输出
  `figures written to outputs\region`。本轮未重跑主实验、未改变报告数值。

## 坐标、内容与视觉 QA

- 定位样例原图尺寸为 1920×1050、灰度；bbox 为 `(611, 7, 1138, 364)`，在图像范围内。
  缓存网格为 `(196,256)`、上下文为 `(196,)`，即 14×14。
- 热力图使用与特征一致的 14×14 网格，以原图宽高为显示范围。原图直接缩放至 224×224
  后提特征，横纵方向采用相同的直接缩放关系；bbox 覆盖处与高响应在该样例目视一致。
- 全量专家 bbox 坐标范围检查未发现越界；这不等于标注语义、图像配准或全部热图正确。
- PNG 尺寸：ROC 1000×800，两个柱状图各 1400×800，定位示例 2400×680；均约 200 dpi。
  ROC 与柱状图另有 PDF；定位图目前只有 PNG。
- 已目检标签、图例、裁切和 bbox 对齐。柱状图缩写已缩短且需在图注解释；坐标轴未从零
  开始已明确披露。未在目标期刊实际栏宽和灰度打印条件下完成最终检查。

## 审核卡（用户委托决定）

- 内容一致性：数值与报告一致；定位图仅单例。
- 可追溯性：结果图通过；方法图尚未生成，不适用。
- 最终尺寸/期刊格式：待目标期刊。
- 用户委托助手决定，选择 `A`：允许插入英文工作稿。
- `manuscript_insertion_status=allowed_for_working_draft`；投稿前仍需人工终审。
