# Zenodo DOI 获取步骤（针对 GitHub 仓库）

## 前提

- 已有 GitHub 账号，仓库已创建（当前为空）。
- 已有 ORCID（张永康：0009-0008-6188-1469），建议用 ORCID 登录 Zenodo 以自动关联。

## 步骤

### 1. 上传代码到 GitHub

- 将 `001/2` 内所有文件复制到仓库根目录。
- 在 GitHub 网页点击 **Add file → Upload files**，拖拽整个文件夹，或安装 GitHub Desktop 后本地提交。
- 提交信息建议：`Initial release: region-disjoint SADD anomaly detection evaluation`。

### 2. 创建 Release

- 仓库页面右侧点击 **Releases → Draft a new release**。
- Tag 建议：`v1.0.0`；Title 建议：`v1.0.0 Region-disjoint SADD evaluation`。
- Description 简述：数据版本、核心结果、复现说明位置。
- 点击 **Publish release**。

### 3. 连接 Zenodo

- 打开 `https://zenodo.org`，用 ORCID 或 GitHub 登录。
- 右上角头像 → **GitHub** → 找到你的仓库，打开 **On** 开关。
- 回到 GitHub 再发一次 Release（或编辑已有 Release 保存），Zenodo 会自动抓取并生成 DOI。
- 在 Zenodo 记录页复制 DOI，格式如 `10.5281/zenodo.XXXXXXX`。

### 4. 回填到论文

- 将 DOI 填入 `DATA_AVAILABILITY_DRAFT_CN.md` 和 `manuscript_draft_en.md` 的 Data Availability 部分。
- 将仓库链接填入 `PROJECT_STATE.yaml` 的 `project_repository` 字段。

## 注意事项

- Zenodo 记录一旦发布不可删除，但可编辑元数据；确保仓库内容无误后再发布。
- 若后续修改代码，可发新 Release，Zenodo 会生成新版本 DOI，原 DOI 仍指向旧版本。
- 期刊投稿时只需在 Data Availability 中引用 Zenodo DOI 和 GitHub 仓库链接。
