# 项目长期笔记

## 工作空间约定
- Skill 存储：用 `.codex/skills/` 目录（OpenAI Codex CLI 格式），不是 WorkBuddy 的 `.workbuddy/skills/`。每个 skill 是 `.codex/skills/<name>/` 目录，含 `SKILL.md`（frontmatter: name + description + 正文规则）+ `agents/openai.yaml`（display_name/short_description/default_prompt）+ 可选 `scripts/`。
- 现有 skill：`deep-learning-notes`（写深度学习笔记）、`render-notebook-formulas`（渲染 notebook 公式）。
- 笔记形式：Jupyter notebook，但用户偏好"纯阅读"——code cell 转成 markdown 用代码块包裹（不执行）。见 05/06/07 三个教学 notebook。
- Python 环境：`D:\PythonWorkSpace\python.exe`（torch 2.11.0+cu128，CUDA 可用，sm_120 兼容）。

## 生成内容前必读 skill（重要约定）
**生成任何 notebook / 笔记 / 内容之前，先读 `.codex/skills/` 下的 SKILL.md，按里面的规则行动。**

### `deep-learning-notes` 关键规则
- 默认写 notebook 笔记，不写独立脚本/训练文件/模型文件（除非用户明确要）
- notebook 不要填大代码块或 API 演示；只在用户要代码或需要小观察时用小 code cell
- 用户偏好纯阅读 notebook（code cell 转成 markdown 用 ``` 代码块包裹，不执行）
- 简单中文，适合初学者
- 按 neural_network_basic_knowledge 风格：是什么 → 为什么 → shape/公式 → 总结 → 自检
- 先看现有 notebook/README 匹配风格再写
- 公式：notebook markdown 用 LaTeX（Jupyter 渲染，`$...$` 行内 / `$$...$$` 独立）；chat 回复用纯文本（不用 `$...$`，维度写 `28 x 28` 不写 `$28\times28$`）

### `render-notebook-formulas` 关键规则
- notebook markdown 里的公式/数学符号/流程说明要转成 LaTeX（MathJax 在 fenced code 里不渲染，要从代码块拿出来）
- `$...$` 行内、`$$...$$` 独立（每个 `$$` 单独一行）、`\begin{aligned}` 多行、`\begin{bmatrix}` 矩阵、`\rightarrow`/`\text{}` 流程
- 中文放 math mode 外，必须放里面用 `\text{...}`
- 张量形状 vs 矩阵乘法要区分：`$B \times h \times N \times d_{\mathrm{head}}$` 是形状，`$(N \times d_k)\cdot(d_k \times N) \rightarrow N \times N$` 是乘法，不要混
- 只改 markdown cell，不动 code cell/输出/metadata/lesson 顺序
- 改完跑 `python .codex/skills/render-notebook-formulas/scripts/audit_notebook_math.py <path>` + JSON 验证 + nbconvert 导出验证
- 记号标准化：`N x D`→`$N \times D$`、`QK^T`→`$QK^{\top}$`、`d_head`→`$d_{\mathrm{head}}$`、`sqrt(d_k)`→`$\sqrt{d_k}$`
