---
name: render-notebook-formulas
description: Convert plain-text mathematical notation in Jupyter Notebook Markdown cells into reliable LaTeX while preserving prose, code, outputs, and notebook structure. Use for .ipynb formula-rendering fixes, including ASCII dimensions such as N x D, QK^T and d_head notation, matrix-shape derivations, formula-like text fences, and audits for unrendered notebook math.
---

# Render Notebook Formulas

Convert mathematical content in notebook Markdown cells to Jupyter-renderable LaTeX, then verify both the notebook structure and rendered export.

## Workflow

1. Locate the requested `.ipynb` files and inspect nearby notebooks to match their style.
2. Check the worktree before editing. Preserve unrelated and pre-existing changes.
3. Inventory formula-like content in Markdown cells:
   - ASCII dimensions such as `N x D` or `B x h x N x d_head`.
   - Transposes, subscripts, functions, and variables such as `QK^T`, `d_head`, `q_i`, `W_Q`, `sqrt(d_k)`, `score`, and `alpha`.
   - Matrix-shape derivations using `@` or arrows.
   - Formula tables or definitions trapped inside fenced `text` blocks or inline code.
4. Edit contextually. Do not perform an unreviewed repository-wide replacement.
5. Run `scripts/audit_notebook_math.py` on the edited path.
6. Parse every edited notebook as JSON, export it with `jupyter nbconvert`, and inspect the diff.

## LaTeX Rules

- Use `$...$` for variables or short expressions embedded in prose.
- Use `$$...$$` for important standalone formulas. Put each `$$` delimiter on its own line.
- Use `\begin{aligned}...\end{aligned}` for multi-line shape derivations.
- Use `\begin{array}` or `\begin{bmatrix}` for mathematical tables and matrices.
- Keep Chinese explanations outside math mode. When text must appear inside display math, wrap it with `\text{...}`.
- Remove formula content from code fences before adding LaTeX; MathJax does not render inside fenced code.
- Leave conceptual prose diagrams in code fences when they are not mathematical expressions.

Normalize common notation:

- `N x D` -> `$N \times D$`
- `QK^T` -> `$QK^{\top}$`
- `d_head` -> `$d_{\mathrm{head}}$`
- `sqrt(d_k)` -> `$\sqrt{d_k}$`
- `alpha_i` -> `$\alpha_i$`
- `softmax` inside a formula -> `\operatorname{softmax}`

## Shape-Safety Rules

Distinguish tensor dimensions from matrix multiplication.

- A tensor shape is one dimension chain: `$B \times h \times N \times d_{\mathrm{head}}$`.
- A matrix-shape multiplication is two grouped shapes: `$(N \times d_k)\cdot(d_k \times N) \rightarrow N \times N$`.
- Never rewrite a four-axis tensor shape as `(B \times h)\cdot(N \times d_{\mathrm{head}})`.
- Preserve the mathematical meaning of `@`; do not flatten both operand shapes into one ambiguous chain of `\times` symbols.

## Editing Boundaries

- Modify Markdown cell sources only unless the user explicitly requests code changes.
- Preserve code cells, outputs, execution counts, notebook metadata, lesson order, and ordinary prose.
- Keep source arrays valid JSON and retain UTF-8 Chinese text.
- Avoid adding notebook cell IDs or normalizing unrelated metadata solely to silence existing warnings.

## Validation

Run the bundled audit:

```powershell
python .codex/skills/render-notebook-formulas/scripts/audit_notebook_math.py <notebook-or-directory>
```

Then perform structural and export checks:

```powershell
python -c "import json, pathlib; [json.loads(p.read_text(encoding='utf-8')) for p in pathlib.Path('<directory>').glob('*.ipynb')]"
jupyter nbconvert --to html --stdout --log-level ERROR <notebook.ipynb> | Out-Null
git diff --check -- <edited-notebooks>
```

Treat audit hits as review candidates, not automatic proof of an error. Inspect surrounding prose before changing them. Finish only after all requested notebooks parse and export successfully and the diff contains no unrelated edits.
