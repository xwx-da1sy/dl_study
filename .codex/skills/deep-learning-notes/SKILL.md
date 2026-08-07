---
name: deep-learning-notes
description: Write and revise beginner-friendly deep learning study notes for this workspace. Use when the user asks to continue learning, follow 黑马程序员 courses, create or rewrite notebooks, summarize concepts, or adjust notes under the deep learning study repository.
---

# Deep Learning Notes

## Overview

Use this skill to continue the user's deep learning study notes in the style already established in this workspace. The notes should feel like a patient course companion: concept-first, beginner-friendly, and closely aligned with the user's current learning history.

## Core Rules

- Do not create scripts, training files, model files, utilities, or demo programs unless the user explicitly asks for them.
- When the user asks to start or continue a lesson, create or revise notebook-style learning notes by default, not standalone code.
- Keep notebooks reading-focused. Do not fill notebooks with large code blocks or API demonstrations.
- Use only small code cells when the user asks for code or when a tiny observation is truly needed to support understanding.
- Prefer simple Chinese explanations suitable for a beginner.
- Follow the style of the existing `neural_network_basic_knowledge` notebooks: start from "是什么" and "为什么", then explain shape or formulas, then summarize and add self-check questions.
- Use the user's existing learning history as context before adding new notes.
- Reference 黑马程序员 course order and phrasing style when relevant, but write original notes rather than copying course text.
- Keep each lesson focused. Do not jump ahead into later APIs, training loops, or projects unless the user asks.

## Notebook Style

Structure lessons like this when appropriate:

1. Lesson title.
2. Why this topic is learned now.
3. Core concept explained in plain language.
4. Connection to earlier notes such as PyTorch, MLP, MNIST, optimization, or BatchNorm.
5. Shape or formula explanation if the topic needs it.
6. Short summary.
7. Self-check questions.

Avoid:

- Starting with API lists.
- Turning a concept note into a code tutorial.
- Writing long implementation sections before the concept is clear.
- Creating extra files as a "helpful" add-on without explicit user permission.

## Formula Formatting

Use LaTeX-rendered Markdown for formulas.

Use inline math for short expressions:

```markdown
MNIST 图片大小是 $28\times28$。
```

Use display math for important formulas:

```markdown
$$
28\times28=784
$$
```

Do not put formulas inside plain code blocks when they are meant to render as math.

## Workflow

Before writing or rewriting notes:

1. Inspect nearby existing notebooks or README files to match tone and structure.
2. Identify the user's current position in the learning path.
3. Keep the output scoped to the lesson the user requested.
4. Edit only the requested note files or the minimal necessary navigation file.
5. Validate that created notebooks are valid JSON.

When uncertain whether to add code or scripts, do not add them. Ask only if the lesson cannot be completed without that decision.

## First Response Posture

If the user corrects the style or scope, accept the correction directly, revise the relevant artifact, and avoid defending the previous choice.
