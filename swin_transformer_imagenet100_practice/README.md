# ImageNet-100 四阶段 Swin Transformer 实践

本目录保留原 ImageNet-1K 项目的四阶段 Swin、Attention Pooling、训练、搜索和可视化流程，只把数据集规模改为 ImageNet-100，并将分类头改为100类。

学习顺序：

1. `01_pretrained_swin_t_inference_and_shape_trace.py`：运行 torchvision 的 ImageNet-1K 预训练 Swin-T，并把真实输出与四阶段理论 shape 对齐；该文件只作为结构参照，因此仍输出1000类。
2. `swin/config.py`：集中保存路径、ImageNet-100数据、四阶段网络结构和训练配置。
3. `swin/data.py`：读取 ImageNet-100，从train按类别均衡划分内部验证集，并提供Mixup/CutMix。
4. `swin/__init__.py`：统一导出项目配置、数据、模型、优化器和训练接口。
5. `swin/embedding.py`：把 `224 × 224` 图片转换为 `56 × 56 × 96` Patch tokens。
6. `swin/encoder.py`：实现窗口划分、相对位置偏置、W-MSA、SW-MSA、mask、Swin Block、Patch Merging 和 Stage。
7. `swin/model.py`：组装四阶段自定义 Swin，并继续使用轻量Attention Pooling输出100类logits。
8. `swin/optimization.py`：创建 AdamW，并对需要与不需要 Weight Decay 的参数进行分组。
9. `swin/training.py`：实现 Label Smoothing、Warmup + 余弦退火、单轮训练、验证和最佳模型保存。
10. `search_hyperparameters.py`：固定内部训练/验证划分，搜索学习率和Weight Decay；ImageNet-100 val不参与选择。
11. `train_custom_swin.py`：使用搜索出的最佳参数完整训练300轮，并保存验证集表现最好的模型。
12. `swin/evaluation.py`：统计 Top-1/Top-5、逐类别准确率和混淆矩阵，并绘制训练与预测结果。
13. `evaluate_custom_swin.py`：确定最终配置后，只在ImageNet-100 val上评估一次并生成结果文件。

数据目录必须整理为 torchvision `ImageFolder` 能读取的形式：

```text
data/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...共100个类别目录
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...与train完全相同的100个类别目录
```

默认从train的每个类别固定划出50张图片，组成5,000张内部验证集；val只供最终评估。

四阶段shape：

```text
B × 3 × 224 × 224
→ Stage 1：B × 56 × 56 × 96
→ Stage 2：B × 28 × 28 × 192
→ Stage 3：B × 14 × 14 × 384
→ Stage 4：B × 7 × 7 × 768
→ Attention Pooling：B × 768
→ 分类层：B × 100
```

训练顺序：

1. 运行 `python search_hyperparameters.py`。默认比较3个学习率和2个 Weight Decay，每组先训练100轮。
2. 查看 `results/hyperparameter_search/search_summary.json`，或直接复制脚本最后打印的最佳参数命令。
3. 使用最佳组合运行 `python train_custom_swin.py --learning-rate ... --weight-decay ...`。正式训练默认覆盖完整300轮余弦退火。
4. 最终模型确定后，再运行 `python evaluate_custom_swin.py` 使用ImageNet-100 val。

实践阶段直接使用可运行的 Python 文件，重点概念和 shape 写在代码旁边的中文注释中，不再创建实践 Notebook。
