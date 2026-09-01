# Swin Transformer 实践

本目录单独记录 Swin Transformer 实践，不与 `swin_transformer_basic_knowledge/` 的概念笔记混放。

学习顺序：

1. `01_pretrained_swin_t_inference_and_shape_trace.py`：运行 torchvision 预训练 Swin-T，并把真实输出与四阶段理论 shape 对齐。
2. `02_load_cifar100.py`：下载 CIFAR-100，固定划分训练/验证/测试集，并创建后续模型可直接复用的 DataLoader。
3. 后续：构建自己的图像分类网络。

实践阶段直接使用可运行的 Python 文件，重点概念和 shape 写在代码旁边的中文注释中，不再创建实践 Notebook。
