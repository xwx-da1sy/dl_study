# Swin Transformer 实践

本目录单独记录 Swin Transformer 实践，不与 `swin_transformer_basic_knowledge/` 的概念笔记混放。

学习顺序：

1. `01_pretrained_swin_t_inference_and_shape_trace.py`：运行 torchvision 预训练 Swin-T，并把真实输出与四阶段理论 shape 对齐。
2. `swin/config.py`：集中保存路径、数据、三阶段网络结构和后续训练配置。
3. `swin/data.py`：下载 CIFAR-100，按类别均衡划分训练/验证集，创建 DataLoader，并提供训练 batch 使用的 Mixup/CutMix。
4. `swin/__init__.py`：统一导出项目配置、数据、模型、优化器和训练接口。
5. `swin/embedding.py`：把 `32 × 32` 图片转换为 `16 × 16 × 96` Patch tokens。
6. `swin/encoder.py`：实现窗口划分、相对位置偏置、W-MSA、SW-MSA、mask、Swin Block、Patch Merging 和 Stage。
7. `swin/model.py`：组装三阶段自定义 Swin，并使用轻量 Attention Pooling 输出100类 logits。
8. `swin/optimization.py`：创建 AdamW，并对需要与不需要 Weight Decay 的参数进行分组。
9. `swin/training.py`：实现 Label Smoothing、Warmup + 余弦退火、单轮训练、验证和最佳模型保存。
10. `train_custom_swin.py`：读取可调整参数并启动正式训练；官方测试集不会在网格搜索阶段使用。
11. `swin/evaluation.py`：统计 Top-1/Top-5、逐类别准确率和混淆矩阵，并绘制训练与预测结果。
12. `evaluate_custom_swin.py`：在网格搜索确定最终模型后，评估一次官方测试集并生成结果文件。
13. 后续：设计网格搜索。

实践阶段直接使用可运行的 Python 文件，重点概念和 shape 写在代码旁边的中文注释中，不再创建实践 Notebook。
