"""Tiny ViT 的运行入口。

模型实现已经按职责拆分到同级 vit 包中。本文件保留两项作用：
1. 支持直接运行，快速检查数据、模型、损失函数、优化器和调度器；
2. 继续兼容之前从 tiny_vit.py 导入各个学习组件的写法。
"""

import torch

# 作为 vision_transformer_practice.tiny_vit 导入时使用相对导入；
# 直接运行 python tiny_vit.py 时使用同级顶层导入。
try:
    from .vit import *  # noqa: F403
    from .vit import __all__ as _vit_exports
except ImportError:
    from vit import *  # noqa: F403
    from vit import __all__ as _vit_exports


__all__ = [*_vit_exports, "main"]


def main():
    """
    作用：
        读取一个 CIFAR-10 batch，检查拆包后的完整 Tiny ViT、
        交叉熵损失、AdamW 和余弦退火能否正常协作。

    参数：无。

    返回值：无；检查结果直接打印到终端。
    """
    train_loader, validation_loader, test_loader, class_names = (
        create_cifar10_dataloaders()  # noqa: F405
    )

    # iter(train_loader) 返回 DataLoader 迭代器；
    # next(...) 返回第一个 batch，即 (images, labels)。
    images, labels = next(iter(train_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    model = TinyViT().to(device)  # noqa: F405
    logits = model(images)

    criterion, optimizer, scheduler = create_training_components(model)  # noqa: F405
    loss = criterion(logits, labels)

    print(f"训练集样本数：{len(train_loader.dataset)}")
    print(f"验证集样本数：{len(validation_loader.dataset)}")
    print(f"测试集样本数：{len(test_loader.dataset)}")
    print(f"类别名称：{class_names}")
    print(f"一个 batch 的图片 shape：{tuple(images.shape)}")
    print(f"一个 batch 的标签 shape：{tuple(labels.shape)}")
    print(f"计算设备：{device}")
    print(f"Encoder Block 数量：{model.encoder.num_blocks}")
    print(f"完整 Tiny ViT 输出 logits shape：{tuple(logits.shape)}")
    print(f"当前 batch 的交叉熵损失：{loss.item():.4f}")
    print(f"损失函数：{criterion.__class__.__name__}")
    print(f"优化器：{optimizer.__class__.__name__}")
    print(f"初始学习率：{optimizer.param_groups[0]['lr']}")
    print(f"权重衰减：{optimizer.param_groups[0]['weight_decay']}")
    print(f"学习率调度器：{scheduler.__class__.__name__}")
    print(f"余弦退火周期 T_max：{scheduler.T_max}")
    print(f"最小学习率 eta_min：{scheduler.eta_min}")


# 只有直接运行 tiny_vit.py 时才执行 main；被其他文件导入时不会自动读取数据。
if __name__ == "__main__":
    main()
