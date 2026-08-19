"""在 CIFAR-10（Resize 到 224×224）上训练 224 ViT。

架构：image 224 / patch 16 / embed 256 / heads 4 / blocks 6 / classes 10。
注意：CIFAR-10 原图 32×32 放大到 224 不增加真实信息，本工程目的是练手 224 ViT 架构流程。
"""

import argparse
from pathlib import Path

import torch

try:
    from .vit_224 import (
        BATCH_SIZE,
        BEST_MODEL_PATH,
        LEARNING_RATE,
        LABEL_SMOOTHING,
        LOG_INTERVAL,
        MIXUP_ALPHA,
        MIN_LEARNING_RATE,
        NUM_EPOCHS,
        NUM_WORKERS,
        TRAINING_HISTORY_PATH,
        WEIGHT_DECAY,
        TinyViT,
        create_cifar10_dataloaders,
        create_training_components,
        evaluate,
        fit,
        set_random_seed,
    )
except ImportError:
    from vit_224 import (
        BATCH_SIZE,
        BEST_MODEL_PATH,
        LEARNING_RATE,
        LABEL_SMOOTHING,
        LOG_INTERVAL,
        MIXUP_ALPHA,
        MIN_LEARNING_RATE,
        NUM_EPOCHS,
        NUM_WORKERS,
        TRAINING_HISTORY_PATH,
        WEIGHT_DECAY,
        TinyViT,
        create_cifar10_dataloaders,
        create_training_components,
        evaluate,
        fit,
        set_random_seed,
    )


def parse_args():
    """
    作用：读取命令行训练参数，便于不改源码直接调整实验配置。

    参数：无；参数来自命令行。

    返回值：args，包含 epoch、batch_size、学习率和检查点路径等配置。
    """
    parser = argparse.ArgumentParser(description="在 CIFAR-10（Resize 224）上训练 224 ViT")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--min-learning-rate", type=float, default=MIN_LEARNING_RATE)
    parser.add_argument("--mixup-alpha", type=float, default=MIXUP_ALPHA)
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BEST_MODEL_PATH,
        help="验证 loss 最低的模型保存位置",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=TRAINING_HISTORY_PATH,
        help="每个 epoch 的训练与验证指标保存位置",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="关闭 CUDA 自动混合精度",
    )
    return parser.parse_args()


def main():
    """
    作用：创建数据、模型和训练组件，完成训练并使用最佳模型评估测试集。

    参数：无；训练配置由 parse_args 从命令行读取。

    返回值：无；训练进度打印到终端，最佳权重保存到 checkpoints_224 目录。
    """
    args = parse_args()
    set_random_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # 允许 CUDA 矩阵乘法使用更高效的实现，适合深度学习训练。
        torch.set_float32_matmul_precision("high")

    train_loader, validation_loader, test_loader, class_names = (
        create_cifar10_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )

    model = TinyViT().to(device)
    criterion, optimizer, scheduler = create_training_components(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        min_learning_rate=args.min_learning_rate,
        label_smoothing=LABEL_SMOOTHING,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("=" * 70)
    print("224 ViT 训练准备完成")
    print(f"设备：{device}")
    print(f"类别：{class_names}")
    print(f"训练/验证/测试：{len(train_loader.dataset)}/"
          f"{len(validation_loader.dataset)}/{len(test_loader.dataset)}")
    print(f"可训练参数：{trainable_parameters:,}")
    print(f"epochs：{args.epochs}")
    print(f"batch size：{args.batch_size}")
    print(f"AdamW learning rate：{args.learning_rate}")
    print(f"weight decay：{args.weight_decay}")
    print(f"余弦退火最低学习率：{args.min_learning_rate}")
    print(f"CUDA 混合精度：{device.type == 'cuda' and not args.no_amp}")
    print(f"最佳模型路径：{args.checkpoint}")
    print(f"训练历史路径：{args.history}")
    print("=" * 70)

    _, best_validation_accuracy = fit(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        checkpoint_path=args.checkpoint,
        history_path=args.history,
        use_amp=not args.no_amp,
        log_interval=args.log_interval,
        mixup_alpha=args.mixup_alpha,
    )

    # 测试集不参与模型选择，只在训练结束后评估一次最佳模型。
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=not args.no_amp,
    )

    print("=" * 70)
    print(f"最佳验证准确率：{best_validation_accuracy * 100:.2f}%")
    print(f"测试集 loss：{test_loss:.4f}")
    print(f"测试集 accuracy：{test_accuracy * 100:.2f}%")
    print("下一步运行 evaluate_vit224.py 生成训练曲线和测试集可视化。")
    print("=" * 70)


if __name__ == "__main__":
    main()
