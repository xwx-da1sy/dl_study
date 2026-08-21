"""使用与 TinyViT 完全相同的训练流程训练 CIFAR-10 CNN。"""

import argparse
from pathlib import Path

import torch

try:
    from vision_transformer_practice.cnn_baseline import CNNBaseline, CNN_CHECKPOINT_PATH, CNN_HISTORY_PATH
    from vision_transformer_practice.vit import (
        BATCH_SIZE,
        LABEL_SMOOTHING,
        LEARNING_RATE,
        LOG_INTERVAL,
        MIN_LEARNING_RATE,
        MIXUP_ALPHA,
        NUM_EPOCHS,
        NUM_WORKERS,
        WEIGHT_DECAY,
        create_cifar10_dataloaders,
        create_training_components,
        evaluate,
        fit,
        set_random_seed,
    )
except ImportError:
    from cnn_baseline import CNNBaseline, CNN_CHECKPOINT_PATH, CNN_HISTORY_PATH
    from vit import (
        BATCH_SIZE,
        LABEL_SMOOTHING,
        LEARNING_RATE,
        LOG_INTERVAL,
        MIN_LEARNING_RATE,
        MIXUP_ALPHA,
        NUM_EPOCHS,
        NUM_WORKERS,
        WEIGHT_DECAY,
        create_cifar10_dataloaders,
        create_training_components,
        evaluate,
        fit,
        set_random_seed,
    )


def parse_args():
    """读取参数；默认值与 train_tiny_vit.py 完全一致。"""
    parser = argparse.ArgumentParser(description="训练 CIFAR-10 CNN 公平对照模型")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--min-learning-rate", type=float, default=MIN_LEARNING_RATE)
    parser.add_argument("--mixup-alpha", type=float, default=MIXUP_ALPHA)
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    parser.add_argument("--checkpoint", type=Path, default=CNN_CHECKPOINT_PATH)
    parser.add_argument("--history", type=Path, default=CNN_HISTORY_PATH)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    """复用 TinyViT 的数据、损失、优化器、调度器和训练循环。"""
    args = parse_args()
    set_random_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    train_loader, validation_loader, test_loader, class_names = (
        create_cifar10_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )
    model = CNNBaseline().to(device)
    criterion, optimizer, scheduler = create_training_components(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        min_learning_rate=args.min_learning_rate,
        label_smoothing=LABEL_SMOOTHING,
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print("=" * 70)
    print("CNN 公平对照训练准备完成")
    print(f"设备：{device}")
    print(f"类别：{class_names}")
    print(
        f"训练/验证/测试：{len(train_loader.dataset)}/"
        f"{len(validation_loader.dataset)}/{len(test_loader.dataset)}"
    )
    print(f"可训练参数：{trainable_parameters:,}")
    print(f"epochs：{args.epochs}")
    print(f"batch size：{args.batch_size}")
    print(f"AdamW learning rate：{args.learning_rate}")
    print(f"weight decay：{args.weight_decay}")
    print(f"Label Smoothing：{LABEL_SMOOTHING}")
    print(f"Mixup alpha：{args.mixup_alpha}")
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

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
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
    print("下一步运行 evaluate_cnn_baseline.py 生成同规格评估图。")
    print("=" * 70)


if __name__ == "__main__":
    main()
