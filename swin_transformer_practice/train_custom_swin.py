"""在 CIFAR-100 上训练自定义 Swin；测试集不参与训练和模型选择。"""

import argparse
from pathlib import Path

import torch

from swin import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CUTMIX_ALPHA,
    CUTMIX_PROBABILITY,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    LOG_INTERVAL,
    MAX_GRAD_NORM,
    MIN_LEARNING_RATE,
    MIXUP_ALPHA,
    NUM_EPOCHS,
    NUM_WORKERS,
    TRAINING_HISTORY_PATH,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    CustomSwin,
    create_cifar100_dataloaders,
    create_loss_function,
    create_optimizer,
    create_warmup_cosine_scheduler,
    fit,
    set_random_seed,
)


def parse_args():
    """读取训练参数；网格搜索时可直接传入不同参数而不修改源码。"""
    parser = argparse.ArgumentParser(
        description="在 CIFAR-100 上训练自定义 Swin"
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=MIN_LEARNING_RATE,
    )
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=LABEL_SMOOTHING,
    )
    parser.add_argument("--mixup-alpha", type=float, default=MIXUP_ALPHA)
    parser.add_argument("--cutmix-alpha", type=float, default=CUTMIX_ALPHA)
    parser.add_argument(
        "--cutmix-probability",
        type=float,
        default=CUTMIX_PROBABILITY,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=MAX_GRAD_NORM,
    )
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BEST_MODEL_PATH,
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=TRAINING_HISTORY_PATH,
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="关闭 CUDA 自动混合精度",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    train_loader, validation_loader, test_loader, _ = (
        create_cifar100_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )
    model = CustomSwin().to(device)
    criterion = create_loss_function(args.label_smoothing)
    optimizer = create_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = create_warmup_cosine_scheduler(
        optimizer=optimizer,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_learning_rate=args.min_learning_rate,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print("=" * 70)
    print("自定义 Swin 训练准备完成")
    print(f"设备：{device}")
    print(
        "训练/验证/官方测试："
        f"{len(train_loader.dataset)}/"
        f"{len(validation_loader.dataset)}/"
        f"{len(test_loader.dataset)}"
    )
    print("官方测试集不会在本训练脚本中使用")
    print(f"可训练参数：{trainable_parameters:,}")
    print(f"epochs：{args.epochs}，warmup：{args.warmup_epochs}")
    print(f"batch size：{args.batch_size}")
    print(f"AdamW learning rate：{args.learning_rate}")
    print(f"weight decay：{args.weight_decay}")
    print(f"最低学习率：{args.min_learning_rate}")
    print(f"Label Smoothing：{args.label_smoothing}")
    print(
        f"Mixup alpha：{args.mixup_alpha}，"
        f"CutMix alpha：{args.cutmix_alpha}，"
        f"CutMix 选择概率：{args.cutmix_probability}"
    )
    print(f"CUDA 混合精度：{device.type == 'cuda' and not args.no_amp}")
    print(f"最佳模型路径：{args.checkpoint}")
    print(f"训练历史路径：{args.history}")
    print("=" * 70)

    result = fit(
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
        max_grad_norm=args.max_grad_norm,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_probability=args.cutmix_probability,
    )

    print("=" * 70)
    print(f"最佳 epoch：{result['best_epoch']}")
    print(
        "最佳验证准确率："
        f"{result['best_validation_accuracy'] * 100:.2f}%"
    )
    print(f"对应验证 loss：{result['best_validation_loss']:.4f}")
    print("官方测试集保持未使用，留到网格搜索确定最终配置之后。")
    print("=" * 70)


if __name__ == "__main__":
    main()
