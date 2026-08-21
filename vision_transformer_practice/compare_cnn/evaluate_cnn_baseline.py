"""评估 CNN 公平对照模型并生成与 TinyViT 同规格的结果。"""

import argparse
from pathlib import Path

import torch
from torch import nn

try:
    from vision_transformer_practice.cnn_baseline import (
        CNNBaseline,
        CNN_CHECKPOINT_PATH,
        CNN_CONFUSION_MATRIX_PATH,
        CNN_EVALUATION_SUMMARY_PATH,
        CNN_HISTORY_PATH,
        CNN_SAMPLE_PREDICTIONS_PATH,
        CNN_TRAINING_CURVES_PATH,
    )
    from vision_transformer_practice.vit import (
        BATCH_SIZE,
        LABEL_SMOOTHING,
        NUM_CLASSES,
        NUM_WORKERS,
        collect_evaluation_results,
        create_cifar10_dataloaders,
        load_training_history,
        plot_confusion_matrix,
        plot_prediction_samples,
        plot_training_history,
        save_evaluation_summary,
    )
except ImportError:
    from cnn_baseline import (
        CNNBaseline,
        CNN_CHECKPOINT_PATH,
        CNN_CONFUSION_MATRIX_PATH,
        CNN_EVALUATION_SUMMARY_PATH,
        CNN_HISTORY_PATH,
        CNN_SAMPLE_PREDICTIONS_PATH,
        CNN_TRAINING_CURVES_PATH,
    )
    from vit import (
        BATCH_SIZE,
        LABEL_SMOOTHING,
        NUM_CLASSES,
        NUM_WORKERS,
        collect_evaluation_results,
        create_cifar10_dataloaders,
        load_training_history,
        plot_confusion_matrix,
        plot_prediction_samples,
        plot_training_history,
        save_evaluation_summary,
    )


def parse_args():
    """读取 checkpoint、训练历史和 DataLoader 参数。"""
    parser = argparse.ArgumentParser(description="评估 CIFAR-10 CNN 公平对照模型")
    parser.add_argument("--checkpoint", type=Path, default=CNN_CHECKPOINT_PATH)
    parser.add_argument("--history", type=Path, default=CNN_HISTORY_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    """加载最佳 CNN，计算测试指标并生成曲线、混淆矩阵和预测图。"""
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"没有找到 CNN checkpoint：{args.checkpoint}\n"
            "请先运行 train_cnn_baseline.py。"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, class_names = create_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = CNNBaseline().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    results = collect_evaluation_results(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=NUM_CLASSES,
        max_visualization_samples=args.samples,
        use_amp=not args.no_amp,
    )
    summary = save_evaluation_summary(
        results, class_names, CNN_EVALUATION_SUMMARY_PATH
    )
    plot_confusion_matrix(
        results["confusion_matrix"], class_names, CNN_CONFUSION_MATRIX_PATH
    )
    plot_prediction_samples(
        results,
        class_names,
        CNN_SAMPLE_PREDICTIONS_PATH,
        model_name="CNN Baseline",
    )
    if args.history.exists():
        history = load_training_history(args.history)
        plot_training_history(history, CNN_TRAINING_CURVES_PATH)

    print("=" * 70)
    print(f"checkpoint epoch：{checkpoint.get('epoch', '未知')}")
    print(f"测试集样本数：{summary['total_samples']}")
    print(f"测试集 loss：{summary['test_loss']:.4f}")
    print(f"测试集 accuracy：{summary['test_accuracy'] * 100:.2f}%")
    print(f"评估结果：{CNN_EVALUATION_SUMMARY_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
