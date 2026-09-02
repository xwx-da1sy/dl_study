"""加载最佳 Swin，在 ImageNet-1K 官方验证集上评估并生成可视化结果。"""

import argparse
from pathlib import Path

import torch
from torch import nn

from swin import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    NUM_CLASSES,
    NUM_WORKERS,
    RESULTS_DIR,
    TRAINING_HISTORY_PATH,
    CustomSwin,
    collect_evaluation_results,
    create_imagenet1k_dataloaders,
    load_training_history,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_prediction_samples,
    plot_top_misclassifications,
    plot_training_history,
    save_evaluation_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="评估并可视化 ImageNet-1K 四阶段自定义 Swin"
    )
    parser.add_argument("--checkpoint", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--history", type=Path, default=TRAINING_HISTORY_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"没有找到模型检查点：{args.checkpoint}\n"
            "请先完成训练和网格搜索，再评估最终选中的模型。"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    _, _, official_validation_loader, class_names = create_imagenet1k_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = CustomSwin().to(device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # 最终评估使用普通交叉熵，Label Smoothing 只用于训练正则化。
    criterion = nn.CrossEntropyLoss()
    results = collect_evaluation_results(
        model=model,
        data_loader=official_validation_loader,
        criterion=criterion,
        device=device,
        num_classes=NUM_CLASSES,
        max_visualization_samples=args.samples,
        use_amp=not args.no_amp,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "evaluation_summary.json"
    confusion_matrix_path = args.output_dir / "confusion_matrix.png"
    per_class_path = args.output_dir / "per_class_accuracy.png"
    top_misclassifications_path = (
        args.output_dir / "top_misclassifications.png"
    )
    sample_predictions_path = args.output_dir / "sample_predictions.png"
    training_curves_path = args.output_dir / "training_curves.png"

    summary = save_evaluation_summary(results, class_names, summary_path)
    plot_confusion_matrix(
        results["confusion_matrix"],
        class_names,
        confusion_matrix_path,
    )
    plot_per_class_accuracy(
        results["per_class_accuracy"],
        class_names,
        results["top1_accuracy"],
        per_class_path,
    )
    plot_top_misclassifications(
        results["confusion_matrix"],
        class_names,
        top_misclassifications_path,
    )
    plot_prediction_samples(
        results,
        class_names,
        sample_predictions_path,
    )

    if args.history.exists():
        history = load_training_history(args.history)
        plot_training_history(history, training_curves_path)
    else:
        training_curves_path = None
        print(f"未找到训练历史，跳过训练曲线：{args.history}")

    print("=" * 70)
    print(f"模型检查点 epoch：{checkpoint.get('epoch', '未知')}")
    print(f"官方验证集样本数：{summary['total_samples']}")
    print(f"官方验证集 loss：{summary['official_validation_loss']:.4f}")
    print(
        "官方验证集 Top-1 / Top-5："
        f"{summary['official_validation_top1_accuracy'] * 100:.2f}% / "
        f"{summary['official_validation_top5_accuracy'] * 100:.2f}%"
    )
    print(f"指标 JSON：{summary_path}")
    print(f"混淆矩阵：{confusion_matrix_path}")
    print(f"逐类别准确率：{per_class_path}")
    print(f"Top 误分类对：{top_misclassifications_path}")
    print(f"样本预测图：{sample_predictions_path}")
    if training_curves_path is not None:
        print(f"训练曲线：{training_curves_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
