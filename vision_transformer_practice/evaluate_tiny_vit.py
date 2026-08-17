"""加载最佳 Tiny ViT 权重，评估测试集并生成可视化结果。"""

import argparse
from pathlib import Path

import torch
from torch import nn

try:
    from .vit import (
        BATCH_SIZE,
        BEST_MODEL_PATH,
        NUM_CLASSES,
        NUM_WORKERS,
        RESULTS_DIR,
        TRAINING_HISTORY_PATH,
        TinyViT,
        collect_evaluation_results,
        create_cifar10_dataloaders,
        load_training_history,
        plot_confusion_matrix,
        plot_prediction_samples,
        plot_training_history,
        save_evaluation_summary,
    )
except ImportError:
    from vit import (
        BATCH_SIZE,
        BEST_MODEL_PATH,
        NUM_CLASSES,
        NUM_WORKERS,
        RESULTS_DIR,
        TRAINING_HISTORY_PATH,
        TinyViT,
        collect_evaluation_results,
        create_cifar10_dataloaders,
        load_training_history,
        plot_confusion_matrix,
        plot_prediction_samples,
        plot_training_history,
        save_evaluation_summary,
    )


def parse_args():
    """
    作用：读取模型检查点、训练历史和可视化输出目录等参数。

    参数：无；参数来自命令行。

    返回值：args，包含评估与可视化配置。
    """
    parser = argparse.ArgumentParser(description="评估并可视化 Tiny ViT")
    parser.add_argument("--checkpoint", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--history", type=Path, default=TRAINING_HISTORY_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    """
    作用：
        加载最佳模型，在测试集上计算总体与逐类别指标，
        并生成训练曲线、混淆矩阵和样本预测图。

    参数：无；评估配置由 parse_args 从命令行读取。

    返回值：无；指标打印到终端并保存到 results 目录。
    """
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"没有找到模型检查点：{args.checkpoint}\n"
            "请先运行 train_tiny_vit.py 完成训练。"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, class_names = create_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = TinyViT().to(device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    results = collect_evaluation_results(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=NUM_CLASSES,
        max_visualization_samples=args.samples,
        use_amp=not args.no_amp,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "evaluation_summary.json"
    confusion_matrix_path = args.output_dir / "confusion_matrix.png"
    sample_predictions_path = args.output_dir / "sample_predictions.png"
    training_curves_path = args.output_dir / "training_curves.png"

    summary = save_evaluation_summary(
        results=results,
        class_names=class_names,
        output_path=summary_path,
    )
    plot_confusion_matrix(
        confusion_matrix=results["confusion_matrix"],
        class_names=class_names,
        output_path=confusion_matrix_path,
    )
    plot_prediction_samples(
        results=results,
        class_names=class_names,
        output_path=sample_predictions_path,
    )

    if args.history.exists():
        history = load_training_history(args.history)
        plot_training_history(history, training_curves_path)
    else:
        training_curves_path = None
        print(f"未找到训练历史，跳过训练曲线：{args.history}")

    print("=" * 70)
    print(f"模型检查点 epoch：{checkpoint.get('epoch', '未知')}")
    print(f"测试集样本数：{summary['total_samples']}")
    print(f"测试集 loss：{summary['test_loss']:.4f}")
    print(f"测试集 accuracy：{summary['test_accuracy'] * 100:.2f}%")
    print("逐类别准确率：")
    for class_name, accuracy in summary["per_class_accuracy"].items():
        print(f"  {class_name:>10s}：{accuracy * 100:.2f}%")
    print("=" * 70)
    print(f"指标 JSON：{summary_path}")
    print(f"混淆矩阵：{confusion_matrix_path}")
    print(f"样本预测图：{sample_predictions_path}")
    if training_curves_path is not None:
        print(f"训练曲线：{training_curves_path}")


if __name__ == "__main__":
    main()
