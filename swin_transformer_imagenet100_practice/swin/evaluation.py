"""自定义 Swin 的评估指标统计与结果可视化。"""

import json
import math
from pathlib import Path

import matplotlib

# 使用无窗口后端，终端运行时直接把图像保存为 PNG。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .data import IMAGENET_MEAN, IMAGENET_STD


@torch.inference_mode()
def collect_evaluation_results(
    model,
    data_loader,
    criterion,
    device,
    num_classes,
    max_visualization_samples=16,
    use_amp=True,
):
    """遍历评估集，收集总体指标、混淆矩阵和展示样本。"""
    if num_classes <= 0:
        raise ValueError("num_classes 必须大于0")
    if max_visualization_samples <= 0:
        raise ValueError("max_visualization_samples 必须大于0")

    model.eval()
    amp_enabled = use_amp and device.type == "cuda"
    total_loss = 0.0
    total_samples = 0
    top1_correct = 0
    top5_correct = 0
    all_labels = []
    all_predictions = []
    sample_images = []
    sample_labels = []
    sample_predictions = []
    sample_confidences = []
    collected_samples = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        probabilities = torch.softmax(logits.float(), dim=1)
        confidences, predictions = probabilities.max(dim=1)
        top5_predictions = logits.topk(
            k=min(5, num_classes),
            dim=1,
        ).indices

        batch_size = labels.shape[0]
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        top1_correct += (predictions == labels).sum().item()
        top5_correct += (
            top5_predictions == labels.unsqueeze(1)
        ).any(dim=1).sum().item()
        all_labels.append(labels.cpu())
        all_predictions.append(predictions.cpu())

        remaining = max_visualization_samples - collected_samples
        if remaining > 0:
            take_count = min(remaining, batch_size)
            sample_images.append(images[:take_count].cpu())
            sample_labels.append(labels[:take_count].cpu())
            sample_predictions.append(predictions[:take_count].cpu())
            sample_confidences.append(confidences[:take_count].cpu())
            collected_samples += take_count

    if total_samples == 0:
        raise RuntimeError("评估集为空，无法收集评估结果")

    labels = torch.cat(all_labels)
    predictions = torch.cat(all_predictions)
    confusion_indices = labels * num_classes + predictions
    confusion_matrix = torch.bincount(
        confusion_indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    class_totals = confusion_matrix.sum(dim=1)
    per_class_accuracy = torch.where(
        class_totals > 0,
        confusion_matrix.diag().float() / class_totals.float(),
        torch.zeros_like(class_totals, dtype=torch.float32),
    )

    return {
        "loss": total_loss / total_samples,
        "top1_accuracy": top1_correct / total_samples,
        "top5_accuracy": top5_correct / total_samples,
        "total_samples": total_samples,
        "confusion_matrix": confusion_matrix,
        "per_class_accuracy": per_class_accuracy,
        "sample_images": torch.cat(sample_images),
        "sample_labels": torch.cat(sample_labels),
        "sample_predictions": torch.cat(sample_predictions),
        "sample_confidences": torch.cat(sample_confidences),
    }


def load_training_history(history_path):
    """读取 fit 每个 epoch 保存的训练历史。"""
    history_path = Path(history_path)
    with history_path.open("r", encoding="utf-8") as history_file:
        history = json.load(history_file)
    if not isinstance(history, list) or not history:
        raise ValueError("训练历史必须是非空列表")
    return history


def save_evaluation_summary(results, class_names, output_path):
    """把官方验证指标和逐类别准确率保存为 JSON。"""
    if len(class_names) != len(results["per_class_accuracy"]):
        raise ValueError("class_names 数量必须与类别准确率数量一致")

    summary = {
        "official_validation_loss": results["loss"],
        "official_validation_top1_accuracy": results["top1_accuracy"],
        "official_validation_top5_accuracy": results["top5_accuracy"],
        "total_samples": results["total_samples"],
        "per_class_accuracy": {
            class_name: accuracy.item()
            for class_name, accuracy in zip(
                class_names,
                results["per_class_accuracy"],
            )
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
    return summary


def plot_training_history(history, output_path):
    """绘制训练/验证损失、准确率和学习率曲线。"""
    epochs = [record["epoch"] for record in history]
    train_loss = [record["train_loss"] for record in history]
    validation_loss = [record["validation_loss"] for record in history]
    train_accuracy = [
        record["train_weighted_accuracy"] * 100 for record in history
    ]
    validation_accuracy = [
        record["validation_accuracy"] * 100 for record in history
    ]
    learning_rates = [record["learning_rate"] for record in history]

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, train_loss, label="Train", linewidth=2)
    axes[0].plot(epochs, validation_loss, label="Validation", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy")
    axes[0].legend()

    axes[1].plot(
        epochs,
        train_accuracy,
        label="Train (mixed labels)",
        linewidth=2,
    )
    axes[1].plot(
        epochs,
        validation_accuracy,
        label="Validation",
        linewidth=2,
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    axes[2].plot(epochs, learning_rates, color="tab:purple", linewidth=2)
    axes[2].set_title("Warmup + Cosine Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")

    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    return _save_figure(figure, output_path)


def plot_confusion_matrix(confusion_matrix, class_names, output_path):
    """绘制归一化混淆矩阵，不在每个格子中堆叠数字。"""
    if confusion_matrix.shape != (len(class_names), len(class_names)):
        raise ValueError("混淆矩阵尺寸必须与 class_names 数量一致")

    confusion_matrix = confusion_matrix.float()
    row_totals = confusion_matrix.sum(dim=1, keepdim=True).clamp_min(1)
    normalized_matrix = 100.0 * confusion_matrix / row_totals

    figure, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(
        normalized_matrix.numpy(),
        cmap="Blues",
        vmin=0,
        vmax=100,
        interpolation="nearest",
    )
    figure.colorbar(image, ax=axis, label="Percentage (%)")
    axis.set_title("ImageNet-100 Normalized Confusion Matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")

    # 100个完整类名仍会重叠，因此只等间隔显示少量坐标标签。
    tick_step = max(1, len(class_names) // 10)
    tick_positions = list(range(0, len(class_names), tick_step))
    tick_labels = [class_names[index] for index in tick_positions]
    axis.set_xticks(tick_positions, labels=tick_labels, rotation=45, ha="right")
    axis.set_yticks(tick_positions, labels=tick_labels)
    axis.tick_params(labelsize=7)
    figure.tight_layout()
    return _save_figure(figure, output_path)


def plot_per_class_accuracy(
    per_class_accuracy,
    class_names,
    total_accuracy,
    output_path,
):
    """按准确率从高到低绘制各类别的横向柱状图。"""
    if len(per_class_accuracy) != len(class_names):
        raise ValueError("类别准确率数量必须与 class_names 一致")

    pairs = sorted(
        zip(class_names, per_class_accuracy.tolist()),
        key=lambda pair: pair[1],
    )
    names = [pair[0] for pair in pairs]
    accuracies = [pair[1] * 100 for pair in pairs]
    total_percentage = total_accuracy * 100
    colors = [
        "#D85A30" if accuracy < total_percentage else "#378ADD"
        for accuracy in accuracies
    ]

    figure, axis = plt.subplots(figsize=(11, 22))
    axis.barh(names, accuracies, color=colors, height=0.75)
    axis.axvline(
        total_percentage,
        color="#5F5E5A",
        linestyle="--",
        linewidth=1.2,
        label=f"Overall Top-1: {total_percentage:.1f}%",
    )
    axis.set_xlim(0, 100)
    axis.set_xlabel("Accuracy (%)")
    axis.set_title("Per-Class Accuracy")
    axis.tick_params(axis="y", labelsize=6)
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    return _save_figure(figure, output_path)


def plot_top_misclassifications(
    confusion_matrix,
    class_names,
    output_path,
    top_n=15,
):
    """绘制比例最高的真实类别→错误预测类别组合。"""
    if top_n <= 0:
        raise ValueError("top_n 必须大于0")

    confusion_matrix = confusion_matrix.float()
    row_totals = confusion_matrix.sum(dim=1, keepdim=True).clamp_min(1)
    normalized_matrix = confusion_matrix / row_totals
    pairs = []
    for true_index in range(len(class_names)):
        for predicted_index in range(len(class_names)):
            if true_index == predicted_index:
                continue
            pairs.append(
                (
                    class_names[true_index],
                    class_names[predicted_index],
                    normalized_matrix[true_index, predicted_index].item() * 100,
                    int(confusion_matrix[true_index, predicted_index].item()),
                )
            )
    pairs.sort(key=lambda pair: pair[2], reverse=True)
    top_pairs = pairs[:top_n]

    labels = [f"{pair[0]} -> {pair[1]}" for pair in top_pairs]
    percentages = [pair[2] for pair in top_pairs]
    counts = [pair[3] for pair in top_pairs]
    figure, axis = plt.subplots(figsize=(11, 7))
    axis.barh(range(len(top_pairs)), percentages, color="#D85A30")
    axis.set_yticks(range(len(top_pairs)), labels=labels)
    axis.invert_yaxis()
    axis.set_xlabel("Misclassification rate within true class (%)")
    axis.set_title(f"Top {len(top_pairs)} Misclassification Pairs")
    axis.grid(axis="x", alpha=0.25)
    for index, (percentage, count) in enumerate(zip(percentages, counts)):
        axis.text(
            percentage + 0.1,
            index,
            f"{percentage:.1f}% ({count})",
            va="center",
            fontsize=8,
        )
    figure.tight_layout()
    return _save_figure(figure, output_path)


def plot_prediction_samples(
    results,
    class_names,
    output_path,
    model_name="Custom Swin",
):
    """绘制测试图片、真实类别、预测类别和置信度。"""
    images = results["sample_images"]
    labels = results["sample_labels"]
    predictions = results["sample_predictions"]
    confidences = results["sample_confidences"]
    if images.shape[0] == 0:
        raise ValueError("没有可用于绘图的预测样本")

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    display_images = (images * std + mean).clamp(0, 1)
    sample_count = images.shape[0]
    columns = min(4, sample_count)
    rows = math.ceil(sample_count / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns, 3.2 * rows),
        squeeze=False,
    )

    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= sample_count:
            continue
        true_label = int(labels[index].item())
        predicted_label = int(predictions[index].item())
        confidence = confidences[index].item() * 100
        title_color = "green" if true_label == predicted_label else "red"
        axis.imshow(display_images[index].permute(1, 2, 0).numpy())
        axis.set_title(
            f"True: {class_names[true_label]}\n"
            f"Pred: {class_names[predicted_label]} ({confidence:.1f}%)",
            color=title_color,
            fontsize=8,
        )

    figure.suptitle(f"{model_name} Predictions", fontsize=14)
    figure.tight_layout()
    return _save_figure(figure, output_path)


def _save_figure(figure, output_path):
    """统一创建输出目录、保存图片并释放 Matplotlib 内存。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path
