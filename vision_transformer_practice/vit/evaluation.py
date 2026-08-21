"""Tiny ViT 的测试指标统计与训练结果可视化。"""

import json
import math
from pathlib import Path

import matplotlib

# 使用无窗口后端，保证从终端运行时也能把图像保存为 PNG。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


# CIFAR-10 标准化所使用的均值和标准差，用于把图片还原到可显示范围。
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


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
    """
    作用：
        在整个测试集上收集 loss、预测类别、混淆矩阵、分类准确率和展示样本。

    参数：
        model：已经加载最佳权重的 TinyViT。
        data_loader：测试集 DataLoader。
        criterion：CrossEntropyLoss。
        device：CPU 或 CUDA 设备。
        num_classes：类别数量，CIFAR-10 为 10。
        max_visualization_samples：最多保留多少张图片用于预测结果图。
        use_amp：CUDA 上是否使用自动混合精度。

    返回值：
        results：包含总体指标、逐类别指标、混淆矩阵和展示样本的字典。
    """
    if num_classes <= 0:
        raise ValueError("num_classes 必须大于 0")
    if max_visualization_samples <= 0:
        raise ValueError("max_visualization_samples 必须大于 0")

    model.eval()
    amp_enabled = use_amp and device.type == "cuda"
    total_loss = 0.0
    total_samples = 0
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

        # Softmax 仅用于评估时计算置信度；训练时仍直接把 logits 交给交叉熵。
        probabilities = torch.softmax(logits.float(), dim=1)
        confidences, predictions = probabilities.max(dim=1)

        batch_size = labels.shape[0]
        total_samples += batch_size
        total_loss += loss.item() * batch_size
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
        raise RuntimeError("测试集为空，无法收集评估结果")

    labels = torch.cat(all_labels)
    predictions = torch.cat(all_predictions)

    # 把 (真实类别, 预测类别) 映射到一维索引，再一次性统计 10 x 10 混淆矩阵。
    confusion_indices = labels * num_classes + predictions
    confusion_matrix = torch.bincount(
        confusion_indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    class_totals = confusion_matrix.sum(dim=1)
    class_correct = confusion_matrix.diag()
    per_class_accuracy = torch.where(
        class_totals > 0,
        class_correct.float() / class_totals.float(),
        torch.zeros_like(class_totals, dtype=torch.float32),
    )

    results = {
        "loss": total_loss / total_samples,
        "accuracy": (predictions == labels).float().mean().item(),
        "total_samples": total_samples,
        "confusion_matrix": confusion_matrix,
        "per_class_accuracy": per_class_accuracy,
        "sample_images": torch.cat(sample_images) if sample_images else torch.empty(0),
        "sample_labels": torch.cat(sample_labels) if sample_labels else torch.empty(0),
        "sample_predictions": (
            torch.cat(sample_predictions) if sample_predictions else torch.empty(0)
        ),
        "sample_confidences": (
            torch.cat(sample_confidences) if sample_confidences else torch.empty(0)
        ),
    }
    return results


def load_training_history(history_path):
    """
    作用：读取训练期间逐 epoch 保存的 JSON 指标。

    参数：history_path：training_history.json 文件路径。

    返回值：history，由多个 epoch 指标字典组成的列表。
    """
    history_path = Path(history_path)
    with history_path.open("r", encoding="utf-8") as history_file:
        history = json.load(history_file)

    if not isinstance(history, list) or not history:
        raise ValueError("训练历史必须是非空列表")
    return history


def save_evaluation_summary(results, class_names, output_path):
    """
    作用：把总体测试指标和逐类别准确率保存为便于阅读的 JSON。

    参数：
        results：collect_evaluation_results 返回的结果。
        class_names：类别名称列表。
        output_path：JSON 保存路径。

    返回值：summary，已经转换为普通 Python 数值的指标字典。
    """
    if len(class_names) != len(results["per_class_accuracy"]):
        raise ValueError("class_names 数量必须与逐类别准确率数量一致")

    summary = {
        "test_loss": results["loss"],
        "test_accuracy": results["accuracy"],
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
    """
    作用：绘制训练/验证 loss、accuracy 和余弦退火学习率曲线。

    参数：
        history：fit 保存的逐 epoch 指标列表。
        output_path：PNG 图片保存路径。

    返回值：output_path，实际保存的 Path 对象。
    """
    epochs = [record["epoch"] for record in history]
    train_loss = [record["train_loss"] for record in history]
    validation_loss = [record["validation_loss"] for record in history]
    train_accuracy = [record["train_accuracy"] * 100 for record in history]
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
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_accuracy, label="Train", linewidth=2)
    axes[1].plot(epochs, validation_accuracy, label="Validation", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, learning_rates, color="tab:purple", linewidth=2)
    axes[2].set_title("Cosine Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].grid(alpha=0.3)

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_confusion_matrix(confusion_matrix, class_names, output_path):
    """
    作用：绘制按真实类别归一化的测试集混淆矩阵。

    参数：
        confusion_matrix：行表示真实类别、列表示预测类别的计数矩阵。
        class_names：类别名称列表。
        output_path：PNG 图片保存路径。

    返回值：output_path，实际保存的 Path 对象。
    """
    confusion_matrix = confusion_matrix.float()
    row_totals = confusion_matrix.sum(dim=1, keepdim=True).clamp_min(1)
    normalized_matrix = 100.0 * confusion_matrix / row_totals

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(normalized_matrix.numpy(), cmap="Blues", vmin=0, vmax=100)
    figure.colorbar(image, ax=axis, label="Percentage (%)")

    axis.set_title("Normalized Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), labels=class_names)

    for row in range(len(class_names)):
        for column in range(len(class_names)):
            value = normalized_matrix[row, column].item()
            text_color = "white" if value >= 50 else "black"
            axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_prediction_samples(results, class_names, output_path, model_name="Tiny ViT"):
    """
    作用：绘制测试图片、真实类别、预测类别和预测置信度。

    参数：
        results：collect_evaluation_results 返回的结果。
        class_names：类别名称列表。
        output_path：PNG 图片保存路径。
        model_name：图像标题中显示的模型名称。

    返回值：output_path，实际保存的 Path 对象。
    """
    images = results["sample_images"]
    labels = results["sample_labels"]
    predictions = results["sample_predictions"]
    confidences = results["sample_confidences"]

    if images.shape[0] == 0:
        raise ValueError("没有可用于绘图的预测样本")

    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
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
            fontsize=9,
        )

    figure.suptitle(f"{model_name} Predictions", fontsize=14)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path
