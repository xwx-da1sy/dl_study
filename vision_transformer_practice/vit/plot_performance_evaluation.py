"""生成 TinyViT 性能评估可视化图。

加载最佳模型，在测试集上收集完整评估数据，生成三张图：
1. per_class_accuracy.png   每类准确率柱状图（按高到低）
2. top_misclassifications.png  Top 误分类对横向柱状图
3. misclassified_samples.png   误分类样本特写

运行：python plot_performance_evaluation.py
"""

from pathlib import Path

import matplotlib

# 无窗口后端，保证从终端运行也能保存 PNG。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

try:
    from vision_transformer_practice.vit import (
        BATCH_SIZE,
        BEST_MODEL_PATH,
        NUM_CLASSES,
        NUM_WORKERS,
        RESULTS_DIR,
        TinyViT,
        create_cifar10_dataloaders,
    )
except ImportError:
    from vit import (
        BATCH_SIZE,
        BEST_MODEL_PATH,
        NUM_CLASSES,
        NUM_WORKERS,
        RESULTS_DIR,
        TinyViT,
        create_cifar10_dataloaders,
    )

# CIFAR-10 标准化参数，用于把图片还原到可显示范围。
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# 中文字体，让图里的中文标注正常显示。
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


@torch.inference_mode()
def collect_performance_data(model, data_loader, device, num_classes, max_misclassified=16):
    """
    作用：遍历测试集，一次性收集混淆矩阵和误分类样本。

    参数：
        model：已加载最佳权重的 TinyViT。
        data_loader：测试集 DataLoader。
        device：CPU 或 CUDA 设备。
        num_classes：类别数量，CIFAR-10 为 10。
        max_misclassified：最多保留多少张误分类样本用于特写图。

    返回值：包含混淆矩阵、每类准确率和误分类样本的字典。
    """
    model.eval()
    all_labels = []
    all_preds = []
    mis_images = []
    mis_labels = []
    mis_preds = []
    mis_confs = []

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        probabilities = torch.softmax(logits.float(), dim=1)
        confidences, predictions = probabilities.max(dim=1)

        all_labels.append(labels.cpu())
        all_preds.append(predictions.cpu())

        # 只收集误分类样本，用于第三张特写图。
        wrong_mask = predictions != labels
        for index in range(labels.shape[0]):
            if wrong_mask[index] and len(mis_images) < max_misclassified:
                mis_images.append(images[index].cpu())
                mis_labels.append(int(labels[index].item()))
                mis_preds.append(int(predictions[index].item()))
                mis_confs.append(float(confidences[index].item()))

    labels_tensor = torch.cat(all_labels)
    preds_tensor = torch.cat(all_preds)

    # 用一维索引统计 10x10 混淆矩阵。
    confusion_indices = labels_tensor * num_classes + preds_tensor
    confusion_matrix = torch.bincount(
        confusion_indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    # 每类准确率 = 对角线 / 行总和（召回率）。
    class_totals = confusion_matrix.sum(dim=1).clamp_min(1)
    per_class_accuracy = confusion_matrix.diag().float() / class_totals.float()

    return {
        "confusion_matrix": confusion_matrix,
        "per_class_accuracy": per_class_accuracy,
        "total_accuracy": (preds_tensor == labels_tensor).float().mean().item(),
        "mis_images": torch.stack(mis_images) if mis_images else torch.empty(0),
        "mis_labels": mis_labels,
        "mis_preds": mis_preds,
        "mis_confs": mis_confs,
    }


def plot_per_class_accuracy(per_class_accuracy, class_names, total_accuracy, output_path):
    """
    作用：画每类准确率柱状图，按高到低排序，低于平均值的标红。

    参数：
        per_class_accuracy：10 个类的准确率张量。
        class_names：类别名称列表。
        total_accuracy：整体测试准确率，画成参考虚线。
        output_path：PNG 保存路径。
    """
    pairs = sorted(
        zip(class_names, per_class_accuracy.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    names = [p[0] for p in pairs]
    accs = [p[1] * 100 for p in pairs]
    threshold = total_accuracy * 100

    figure, axis = plt.subplots(figsize=(11, 5))
    # 低于平均准确率的类标红，高于的标蓝。
    colors = ["#D85A30" if acc < threshold else "#378ADD" for acc in accs]
    bars = axis.bar(names, accs, color=colors, edgecolor="#2C2C2A", linewidth=0.5)

    # 平均准确率参考线。
    axis.axhline(
        threshold,
        color="#5F5E5A",
        linestyle="--",
        linewidth=1,
        label="平均 test_acc %.1f%%" % threshold,
    )

    # 每根柱子顶部标数值。
    for bar, acc in zip(bars, accs):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            acc + 1.5,
            "%.1f%%" % acc,
            ha="center",
            fontsize=10,
        )

    axis.set_ylabel("准确率 (%)")
    axis.set_title("每类准确率（按高到低）", fontsize=14)
    axis.set_ylim(0, 100)
    axis.legend(loc="lower right")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_top_misclassifications(confusion_matrix, class_names, output_path, top_n=10):
    """
    作用：从混淆矩阵找 Top-N 误分类对，画横向柱状图。

    参数：
        confusion_matrix：行=真实类、列=预测类的计数矩阵。
        class_names：类别名称列表。
        output_path：PNG 保存路径。
        top_n：显示前多少个误分类对。
    """
    cm = confusion_matrix.float()
    row_totals = cm.sum(dim=1, keepdim=True).clamp_min(1)
    # 行归一化：每个误分对占该真实类的比例。
    norm_cm = cm / row_totals

    # 收集所有非对角线项。
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                pairs.append(
                    (
                        class_names[i],
                        class_names[j],
                        norm_cm[i, j].item() * 100,
                        cm[i, j].item(),
                    )
                )
    pairs.sort(key=lambda x: x[2], reverse=True)
    top = pairs[:top_n]

    labels = ["%s -> %s" % (p[0], p[1]) for p in top]
    values = [p[2] for p in top]
    counts = [p[3] for p in top]

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(
        range(len(top)),
        values,
        color="#D85A30",
        edgecolor="#2C2C2A",
        linewidth=0.5,
    )
    axis.set_yticks(range(len(top)))
    axis.set_yticklabels(labels, fontsize=11)
    axis.invert_yaxis()
    axis.set_xlabel("误分率（占该真实类 %）")
    axis.set_title("Top %d 误分类对" % top_n, fontsize=14)

    for index, (value, count) in enumerate(zip(values, counts)):
        axis.text(
            value + 0.3,
            index,
            "%.1f%% (%d张)" % (value, count),
            va="center",
            fontsize=9,
        )

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_misclassified_samples(results, class_names, output_path):
    """
    作用：画误分类样本特写，每格显示图片、真实类、预测类和置信度。

    参数：
        results：collect_performance_data 返回的字典。
        class_names：类别名称列表。
        output_path：PNG 保存路径。
    """
    images = results["mis_images"]
    mis_labels = results["mis_labels"]
    mis_preds = results["mis_preds"]
    mis_confs = results["mis_confs"]

    if images.shape[0] == 0:
        print("没有误分类样本，跳过特写图。")
        return None

    # 把标准化后的图片还原到可显示范围。
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    display_images = (images * std + mean).clamp(0, 1)

    sample_count = images.shape[0]
    columns = 4
    rows = (sample_count + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.5 * columns, 3.5 * rows),
        squeeze=False,
    )

    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= sample_count:
            continue

        true_label = mis_labels[index]
        predicted_label = mis_preds[index]
        confidence = mis_confs[index] * 100

        axis.imshow(display_images[index].permute(1, 2, 0).numpy())
        axis.set_title(
            "True: %s\nPred: %s (%.1f%%)" % (class_names[true_label], class_names[predicted_label], confidence),
            color="#D85A30",
            fontsize=10,
        )

    figure.suptitle("误分类样本特写", fontsize=14)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main():
    """
    作用：加载最佳模型，收集测试集评估数据，生成三张性能图。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备：%s" % device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    _, _, test_loader, class_names = create_cifar10_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = TinyViT().to(device)
    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print("已加载检查点 epoch：%s" % checkpoint.get("epoch", "未知"))

    print("开始遍历测试集收集评估数据...")
    results = collect_performance_data(
        model=model,
        data_loader=test_loader,
        device=device,
        num_classes=NUM_CLASSES,
        max_misclassified=16,
    )
    print("测试准确率：%.4f" % results["total_accuracy"])

    per_class_path = RESULTS_DIR / "per_class_accuracy.png"
    top_mis_path = RESULTS_DIR / "top_misclassifications.png"
    mis_samples_path = RESULTS_DIR / "misclassified_samples.png"

    plot_per_class_accuracy(
        per_class_accuracy=results["per_class_accuracy"],
        class_names=class_names,
        total_accuracy=results["total_accuracy"],
        output_path=per_class_path,
    )
    plot_top_misclassifications(
        confusion_matrix=results["confusion_matrix"],
        class_names=class_names,
        output_path=top_mis_path,
    )
    plot_misclassified_samples(
        results=results,
        class_names=class_names,
        output_path=mis_samples_path,
    )

    print("=" * 70)
    print("每类准确率柱状图：%s" % per_class_path)
    print("Top 误分类对：%s" % top_mis_path)
    print("误分类样本特写：%s" % mis_samples_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
