"""使用训练好的 Tiny ViT 检查点做图片推理 / 预测。

支持两种用法：
1. 对本地任意图片文件预测（自动缩放并套用 CIFAR-10 归一化）。
2. 从 CIFAR-10 测试集中抽样预测，并打印整体准确率。

模型权重来自 vit/ 包中的 TinyViT 定义，检查点默认读取
checkpoints/tiny_vit_best.pt（只保存了验证集准确率最高的那一份）。
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# 直接运行 python infer.py 时使用同级顶层导入；
# 被其他模块以包形式导入时退回相对导入。
try:
    from vit import BEST_MODEL_PATH, TinyViT
except ImportError:
    from .vit import BEST_MODEL_PATH, TinyViT


# CIFAR-10 的 10 个类别名称，顺序与 torchvision 保持一致。
# 若模型换用其他数据集，可通过 --class-names 覆盖。
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# 与 data.py 中完全相同的均值与标准差，保证预处理一致。
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

IMAGE_SIZE = 32
TOP_K = 3


def build_model(checkpoint_path=BEST_MODEL_PATH, device=None):
    """
    作用：加载训练好的 Tiny ViT 权重，返回处于评估模式的模型。

    参数：
        checkpoint_path：检查点文件路径；文件内应至少包含 model_state_dict。
        device：目标设备；为 None 时自动选择 CUDA 或 CPU。

    返回值：
        model：已加载权重并切换到 eval() 的 TinyViT。
        device：实际使用的设备。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"检查点 {checkpoint_path} 中未找到 model_state_dict，"
            "请确认它是本工程 train_tiny_vit.py 生成的 best 模型"
        )

    model = TinyViT()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


def load_image_tensor(image_path, device=None):
    """
    作用：把本地图片文件整理成 Tiny ViT 需要的单张输入张量。

    参数：
        image_path：图片文件路径（jpg/png 等常见格式）。
        device：目标设备；为 None 时自动选择 CUDA 或 CPU。

    返回值：
        image_tensor：shape 为 1 x 3 x 32 x 32 的输入张量。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            # 模型在 32 x 32 上训练，先把任意尺寸图片缩放成统一大小。
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            # 必须与训练时一致，否则输入分布偏移会导致预测失准。
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    # unsqueeze(0)：把 3 x 32 x 32 升为 1 x 3 x 32 x 32，凑出 batch 维。
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor


@torch.inference_mode()
def predict_image(model, image_path, class_names=CIFAR10_CLASSES, top_k=TOP_K, device=None):
    """
    作用：对单张图片做预测，返回最可能的 top_k 个类别及对应概率。

    参数：
        model：处于 eval 模式的 TinyViT。
        image_path：本地图片路径。
        class_names：类别名称元组，长度需与模型输出维度一致。
        top_k：返回前 k 个最可能的类别。
        device：目标设备。

    返回值：
        results：长度为 top_k 的列表，每项是一个 (类别名, 概率) 元组。
    """
    image_tensor = load_image_tensor(image_path, device=device)
    logits = model(image_tensor)
    # 对单张图片的 logits 做 Softmax，得到 10 个类别的概率分布。
    probabilities = torch.softmax(logits, dim=1)[0]
    topk = torch.topk(probabilities, k=min(top_k, len(class_names)))
    results = [
        (class_names[index.item()], probability.item())
        for probability, index in zip(topk.values, topk.indices)
    ]
    return results


def predict_test_samples(model, num_samples, class_names=CIFAR10_CLASSES, device=None):
    """
    作用：从 CIFAR-10 测试集中抽取若干样本预测，并打印整体准确率。

    参数：
        model：处于 eval 模式的 TinyViT。
        num_samples：需要预测的样本数量。
        class_names：类别名称元组。
        device：目标设备。

    返回值：
        accuracy：抽样集合上的分类准确率，范围 0～1。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 复用 vit 包里已经写好的测试集 DataLoader，避免重复数据加载逻辑。
    try:
        from vit import create_cifar10_dataloaders
    except ImportError:
        from .vit import create_cifar10_dataloaders

    _, _, test_loader, _ = create_cifar10_dataloaders()

    model.eval()
    total = 0
    correct = 0
    shown = 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.inference_mode():
            predictions = model(images).argmax(dim=1)
        total += labels.shape[0]
        correct += (predictions == labels).sum().item()

        # 只打印前一小批样本的预测明细，避免刷屏。
        if shown < num_samples:
            for image, label, prediction in zip(images, labels, predictions):
                if shown >= num_samples:
                    break
                shown += 1
                mark = "OK" if prediction == label else "XX"
                print(
                    f"[{mark}] 真实：{class_names[label.item()]:<11} "
                    f"预测：{class_names[prediction.item()]}"
                )

        if total >= num_samples:
            break

    accuracy = correct / total
    print("-" * 50)
    print(f"抽样 {total} 张测试图片，准确率：{accuracy * 100:.2f}%")
    return accuracy


def parse_args():
    """解析命令行参数，支持单图预测和测试集抽样两种模式。"""
    parser = argparse.ArgumentParser(description="使用 Tiny ViT 检查点做推理")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BEST_MODEL_PATH,
        help="训练好的模型检查点路径",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="对单张本地图片做预测（jpg/png 等）",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=0,
        help="从 CIFAR-10 测试集抽取的样本数量，0 表示不抽样",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="单图预测时返回前 k 个最可能的类别",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.image is None and args.test_samples <= 0:
        print("请指定 --image 单图路径，或 --test-samples N 做测试集抽样。")
        return

    model, device = build_model(checkpoint_path=args.checkpoint, device=None)
    print(f"已加载检查点：{args.checkpoint}")
    print(f"计算设备：{device}")

    if args.image is not None:
        results = predict_image(
            model=model,
            image_path=args.image,
            top_k=args.top_k,
            device=device,
        )
        print(f"图片：{args.image}")
        for rank, (name, prob) in enumerate(results, start=1):
            print(f"  Top {rank}: {name:<11} {prob * 100:.2f}%")

    if args.test_samples > 0:
        predict_test_samples(
            model=model,
            num_samples=args.test_samples,
            device=device,
        )


if __name__ == "__main__":
    main()
