"""CIFAR-100 数据集、数据增强、Mixup/CutMix 和 DataLoader。"""

import math

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import (
    BATCH_SIZE,
    CUTMIX_ALPHA,
    CUTMIX_PROBABILITY,
    DATA_ROOT,
    MIXUP_ALPHA,
    NUM_CLASSES,
    NUM_WORKERS,
    RANDOM_ERASING_PROBABILITY,
    RANDOM_SEED,
    VALIDATION_SIZE,
)


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def apply_mixup_or_cutmix(
    images,
    labels,
    mixup_alpha=MIXUP_ALPHA,
    cutmix_alpha=CUTMIX_ALPHA,
    cutmix_probability=CUTMIX_PROBABILITY,
):
    """随机使用 Mixup 或 CutMix，并返回计算混合损失需要的信息。"""
    if images.ndim != 4:
        raise ValueError("images 必须是 B x C x H x W 四维张量")
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError("labels 必须是一维张量，并且数量与 images 一致")
    if not images.is_floating_point():
        raise TypeError("Mixup/CutMix 要求 images 是浮点张量")
    if mixup_alpha < 0 or cutmix_alpha < 0:
        raise ValueError("mixup_alpha 和 cutmix_alpha 不能为负数")
    if not 0.0 <= cutmix_probability <= 1.0:
        raise ValueError("cutmix_probability 必须位于[0, 1]之间")

    use_mixup = mixup_alpha > 0
    use_cutmix = cutmix_alpha > 0
    if not use_mixup and not use_cutmix:
        return images, labels, labels, 1.0, "none"

    permutation = torch.randperm(images.shape[0], device=images.device)
    shuffled_images = images[permutation]
    shuffled_labels = labels[permutation]

    # 两种方法都开启时随机选择一种；只开启一种时直接使用该方法。
    choose_cutmix = use_cutmix and (
        not use_mixup or torch.rand(1).item() < cutmix_probability
    )

    if not choose_cutmix:
        mixing_lambda = torch.distributions.Beta(
            mixup_alpha,
            mixup_alpha,
        ).sample().item()
        mixed_images = (
            mixing_lambda * images
            + (1.0 - mixing_lambda) * shuffled_images
        )
        return (
            mixed_images,
            labels,
            shuffled_labels,
            mixing_lambda,
            "mixup",
        )

    mixing_lambda = torch.distributions.Beta(
        cutmix_alpha,
        cutmix_alpha,
    ).sample().item()
    _, _, height, width = images.shape

    # 被替换区域的宽高由 1-lambda 决定；中心点在图片内随机选择。
    cut_ratio = math.sqrt(1.0 - mixing_lambda)
    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)
    center_x = torch.randint(width, size=(1,)).item()
    center_y = torch.randint(height, size=(1,)).item()

    left = max(center_x - cut_width // 2, 0)
    right = min(center_x + (cut_width + 1) // 2, width)
    top = max(center_y - cut_height // 2, 0)
    bottom = min(center_y + (cut_height + 1) // 2, height)

    mixed_images = images.clone()
    mixed_images[:, :, top:bottom, left:right] = shuffled_images[
        :, :, top:bottom, left:right
    ]

    # 矩形可能被图片边界截断，所以用实际替换面积重新计算 lambda。
    replaced_area = (right - left) * (bottom - top)
    mixing_lambda = 1.0 - replaced_area / (height * width)
    return (
        mixed_images,
        labels,
        shuffled_labels,
        mixing_lambda,
        "cutmix",
    )


def calculate_mixed_loss(
    criterion,
    logits,
    labels_a,
    labels_b,
    mixing_lambda,
):
    """按照图片中两类内容的占比，加权计算两个标签的分类损失。"""
    return (
        mixing_lambda * criterion(logits, labels_a)
        + (1.0 - mixing_lambda) * criterion(logits, labels_b)
    )


def create_cifar100_dataloaders(
    data_root=DATA_ROOT,
    batch_size=BATCH_SIZE,
    validation_size=VALIDATION_SIZE,
    num_workers=NUM_WORKERS,
    random_seed=RANDOM_SEED,
):
    """
    下载或读取 CIFAR-100，并创建训练、验证和测试 DataLoader。

    返回：
        train_loader：训练集，每个 batch 的图片 shape 为 B x 3 x 32 x 32。
        validation_loader：从官方训练集中按类别均衡划出的验证集。
        test_loader：官方测试集。
        class_names：100个类别名称。
    """

    # 训练集使用随机增强；验证集和测试集只做确定性转换。
    # RandAugment 必须在 ToTensor 之前处理 PIL 图片；
    # RandomErasing 必须在 ToTensor 之后处理 Tensor。
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            transforms.RandomErasing(
                p=RANDOM_ERASING_PROBABILITY,
                scale=(0.02, 0.20),
                ratio=(0.3, 3.3),
                value=0,
            ),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )

    # 训练源和验证源读取相同的50,000张图片，但使用不同 transform。
    # 不能只创建一个 Dataset 后再 random_split，否则两部分会共享训练增强。
    train_source_dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        transform=train_transform,
        download=True,
    )
    validation_source_dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        transform=evaluation_transform,
        download=True,
    )
    test_dataset = datasets.CIFAR100(
        root=data_root,
        train=False,
        transform=evaluation_transform,
        download=True,
    )

    class_count = len(train_source_dataset.classes)
    if class_count != NUM_CLASSES:
        raise RuntimeError(f"期望{NUM_CLASSES}个类别，实际读取到{class_count}个")

    if validation_size <= 0 or validation_size >= len(train_source_dataset):
        raise ValueError(
            f"validation_size 必须位于1到{len(train_source_dataset) - 1}之间，"
            f"实际为{validation_size}"
        )

    if validation_size % class_count != 0:
        raise ValueError("validation_size 必须能被类别数整除，才能按类别均衡划分")

    validation_per_class = validation_size // class_count
    samples_per_class = len(train_source_dataset) // class_count
    if validation_per_class >= samples_per_class:
        raise ValueError("每类验证样本数必须小于每类总样本数")

    # 每个类别独立打乱，然后固定取相同数量的验证样本。
    # 默认配置下，每类450张训练、50张验证。
    split_generator = torch.Generator().manual_seed(random_seed)
    targets = torch.tensor(train_source_dataset.targets)
    train_indices = []
    validation_indices = []

    for class_index in range(class_count):
        class_indices = torch.where(targets == class_index)[0]
        random_order = torch.randperm(
            len(class_indices),
            generator=split_generator,
        )
        class_indices = class_indices[random_order].tolist()

        validation_indices.extend(class_indices[:validation_per_class])
        train_indices.extend(class_indices[validation_per_class:])

    train_dataset = Subset(train_source_dataset, train_indices)
    validation_dataset = Subset(
        validation_source_dataset,
        validation_indices,
    )

    use_pin_memory = torch.cuda.is_available()
    use_persistent_workers = num_workers > 0
    train_loader_generator = torch.Generator().manual_seed(random_seed)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
        generator=train_loader_generator,
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        train_source_dataset.classes,
    )
