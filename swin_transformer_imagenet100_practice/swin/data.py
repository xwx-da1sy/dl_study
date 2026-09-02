"""ImageNet-100 数据集、数据增强、Mixup/CutMix 和 DataLoader。"""

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import (
    BATCH_SIZE,
    CUTMIX_ALPHA,
    CUTMIX_PROBABILITY,
    DATA_ROOT,
    IMAGE_SIZE,
    MIXUP_ALPHA,
    NUM_CLASSES,
    NUM_WORKERS,
    RANDOM_ERASING_PROBABILITY,
    RANDOM_SEED,
    VALIDATION_SIZE,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def create_imagenet100_dataloaders(
    data_root=DATA_ROOT,
    batch_size=BATCH_SIZE,
    validation_size=VALIDATION_SIZE,
    num_workers=NUM_WORKERS,
    random_seed=RANDOM_SEED,
):
    """
    读取 ImageNet-100，并创建训练、内部验证和官方验证 DataLoader。

    data_root 下必须存在：
        train/<类别目录>/<图片>
        val/<类别目录>/<图片>

    返回：
        train_loader：官方 train 扣除内部验证后的训练集。
        validation_loader：从官方 train 按类别均衡划出的内部验证集。
        official_validation_loader：官方 val，只在最终模型确定后使用。
        class_names：100个类别目录名称。
    """
    data_root = Path(data_root)
    train_directory = data_root / "train"
    official_validation_directory = data_root / "val"
    missing_directories = [
        path
        for path in (train_directory, official_validation_directory)
        if not path.is_dir()
    ]
    if missing_directories:
        expected = "\n".join(
            [
                f"  {train_directory}/<类别目录>/<图片>",
                f"  {official_validation_directory}/<类别目录>/<图片>",
            ]
        )
        raise FileNotFoundError(
            "没有找到完整的 ImageNet-100 目录。期望结构：\n" + expected
        )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=IMAGE_SIZE,
                scale=(0.08, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(
                num_ops=2,
                magnitude=9,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
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
            transforms.Resize(
                size=256,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(size=IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    # 同一批 train 图片分别绑定随机增强和确定性增强，保证内部验证不含随机变换。
    train_source_dataset = datasets.ImageFolder(
        root=train_directory,
        transform=train_transform,
    )
    validation_source_dataset = datasets.ImageFolder(
        root=train_directory,
        transform=evaluation_transform,
    )
    official_validation_dataset = datasets.ImageFolder(
        root=official_validation_directory,
        transform=evaluation_transform,
    )

    class_count = len(train_source_dataset.classes)
    if class_count != NUM_CLASSES:
        raise RuntimeError(f"期望{NUM_CLASSES}个类别，实际读取到{class_count}个")
    if train_source_dataset.class_to_idx != official_validation_dataset.class_to_idx:
        raise RuntimeError("train 与 val 的类别目录不一致")
    if validation_size <= 0 or validation_size >= len(train_source_dataset):
        raise ValueError(
            f"validation_size 必须位于1到{len(train_source_dataset) - 1}之间，"
            f"实际为{validation_size}"
        )
    if validation_size % class_count != 0:
        raise ValueError("validation_size 必须能被类别数整除，才能按类别均衡划分")

    validation_per_class = validation_size // class_count
    split_generator = torch.Generator().manual_seed(random_seed)
    targets = torch.tensor(train_source_dataset.targets)
    train_indices = []
    validation_indices = []

    for class_index in range(class_count):
        class_indices = torch.where(targets == class_index)[0]
        if len(class_indices) <= validation_per_class:
            raise RuntimeError(
                f"类别{class_index}只有{len(class_indices)}张图片，"
                f"不足以划出{validation_per_class}张内部验证图片"
            )
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
    common_loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
        "persistent_workers": use_persistent_workers,
    }

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        generator=train_loader_generator,
        **common_loader_options,
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        shuffle=False,
        **common_loader_options,
    )
    official_validation_loader = DataLoader(
        dataset=official_validation_dataset,
        shuffle=False,
        **common_loader_options,
    )

    return (
        train_loader,
        validation_loader,
        official_validation_loader,
        train_source_dataset.classes,
    )
