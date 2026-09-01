"""CIFAR-100 数据集、数据增强和 DataLoader。"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import (
    BATCH_SIZE,
    DATA_ROOT,
    NUM_CLASSES,
    NUM_WORKERS,
    RANDOM_SEED,
    VALIDATION_SIZE,
)


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


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
                p=0.25,
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
