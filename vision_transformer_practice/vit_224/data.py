"""CIFAR-10 数据集（Resize 到 224×224）与 DataLoader。

把 CIFAR-10 原始 32×32 图片放大到 224×224，配合标准 ViT patch=16 架构。

注意：放大不增加真实信息（32×32 是信息天花板），本工程目的是练手 224 ViT
架构流程，不追求比原 32×32 版更高的准确率。
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import (
    BATCH_SIZE,
    DATA_ROOT,
    NUM_WORKERS,
    RANDOM_SEED,
    VALIDATION_SIZE,
)


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def create_cifar10_dataloaders(
    data_root=DATA_ROOT,
    batch_size=BATCH_SIZE,
    validation_size=VALIDATION_SIZE,
    num_workers=NUM_WORKERS,
    random_seed=RANDOM_SEED,
):
    """
    作用：
        读取 CIFAR-10，把 32×32 图片 Resize 到 224×224，
        将原训练集划分为训练集和验证集，最后创建训练、验证、测试 DataLoader。

    参数：
        data_root：数据集保存目录。
        batch_size：每个 batch 包含的图片数量。
        validation_size：从原训练集中划出的验证样本数量。
        num_workers：DataLoader 读取数据使用的子进程数量。
        random_seed：控制随机划分结果，保证实验可以复现。

    返回值：
        train_loader：训练集 DataLoader，图片 shape 为 B x 3 x 224 x 224。
        validation_loader：验证集 DataLoader。
        test_loader：测试集 DataLoader。
        class_names：10 个类别名称组成的列表。
    """

    # 训练集变换：先 Resize 到 224，再做随机增强。
    # RandomCrop(224, padding=28)：先在四周填充 28 个像素（224/8≈28），再裁回 224x224，
    #   模拟目标在画面中的位置变化。
    # RandAugment 和 RandomErasing 沿用原正则化版配置。
    train_transform = transforms.Compose(
        [
            # 把 32×32 放大到 224×224（双线性插值，不增加真实信息）。
            transforms.Resize(224),

            # 随机裁剪：先填 28 再裁回 224，模拟位置变化。
            transforms.RandomCrop(224, padding=28),

            # 随机水平翻转：以 0.5 的概率左右翻转图片。
            transforms.RandomHorizontalFlip(p=0.5),

            # 随机选择 2 个图像增强操作，magnitude=9 控制增强强度。
            transforms.RandAugment(num_ops=2, magnitude=9),

            # PIL Image -> Tensor，并把像素值缩放到 0～1。
            transforms.ToTensor(),

            # CIFAR-10 标准化。
            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=CIFAR10_STD,
            ),

            # 以 0.25 的概率随机擦除一小块区域，迫使模型不要依赖局部记忆。
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.20),
                ratio=(0.3, 3.3),
                value=0,
            ),
        ]
    )

    # 验证集与测试集不能使用随机增强，否则同一模型每次评估结果会发生波动。
    # 只做 Resize + ToTensor + Normalize 这三步确定性预处理。
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=CIFAR10_STD,
            ),
        ]
    )

    # 训练集：同一批原始 CIFAR-10 图片，但读取时应用随机训练增强。
    train_source_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=train_transform,
        download=True,
    )

    # 验证集：使用同一原始训练集中的固定样本，但不使用随机增强。
    validation_source_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=evaluation_transform,
        download=True,
    )

    # 官方测试集，共 10000 张图片。
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        transform=evaluation_transform,
        download=True,
    )

    if validation_size <= 0 or validation_size >= len(train_source_dataset):
        raise ValueError(
            "validation_size 必须位于 1～%d 之间，实际为 %d"
            % (len(train_source_dataset) - 1, validation_size)
        )

    train_size = len(train_source_dataset) - validation_size
    split_generator = torch.Generator().manual_seed(random_seed)

    # 固定随机种子生成索引排列，保证训练/验证划分可复现。
    shuffled_indices = torch.randperm(
        len(train_source_dataset),
        generator=split_generator,
    ).tolist()
    validation_indices = shuffled_indices[:validation_size]
    train_indices = shuffled_indices[validation_size:]

    if len(train_indices) != train_size:
        raise RuntimeError("训练集划分数量不正确")

    # 训练和验证使用同一组固定索引划分，但各自应用不同 transform。
    train_dataset = Subset(train_source_dataset, train_indices)
    validation_dataset = Subset(
        validation_source_dataset,
        validation_indices,
    )

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        train_source_dataset.classes,
    )
