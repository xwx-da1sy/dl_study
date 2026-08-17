"""CIFAR-10 数据集与 DataLoader。"""

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
        下载或读取 CIFAR-10，将原训练集划分为训练集和验证集，
        最后创建训练、验证和测试 DataLoader。

    本版本针对 Tiny ViT 从零训练时的过拟合加强了训练集数据增强：
        RandomCrop + RandomHorizontalFlip + RandAugment + RandomErasing。
        验证集和测试集仍然只做确定性的 ToTensor + Normalize。

    参数：
        data_root：数据集保存目录。
        batch_size：每个 batch 包含的图片数量。
        validation_size：从原训练集中划出的验证样本数量。
        num_workers：DataLoader 读取数据使用的子进程数量。
        random_seed：控制随机划分结果，保证实验可以复现。

    返回值：
        train_loader：训练集 DataLoader，图片 shape 为 B x 3 x 32 x 32。
        validation_loader：验证集 DataLoader。
        test_loader：测试集 DataLoader。
        class_names：10 个类别名称组成的列表。
    """

    # 训练集使用更强的数据增强。
    # RandAugment 在 PIL 图片阶段执行；RandomErasing 需要 Tensor，因此放在 Normalize 之后。
    train_transform = transforms.Compose(
        [
            # 先在四周填充 4 个像素，再随机裁回 32 x 32。
            transforms.RandomCrop(size=32, padding=4),

            # 以 0.5 的概率进行水平翻转。
            transforms.RandomHorizontalFlip(p=0.5),

            # 随机选择 2 个图像增强操作，magnitude=9 控制增强强度。
            # 相比只有 Crop/Flip，可以显著增加训练样本的外观多样性。
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
    evaluation_transform = transforms.Compose(
        [
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
            f"validation_size 必须位于 1～{len(train_source_dataset) - 1} 之间，"
            f"实际为 {validation_size}"
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
