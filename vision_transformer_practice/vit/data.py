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

    # 训练集变换
    # RandomCrop 和 RandomHorizontalFlip 每次读取时都会产生不同的图片变体，
    # 可以增加训练数据的多样性，降低从零训练 ViT 时的过拟合风险。
    train_transform = transforms.Compose(
        [
            # 随机裁剪：先在四周填充 4 个像素，再随机裁回 32 x 32。
            transforms.RandomCrop(size=32, padding=4),

            # 随机水平翻转：以 0.5 的概率左右翻转图片。
            transforms.RandomHorizontalFlip(p=0.5),

            # transforms.ToTensor
            # 作用：把 PIL 图片转换为 C x H x W 的浮点 Tensor，
            #      同时把像素值从 0～255 缩放到 0～1。
            # 参数：无。
            # 返回值：shape 为 3 x 32 x 32 的 Tensor。
            transforms.ToTensor(),

            # transforms.Normalize
            # 作用：分别对 RGB 三个通道执行 (像素值 - mean) / std。
            # 参数：mean 和 std 分别是 CIFAR-10 三个通道的均值与标准差。
            # 返回值：shape 不变的标准化 Tensor。
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    # 验证集与测试集不能使用随机增强，否则同一模型每次评估结果会发生波动。
    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    # datasets.CIFAR10
    # 作用：下载或读取 CIFAR-10 原训练集。
    # 重要参数：
    #   root：数据保存目录。
    #   train=True：读取 50000 张原训练图片。
    #   transform：每次读取图片时应用的预处理。
    #   download=True：本地不存在时自动下载，已经存在时直接复用。
    # 返回值：Dataset 对象，每个样本是 (image, label)。
    train_source_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=train_transform,
        download=True,
    )

    # 验证集与训练集使用相同的原始 CIFAR-10 图片，但应用确定性的评估变换。
    validation_source_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=evaluation_transform,
        download=True,
    )

    # train=False 表示读取 CIFAR-10 官方测试集，共 10000 张图片。
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        transform=evaluation_transform,
        download=True,
    )

    train_size = len(train_source_dataset) - validation_size
    split_generator = torch.Generator().manual_seed(random_seed)

    # torch.randperm
    # 作用：生成 0～49999 的随机排列，以固定种子划分训练和验证索引。
    # 返回值：包含全部样本索引的一维 Tensor。
    shuffled_indices = torch.randperm(
        len(train_source_dataset),
        generator=split_generator,
    ).tolist()
    validation_indices = shuffled_indices[:validation_size]
    train_indices = shuffled_indices[validation_size:]

    if len(train_indices) != train_size:
        raise RuntimeError("训练集划分数量不正确")

    # Subset
    # 作用：让训练和验证使用相同的固定索引划分，但分别应用各自的数据变换。
    # 参数：原始 Dataset 和需要保留的样本索引。
    # 返回值：训练集 45000 张、验证集 5000 张。
    train_dataset = Subset(train_source_dataset, train_indices)
    validation_dataset = Subset(
        validation_source_dataset,
        validation_indices,
    )

    # DataLoader
    # 作用：把 Dataset 包装成可以按 batch 迭代的数据加载器。
    # 重要参数：
    #   dataset：需要读取的数据集。
    #   batch_size：每次读取的样本数量。
    #   shuffle：是否在每个 epoch 打乱顺序，训练集为 True。
    #   num_workers：读取数据的子进程数量。
    #   pin_memory：使用 CUDA 时锁页内存，可加快 CPU 到 GPU 的传输。
    # 返回值：DataLoader；每次迭代得到 (images, labels)。
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
