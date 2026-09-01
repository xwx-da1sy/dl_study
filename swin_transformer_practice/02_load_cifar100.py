"""下载并加载 CIFAR-100，生成训练、验证和测试 DataLoader。"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# CIFAR-100 图片本身是 3 x 32 x 32。
# 训练集加入随机裁剪和水平翻转，让同一张图片产生轻微变化，减少过拟合。
# 验证集和测试集不能使用随机增强，否则每次评估面对的数据都会变化。
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ]
)


def create_dataloaders(batch_size=128, num_workers=4, seed=42):
    """创建按类别均衡划分的 45,000/5,000/10,000 DataLoader。"""

    data_root = Path(__file__).resolve().parent / "data"

    # train_dataset 和 validation_dataset 指向同一批 50,000 张原始训练图片，
    # 但使用不同 transform：训练集有随机增强，验证集没有随机增强。
    train_dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        transform=TRAIN_TRANSFORM,
        download=True,
    )
    validation_dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        transform=EVAL_TRANSFORM,
        download=False,
    )
    test_dataset = datasets.CIFAR100(
        root=data_root,
        train=False,
        transform=EVAL_TRANSFORM,
        download=True,
    )

    # CIFAR-100 官方只提供训练集和测试集。官方训练集的每个类别都有500张图片，
    # 所以每类固定取50张做验证、剩余450张训练：验证集不会偏向某几个类别。
    # 固定 seed 后，每次运行都会得到相同的划分，实验结果才可以公平比较。
    split_generator = torch.Generator().manual_seed(seed)
    targets = torch.tensor(train_dataset.targets)
    train_indices = []
    validation_indices = []

    for class_index in range(len(train_dataset.classes)):
        class_indices = torch.where(targets == class_index)[0]
        random_order = torch.randperm(len(class_indices), generator=split_generator)
        class_indices = class_indices[random_order].tolist()

        validation_indices.extend(class_indices[:50])
        train_indices.extend(class_indices[50:])

    train_subset = Subset(train_dataset, train_indices)
    validation_subset = Subset(validation_dataset, validation_indices)

    # 只有训练集需要 shuffle；验证集和测试集保持固定顺序即可。
    # CUDA 训练时 pin_memory=True 可以加快 CPU 到 GPU 的数据传输。
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_subset,
        shuffle=True,
        generator=train_loader_generator,
        **loader_options,
    )
    validation_loader = DataLoader(validation_subset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    return train_loader, validation_loader, test_loader, train_dataset.classes


def main():
    train_loader, validation_loader, test_loader, class_names = create_dataloaders()

    print("训练集：", len(train_loader.dataset))
    print("验证集：", len(validation_loader.dataset))
    print("测试集：", len(test_loader.dataset))
    print("类别数：", len(class_names))

    # 取一个 batch，确认后续网络实际收到的数据形状。
    images, labels = next(iter(train_loader))
    print("图片 batch shape：", tuple(images.shape))
    print("标签 batch shape：", tuple(labels.shape))
    print("前10个标签编号：", labels[:10].tolist())
    print("前10个类别名称：", [class_names[index] for index in labels[:10].tolist()])

    # batch_size=128 时：
    # images 的 shape 是 128 x 3 x 32 x 32。
    # labels 的 shape 是 128，每个数字范围为 0~99，对应一个 CIFAR-100 类别。


if __name__ == "__main__":
    main()
