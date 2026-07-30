from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 第一步：导包和加载数据集。
# 目标不是训练模型，而是先确认 MNIST 数据能够被正确下载、读取，并按 batch 输出。


# device 表示代码运行在哪个设备上。
# 如果你的电脑有可用 NVIDIA GPU，torch.cuda.is_available() 会返回 True，就使用 cuda。
# 否则使用 cpu。MNIST 很小，用 CPU 也能正常训练。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# batch_size 表示每次从数据集中取多少张图片送进模型。
# 训练时不是一张一张更新参数，而是一批一批更新参数。
batch_size = 64


# data_dir 是数据集保存的位置。
# 第一次运行时 download=True 会自动下载 MNIST 到这个文件夹。
# 后续再次运行时，如果本地已经有数据，就不会重复下载。
data_dir = Path(__file__).resolve().parent / "data"


# Windows + 普通 Python 脚本里可以设置 num_workers > 0。
# 但初学阶段先用 0，更容易避免多进程相关问题。
num_workers = 0


# transforms.Compose 用来把多个数据预处理步骤组合起来。
# MNIST 原始图片是 PIL Image，不是 Tensor，所以要先 ToTensor。
# ToTensor 会做两件事：
# 1. 把图片转成形状为 [channel, height, width] 的 Tensor。
# 2. 把像素值从 0~255 缩放到 0~1。
#
# Normalize((0.1307,), (0.3081,)) 是 MNIST 常用标准化参数。
# 因为 MNIST 是单通道灰度图，所以 mean 和 std 都只有一个数字。
# 标准化公式是：
# x_norm = (x - mean) / std
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


# train=True 表示加载训练集，MNIST 训练集有 60000 张图片。
# train=False 表示加载测试集，MNIST 测试集有 10000 张图片。
# transform=transform 表示每次读取图片时都会自动执行上面定义的数据预处理。
train_dataset = datasets.MNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root=data_dir,
    train=False,
    download=True,
    transform=transform,
)


# DataLoader 负责把 Dataset 包装成可以按 batch 迭代的对象。
# shuffle=True 表示每个 epoch 都打乱训练数据顺序，训练集通常要打乱。
# 测试集不需要训练模型参数，所以 shuffle=False 即可。
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)


def denormalize(image, mean=0.1307, std=0.3081):
    """把标准化后的 MNIST 图片还原到更适合显示的范围。"""
    return image * std + mean


def show_batch(images, labels, count=10):
    """可视化一个 batch 里的前 count 张图片。"""
    count = min(count, len(images))
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    for index, ax in enumerate(axes.flat):
        if index >= count:
            ax.axis("off")
            continue

        # images[index] 的形状是 [1, 28, 28]。
        # squeeze(0) 去掉通道维度，变成 [28, 28]，方便 matplotlib 显示灰度图。
        image = denormalize(images[index].squeeze(0))
        label = labels[index].item()

        ax.imshow(image, cmap="gray")
        ax.set_title(f"label: {label}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("device:", device)
    print("data dir:", data_dir)
    print("train samples:", len(train_dataset))
    print("test samples:", len(test_dataset))
    print("train batches:", len(train_loader))
    print("test batches:", len(test_loader))

    # next(iter(train_loader)) 取出训练集中的第一个 batch。
    # images 的形状应该是 [64, 1, 28, 28]：
    # 64 表示 batch_size，1 表示灰度通道，28 和 28 表示图片高宽。
    # labels 的形状应该是 [64]，里面是每张图片对应的真实数字标签。
    images, labels = next(iter(train_loader))

    print("images shape:", images.shape)
    print("labels shape:", labels.shape)
    print("first 10 labels:", labels[:10].tolist())

    show_batch(images, labels)
