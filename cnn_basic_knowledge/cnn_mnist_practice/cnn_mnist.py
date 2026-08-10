"""CNN-MNIST 手写数字识别实战。

当前进度：
1. 导入需要用到的包。
2. 准备 MNIST 训练集和测试集。
3. 创建 DataLoader，把数据集按 batch 送给模型。
4. 定义 CNN 模型结构。
5. 定义损失函数。
6. 定义 Adam 优化器。
7. 定义训练一个 batch 的流程。
8. 定义训练一个 epoch 的流程。
9. 定义测试/评估流程。
10. 定义完整训练流程。
11. 定义 main 入口，把完整实战流程串起来。
12. 加载训练好的模型，并预测单张图片。
"""

import json
from pathlib import Path

# torch：PyTorch 的核心库，负责张量计算、自动求导、模型训练等基础能力。
import torch

# nn：神经网络模块，后面会用它定义卷积层、池化层、全连接层、损失函数等。
from torch import nn

# DataLoader：数据加载器，后面会用它把数据集按 batch 一批一批送给模型。
from torch.utils.data import DataLoader

# datasets：torchvision 提供的常用数据集工具，这里用它导入 MNIST。
# transforms：图像预处理工具，这里用它把图片转换成 Tensor。
from torchvision import datasets, transforms

# pyplot：画图工具，后面可以用来查看图片、loss 曲线或准确率变化。
import matplotlib.pyplot as plt


# DATA_DIR：MNIST 数据保存位置。
# 这里把数据放在 cnn_basic_knowledge/data 下面，避免和代码文件混在一起。
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# MODEL_DIR：训练结果保存位置。
# 这里用来保存模型参数和每轮训练记录。
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

# MODEL_PATH：模型参数保存文件。
# 只保存参数，不保存整个模型对象，后面更容易重新加载。
MODEL_PATH = MODEL_DIR / "cnn_mnist_model.pth"

# HISTORY_PATH：训练记录保存文件。
# 保存每一轮的 train_loss、test_loss 和 test_accuracy。
HISTORY_PATH = MODEL_DIR / "cnn_mnist_history.json"

# BATCH_SIZE：每次送入模型的图片数量。
# 比如 BATCH_SIZE = 64，表示模型一次看 64 张图片。
BATCH_SIZE = 64

# NUM_CLASSES：MNIST 一共有 10 个类别，分别是数字 0 到 9。
NUM_CLASSES = 10

# DROPOUT_PROB：Dropout 的丢弃比例。
# 0.2 表示训练时随机让 20% 的特征暂时不参与本轮计算，用来缓解过拟合。
DROPOUT_PROB = 0.2

# LEARNING_RATE：学习率。
# Adam 优化器会根据梯度更新参数，学习率控制每次更新参数的步子大小。
LEARNING_RATE = 0.001

# EPOCHS：训练轮数。
# 5 表示让模型完整看 5 遍训练集。
EPOCHS = 100

# DEVICE：模型训练使用的设备。
# 如果电脑支持 CUDA，就使用 GPU；否则使用 CPU。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform():
    """构建 MNIST 图片预处理流程。

    函数作用：
        把原始 MNIST 图片转换成 PyTorch 可以处理的 Tensor。

    参数：
        无。

    返回值：
        transform：图片预处理流程。

    说明：
        MNIST 原始图片不是 Tensor，模型不能直接训练。
        transforms.ToTensor() 会把图片转换成形状类似 1 x 28 x 28 的张量。
    """
    transform = transforms.ToTensor()
    return transform


def load_mnist_datasets(data_dir=DATA_DIR, download=True):
    """导入 MNIST 训练集和测试集。


    函数作用：
        从指定目录读取 MNIST 数据集。
        如果本地没有数据，并且 download=True，就自动下载。

    参数：
        data_dir：
            数据集保存目录，默认是 cnn_basic_knowledge/data。
        download：
            是否允许自动下载 MNIST。
            True 表示本地没有数据时自动下载。
            False 表示只从本地读取，不下载。

    返回值：
        train_dataset：
            MNIST 训练集，用来训练模型、更新参数。
        test_dataset：
            MNIST 测试集，用来评估模型效果，不用来更新参数。
    """
    transform = build_transform()

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=download,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=download,
    )

    return train_dataset, test_dataset


def create_data_loaders(train_dataset, test_dataset, batch_size=BATCH_SIZE):
    """创建训练集和测试集的 DataLoader。

    函数作用：
        把 Dataset 包装成 DataLoader。
        DataLoader 会按 batch 一批一批取出图片和标签。

    参数：
        train_dataset：
            MNIST 训练集，用来训练模型、更新参数。
        test_dataset：
            MNIST 测试集，用来评估模型效果，不用来更新参数。
        batch_size：
            每个 batch 中包含多少张图片。
            默认值是 64。

    返回值：
        train_loader：
            训练集 DataLoader。
            训练时会打乱数据顺序，让模型不要总是按固定顺序看图片。
        test_loader：
            测试集 DataLoader。
            测试时通常不需要打乱顺序，因为只是评估模型效果。

    说明：
        Dataset 像一本完整练习册，保存全部图片和标签。
        DataLoader 像每次从练习册里取出一小批题目，交给模型学习。
    """
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


class CNNMnistModel(nn.Module):
    """CNN-MNIST 分类模型。

    类的作用：
        接收 MNIST 手写数字图片，输出 10 个类别分数。

    输入形状：
        x：形状通常是 B x 1 x 28 x 28。
        B 表示 batch size，1 表示灰度图通道数，28 x 28 表示图片高和宽。

    输出形状：
        output：形状通常是 B x 10。
        每张图片对应 10 个类别分数，分别表示数字 0 到 9 的得分。

    模型结构：
        Conv2d: 1 -> 8，卷积核 3 x 3，padding=1
        ReLU
        MaxPool2d: 高和宽减半
        Conv2d: 8 -> 32，卷积核 3 x 3，padding=1
        ReLU
        MaxPool2d: 高和宽再次减半
        Flatten
        Linear: 1568 -> 128
        ReLU
        Dropout: p=0.2
        Linear: 128 -> 10
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout_prob=DROPOUT_PROB):
        """初始化 CNN 模型结构。

        函数作用：
            定义 CNN 需要用到的网络层。

        参数：
            num_classes：
                输出类别数。
                MNIST 有 10 个类别，所以默认值是 10。
            dropout_prob：
                Dropout 的丢弃比例。
                默认值是 0.2。

        返回值：
            无。
            __init__ 只负责把模型层准备好。
        """
        super().__init__()

        # features：特征提取部分。
        # 输入：B x 1 x 28 x 28
        self.features = nn.Sequential(
            # 第 1 个卷积层：
            # 1 表示输入是 1 通道灰度图。
            # 8 表示使用 8 个卷积核，输出 8 张特征图。
            # kernel_size=3 表示卷积核大小是 3 x 3。
            # padding=1 可以让高和宽保持 28 x 28 不变。
            # 输出形状：B x 8 x 28 x 28
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),

            # ReLU：激活函数，只改变数值，不改变形状。
            # 输出形状：B x 8 x 28 x 28
            nn.ReLU(),

            # MaxPool2d：最大池化层。
            # kernel_size=2 表示每次看 2 x 2 区域。
            # stride=2 表示窗口每次移动 2 格。
            # 输出形状：B x 8 x 14 x 14
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第 2 个卷积层：
            # 输入通道数是 8，因为上一层输出了 8 张特征图。
            # 输出通道数是 32，表示这一层输出 32 张特征图。
            # padding=1 继续保持高和宽不变。
            # 输出形状：B x 32 x 14 x 14
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1),

            # ReLU：只改变数值，不改变形状。
            # 输出形状：B x 32 x 14 x 14
            nn.ReLU(),

            # 第二次最大池化，高和宽从 14 x 14 变成 7 x 7。
            # 输出形状：B x 32 x 7 x 7
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # classifier：分类部分。
        # 经过特征提取后，形状是 B x 32 x 7 x 7。
        # Flatten 后每张图片的特征长度是 32 x 7 x 7 = 1568。
        self.classifier = nn.Sequential(
            # Flatten：把每张图片的多张特征图展平成一维向量。
            # 输出形状：B x 1568
            nn.Flatten(),

            # 第 1 个全连接层：
            # 输入 1568 个特征，输出 128 个特征。
            # 输出形状：B x 128
            nn.Linear(in_features=32 * 7 * 7, out_features=128),

            # ReLU：增加非线性表达能力，形状不变。
            # 输出形状：B x 128
            nn.ReLU(),

            # Dropout：训练时随机丢弃一部分特征，用来缓解过拟合。
            # 这里 p=0.2，表示每次训练时随机丢弃 20% 的特征。
            # Dropout 只在训练模式下生效，评估模式下不会随机丢弃特征。
            # 输出形状：B x 128
            nn.Dropout(p=dropout_prob),

            # 输出层：
            # 输入 128 个特征，输出 10 个类别分数。
            # 输出形状：B x 10
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self, x):
        """定义模型前向传播过程。

        函数作用：
            规定图片进入模型后，按什么顺序经过各层。

        参数：
            x：
                输入图片 batch。
                形状通常是 B x 1 x 28 x 28。

        返回值：
            output：
                模型输出的类别分数。
                形状通常是 B x 10。
        """
        x = self.features(x)
        output = self.classifier(x)
        return output


def create_loss_fn():
    """创建分类任务使用的损失函数。

    函数作用：
        创建交叉熵损失函数，用来衡量模型预测和真实标签之间的差距。

    参数：
        无。

    返回值：
        loss_fn：
            交叉熵损失函数。

    说明：
        MNIST 是 10 分类任务，所以适合使用 CrossEntropyLoss。

        模型最后一层直接输出 B x 10 的原始类别分数。
        这里不需要在模型中手动添加 Softmax。

        原因是：
        CrossEntropyLoss 内部已经包含了和 Softmax 相关的处理。
        如果训练时先手动 Softmax，再交给 CrossEntropyLoss，反而容易让训练效果变差。

        Softmax 更适合放在预测展示阶段。
        比如模型训练好之后，如果想把类别分数转换成“概率”，再使用 Softmax。
    """
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn


def create_optimizer(model, learning_rate=LEARNING_RATE):
    """创建 Adam 优化器。

    函数作用：
        创建优化器，用来根据梯度更新模型中的可学习参数。

    参数：
        model：
            需要训练的 CNN 模型。
            优化器会更新这个模型中的卷积层和全连接层参数。
        learning_rate：
            学习率，控制每次参数更新的步子大小。
            默认值是 0.001。

    返回值：
        optimizer：
            Adam 优化器。

    说明：
        这里先不使用余弦退火学习率调度器。
        第一版实战先让训练主流程保持清楚：
        前向传播 -> 计算 loss -> 反向传播 -> Adam 更新参数。
    """
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
    )
    return optimizer


def train_one_batch(model, images, labels, loss_fn, optimizer, device=DEVICE):
    """训练一个 batch。

    函数作用：
        完成一次最小训练流程：
        清空旧梯度 -> 前向传播 -> 计算 loss -> 反向传播 -> 更新参数。

    参数：
        model：
            需要训练的 CNN 模型。
        images：
            一个 batch 的图片。
            形状通常是 B x 1 x 28 x 28。
        labels：
            一个 batch 的真实标签。
            形状通常是 B。
            每个标签是 0 到 9 中的一个整数。
        loss_fn：
            损失函数，这里使用交叉熵损失。
        optimizer：
            优化器，这里使用 Adam。
        device：
            训练设备。
            可以是 CPU，也可以是 CUDA GPU。

    返回值：
        loss_value：
            当前 batch 的 loss 数值。
            返回普通 Python 数字，方便后面打印或记录。

    说明：
        这是训练循环中最核心的一小步。
        后面训练一个 epoch，本质上就是对很多个 batch 重复调用这套流程。
    """
    # model.train()：把模型切换到训练模式。
    # Dropout 只有在训练模式下才会随机丢弃一部分特征。
    model.train()

    # 把图片和标签放到和模型相同的设备上。
    images = images.to(device)
    labels = labels.to(device)

    # 第 1 步：清空上一轮留下的梯度。
    # PyTorch 默认会累加梯度，所以每次反向传播前都要先清空。
    optimizer.zero_grad()

    # 第 2 步：前向传播。
    # 图片进入 CNN，得到每张图片对应 10 个类别的原始分数。
    outputs = model(images)

    # 第 3 步：计算 loss。
    # outputs 是模型预测分数，labels 是真实答案。
    loss = loss_fn(outputs, labels)

    # 第 4 步：反向传播。
    # 根据 loss 计算每个可学习参数的梯度。
    loss.backward()

    # 第 5 步：更新参数。
    # Adam 根据梯度修改卷积层和全连接层中的权重、bias。
    optimizer.step()

    # loss.item() 会把张量形式的 loss 转成普通数字。
    loss_value = loss.item()
    return loss_value


def train_one_epoch(model, train_loader, loss_fn, optimizer, device=DEVICE):
    """训练一个 epoch。

    函数作用：
        让模型完整看一遍训练集。
        训练集会被 DataLoader 拆成很多个 batch。
        每个 batch 都会调用 train_one_batch 完成一次参数更新。

    参数：
        model：
            需要训练的 CNN 模型。
        train_loader：
            训练集 DataLoader。
            每次循环会取出一个 batch 的图片和标签。
        loss_fn：
            损失函数，这里使用交叉熵损失。
        optimizer：
            优化器，这里使用 Adam。
        device：
            训练设备。
            可以是 CPU，也可以是 CUDA GPU。

    返回值：
        average_loss：
            当前 epoch 的平均 loss。
            它表示这一轮训练中，每个 batch 的 loss 平均水平。

    说明：
        一个 batch 是训练的一小步。
        一个 epoch 是模型完整看完一遍训练集。
    """
    total_loss = 0.0
    batch_count = 0

    for images, labels in train_loader:
        loss_value = train_one_batch(
            model=model,
            images=images,
            labels=labels,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        total_loss += loss_value
        batch_count += 1

    average_loss = total_loss / batch_count
    return average_loss


def evaluate(model, test_loader, loss_fn, device=DEVICE):
    """评估模型在测试集上的效果。

    函数作用：
        在测试集上计算平均 loss 和准确率。
        评估阶段只看模型效果，不更新模型参数。

    参数：
        model：
            已经定义好的 CNN 模型。
        test_loader：
            测试集 DataLoader。
            每次循环会取出一个 batch 的图片和标签。
        loss_fn：
            损失函数，这里使用交叉熵损失。
        device：
            评估设备。
            需要和模型所在设备保持一致。

    返回值：
        average_loss：
            测试集上的平均 loss。
        accuracy：
            测试集准确率。
            例如 0.98 表示准确率是 98%。

    说明：
        model.eval() 会把模型切换到评估模式。
        Dropout 在评估模式下不会随机丢弃特征。

        torch.no_grad() 表示评估时不计算梯度。
        因为测试阶段只是检查模型效果，不需要反向传播，也不需要更新参数。
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_count = 0

    with torch.no_grad():
        for images, labels in test_loader:
            # 把图片和标签放到和模型相同的设备上。
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播：得到每张图片对应 10 个类别的原始分数。
            outputs = model(images)

            # 计算当前 batch 的 loss。
            loss = loss_fn(outputs, labels)

            # torch.argmax 会找出每张图片分数最高的类别。
            # dim=1 表示在 10 个类别分数这个维度上找最大值。
            predictions = torch.argmax(outputs, dim=1)

            # 统计当前 batch 预测正确的图片数量。
            correct = (predictions == labels).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
            batch_count += 1

    average_loss = total_loss / batch_count
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def train_model(
    model,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    epochs=EPOCHS,
    device=DEVICE,
):
    """完整训练模型。

    函数作用：
        按照指定 epoch 数训练 CNN 模型。
        每个 epoch 结束后，在测试集上评估一次模型效果。

    参数：
        model：
            需要训练的 CNN 模型。
        train_loader：
            训练集 DataLoader。
            用来训练模型、更新参数。
        test_loader：
            测试集 DataLoader。
            用来评估模型效果，不更新参数。
        loss_fn：
            损失函数，这里使用交叉熵损失。
        optimizer：
            优化器，这里使用 Adam。
        epochs：
            训练轮数。
            默认值是 5。
        device：
            训练和评估使用的设备。
            需要和模型所在设备保持一致。

    返回值：
        history：
            训练记录。
            它是一个列表，每个元素保存一轮训练后的结果。
            每轮结果包括：
            - epoch：第几轮
            - train_loss：训练集平均 loss
            - test_loss：测试集平均 loss
            - test_accuracy：测试集准确率

    说明：
        这个函数只是把前面的小函数串起来。

        一轮完整流程是：
        训练一个 epoch -> 在测试集上评估 -> 记录结果。
    """
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_accuracy = evaluate(
            model=model,
            test_loader=test_loader,
            loss_fn=loss_fn,
            device=device,
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_accuracy: {test_accuracy:.4f}"
        )

    return history


def save_training_result(model, history, model_path=MODEL_PATH, history_path=HISTORY_PATH):
    """保存训练结果。

    函数作用：
        保存训练好的模型参数和训练过程记录。

    参数：
        model：
            训练后的 CNN 模型。
        history：
            训练记录。
            里面保存每一轮的 train_loss、test_loss、test_accuracy。
        model_path：
            模型参数保存路径。
            默认保存到 cnn_basic_knowledge/models/cnn_mnist_model.pth。
        history_path：
            训练记录保存路径。
            默认保存到 cnn_basic_knowledge/models/cnn_mnist_history.json。

    返回值：
        无。

    说明：
        torch.save(model.state_dict(), model_path) 保存的是模型参数。
        以后如果要预测或继续训练，可以重新创建同样结构的模型，再加载这些参数。
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)


def load_trained_model(model_path=MODEL_PATH, device=DEVICE):
    """加载训练好的 CNN 模型。

    函数作用：
        创建一个和训练时结构相同的 CNN 模型。
        然后从模型参数文件中加载已经训练好的参数。

    参数：
        model_path：
            模型参数文件路径。
            默认读取 cnn_basic_knowledge/models/cnn_mnist_model.pth。
        device：
            模型加载到哪个设备上。
            可以是 CPU，也可以是 CUDA GPU。

    返回值：
        model：
            已经加载好参数的 CNN 模型。

    说明：
        保存时保存的是 state_dict，也就是模型参数。
        加载时需要先创建同样结构的 CNNMnistModel，再把参数装进去。
    """
    model = CNNMnistModel().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_one_image(model, image, device=DEVICE):
    """预测单张 MNIST 图片。

    函数作用：
        使用训练好的 CNN 模型，对一张手写数字图片进行分类预测。

    参数：
        model：
            已经训练好的 CNN 模型。
        image：
            单张 MNIST 图片。
            形状通常是 1 x 28 x 28。
        device：
            预测使用的设备。
            需要和模型所在设备保持一致。

    返回值：
        predicted_label：
            模型预测出的数字类别。
            取值范围是 0 到 9。
        probabilities：
            每个类别对应的预测概率。
            形状是 10。

    说明：
        训练时模型最后不加 Softmax，因为 CrossEntropyLoss 内部会处理。
        预测展示时可以使用 Softmax，把 10 个类别分数转换成概率。
    """
    model.eval()

    # 单张图片形状是 1 x 28 x 28。
    # 模型需要 batch 维度，所以这里用 unsqueeze(0) 变成 1 x 1 x 28 x 28。
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()

    return predicted_label, probabilities.squeeze(0).cpu()


def demo_predict_one_image(index=0, model_path=MODEL_PATH, device=DEVICE):
    """演示预测测试集中的一张图片。

    函数作用：
        加载训练好的模型。
        读取 MNIST 测试集。
        取出指定下标的一张图片，并输出预测结果。

    参数：
        index：
            要预测测试集中第几张图片。
            默认值是 0。
        model_path：
            模型参数文件路径。
            默认读取 cnn_basic_knowledge/models/cnn_mnist_model.pth。
        device：
            预测使用的设备。

    返回值：
        result：
            单张图片预测结果。
            包含真实标签、预测标签、预测是否正确、各类别概率。

    说明：
        这个函数只是演示单张图片预测。
        它不会训练模型，也不会更新模型参数。
    """
    model = load_trained_model(model_path=model_path, device=device)

    _, test_dataset = load_mnist_datasets(download=False)
    image, true_label = test_dataset[index]

    predicted_label, probabilities = predict_one_image(
        model=model,
        image=image,
        device=device,
    )

    result = {
        "index": index,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "is_correct": predicted_label == true_label,
        "probabilities": probabilities.tolist(),
    }

    print(f"测试集第 {index} 张图片")
    print(f"真实标签：{true_label}")
    print(f"预测标签：{predicted_label}")
    print(f"是否预测正确：{result['is_correct']}")

    return result


def main():
    """运行 CNN-MNIST 完整实战流程。

    函数作用：
        按顺序完成：
        导入数据集 -> 创建 DataLoader -> 创建模型 -> 创建 loss ->
        创建优化器 -> 训练模型。

    参数：
        无。

    返回值：
        model：
            训练后的 CNN 模型。
        history：
            每一轮训练和测试的记录。

    说明：
        这个函数是真正把前面的所有步骤串起来的入口。
        运行本文件时，会从这里开始执行完整训练流程。
    """
    print(f"当前训练设备：{DEVICE}")

    train_dataset, test_dataset = load_mnist_datasets()
    train_loader, test_loader = create_data_loaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )

    model = CNNMnistModel().to(DEVICE)
    loss_fn = create_loss_fn()
    optimizer = create_optimizer(model)

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=DEVICE,
    )

    save_training_result(model=model, history=history)

    return model, history


if __name__ == "__main__":
    main()
