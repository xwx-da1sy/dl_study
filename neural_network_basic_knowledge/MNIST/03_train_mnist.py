from pathlib import Path
import importlib.util

import torch
import torch.nn as nn


# 第三步：训练 MNIST 手写数字识别模型。
# 前面我们已经分别写好了两个文件：
# 1. 01_load_mnist_data.py：负责加载 MNIST 数据集。
# 2. 02_define_mlp_model.py：负责定义 MLP 模型。
# 所以这个训练脚本要复用前面的文件，不重复写已经完成的代码。


# 01_load_mnist_data.py 这个文件名以数字开头。
# Python 不能直接写：
#
# from 01_load_mnist_data import train_loader
#
# 因为模块名不能以数字开头。
# importlib 可以让我们按照文件路径导入一个 Python 文件。
data_module_path = Path(__file__).resolve().parent / "01_load_mnist_data.py"

spec = importlib.util.spec_from_file_location("load_mnist_data", data_module_path)
mnist_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mnist_data)


# 复用 01_load_mnist_data.py 里已经准备好的数据对象。
train_dataset = mnist_data.train_dataset
test_dataset = mnist_data.test_dataset
train_loader = mnist_data.train_loader
test_loader = mnist_data.test_loader


# 02_define_mlp_model.py 这个文件名也以数字开头。
# 所以这里同样使用 importlib 按文件路径导入它。
model_module_path = Path(__file__).resolve().parent / "02_define_mlp_model.py"

spec = importlib.util.spec_from_file_location("define_mlp_model", model_module_path)
mlp_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlp_model)


# 从 02_define_mlp_model.py 里取出已经写好的 MLP 类。
MLP = mlp_model.MLP


# device 表示模型和数据要放在哪个设备上运行。
# 如果电脑有可用 GPU，就使用 cuda；否则使用 cpu。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 创建模型对象，并把模型移动到 device 上。
# 后面训练时，图片和标签也要移动到同一个 device。
model = MLP().to(device)


# criterion 是损失函数。
# MNIST 是 10 分类任务，所以使用 CrossEntropyLoss。
# 注意：CrossEntropyLoss 接收的是模型输出的 logits，
# 不需要在模型最后手动加 softmax。
criterion = nn.CrossEntropyLoss()


# learning_rate 是学习率，表示每次参数更新的步子大小。
# 学习率太大，训练可能不稳定；学习率太小，训练会很慢。
learning_rate = 0.001


# optimizer 是优化器，负责根据梯度更新模型参数。
# model.parameters() 表示把模型中所有可训练参数交给优化器管理。
# Adam 是一个常用且比较省心的优化器，适合入门项目先使用。
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# num_epochs 表示完整遍历训练集多少轮。
# 1 个 epoch = 模型把 60000 张训练图片都学习一遍。
# MNIST 比较简单，先训练 5 轮就能看到不错的效果。
num_epochs = 100


# model_save_path 表示模型训练完成后保存到哪里。
# .pth 是 PyTorch 保存模型参数时常用的文件后缀。
model_save_path = Path(__file__).resolve().parent / "mlp_mnist_deeper.pth"


def train_one_epoch():
    # model.train() 表示把模型切换到训练模式。
    # 如果模型里有 Dropout、BatchNorm，训练模式和测试模式的行为会不同。
    model.train()

    # total_loss 用来累计每个 batch 的损失。
    # correct 用来累计预测正确的样本数量。
    # total 用来累计总样本数量。
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # 模型已经被移动到 device 上。
        # 所以输入图片和真实标签也要移动到同一个 device 上。
        images = images.to(device)
        labels = labels.to(device)

        # 清空上一轮 batch 留下的梯度。
        # PyTorch 默认会累积梯度，所以每个 batch 训练前都要清零。
        optimizer.zero_grad()

        # 前向传播：把图片输入模型，得到 10 个类别的原始分数 logits。
        outputs = model(images)

        # 计算损失：比较模型输出 outputs 和真实标签 labels，这里用交叉熵。
        loss = criterion(outputs, labels)

        # 反向传播：根据 loss 计算每个参数的梯度。
        loss.backward()

        # 参数更新：优化器根据梯度调整模型参数。
        optimizer.step()

        # loss.item() 把 Tensor 类型的 loss 转成 Python 数字，方便统计。
        total_loss += loss.item()

        # outputs 的形状是 [batch_size, 10]。
        # argmax(dim=1) 表示取每张图片分数最高的类别作为预测结果。
        preds = outputs.argmax(dim=1)

        # 统计当前 batch 中预测正确的数量。
        correct += (preds == labels).sum().item()

        # labels.size(0) 表示当前 batch 的样本数量。
        total += labels.size(0)

    average_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return average_loss, accuracy


def evaluate():
    # model.eval() 表示把模型切换到评估模式。
    # 测试集只用来评估模型效果，不用来更新模型参数。
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # torch.no_grad() 会关闭梯度计算。
    # 评估阶段不需要 backward，也不需要保存梯度。
    # 这样可以节省显存和计算时间。
    with torch.no_grad():
        for images, labels in test_loader:
            # 测试数据也要移动到和模型相同的设备上。
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播，得到 10 个类别分数。
            outputs = model(images)

            # 计算测试集上的 loss。
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # 取分数最高的类别作为预测结果。
            preds = outputs.argmax(dim=1)

            # 累计预测正确数量和总样本数量。
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / len(test_loader)
    accuracy = correct / total

    return average_loss, accuracy


if __name__ == "__main__":
    print("device:", device)
    print("model:", model)
    print("train samples:", len(train_dataset))
    print("test samples:", len(test_dataset))

    for epoch in range(num_epochs):
        # 训练一轮，得到训练集上的平均 loss 和准确率。
        train_loss, train_accuracy = train_one_epoch()

        # 在测试集上评估当前模型，得到测试 loss 和准确率。
        test_loss, test_accuracy = evaluate()

        # epoch 从 0 开始计数，所以打印时用 epoch + 1 更符合人的阅读习惯。
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"train_loss: {train_loss:.4f} "
            f"train_acc: {train_accuracy * 100:.2f}% "
            f"test_loss: {test_loss:.4f} "
            f"test_acc: {test_accuracy * 100:.2f}%"
        )

    # 训练结束后保存模型参数。
    # state_dict 里保存的是每一层的权重和偏置。
    # 保存 state_dict 比直接保存整个模型对象更常见，也更推荐。
    torch.save(model.state_dict(), model_save_path)
    print("model saved to:", model_save_path)
