# 第一步：导包
import torch
import torch.nn as nn
import torch.nn.functional as F


# 第二步：定义 MLP 模型。
# 这一部分先只写导包。
# 后面我们会继续写 class MLP(nn.Module)，用它把 28x28 的 MNIST 图片分类成 0~9。
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # MNIST 每张图片大小是 28x28。
        # MLP 不直接处理二维图片，而是先把图片展平成一维向量。
        # 所以输入特征数是 28 * 28 = 784。
        input_size = 28 * 28

        # hidden_size 表示隐藏层神经元个数。
        # 这个值不是固定答案，可以调整。
        # 128 对 MNIST 这种入门任务来说已经够用。
        hidden_size = 128

        # MNIST 要识别 0~9，一共有 10 个类别。
        # 所以最后输出层需要输出 10 个分数。
        num_classes = 10

        # 第一个全连接层：把 784 维输入映射到 128 维隐藏表示。
        self.fc1 = nn.Linear(input_size, hidden_size)

        # 第二个全连接层：把 128 维隐藏表示映射到 10 个类别分数。
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x 是一个 batch 的 MNIST 图片。
        # 进入模型时，x 的形状通常是：
        # [batch_size, 1, 28, 28]
        #
        # nn.Linear 需要二维输入：
        # [batch_size, feature_size]
        #
        # x.size(0) 表示 batch_size。
        # -1 表示让 PyTorch 自动计算剩下的维度，也就是 1 * 28 * 28 = 784。
        x = x.reshape(x.size(0), -1)

        # 第一层全连接，把 784 维输入变成 128 维隐藏表示。
        x = self.fc1(x)

        # ReLU 激活函数负责加入非线性能力。
        # 如果没有激活函数，多层线性层叠在一起本质上仍然只是一个线性变换。
        x = F.relu(x)

        # 第二层全连接，把 128 维隐藏表示变成 10 个类别分数。
        # 这里输出的是 logits，也就是原始分数，不需要手动 softmax。
        # 后面使用 CrossEntropyLoss 时，它内部会处理对应的 softmax 逻辑。
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # 创建一个 MLP 模型对象。
    model = MLP()

    # 随机造一个 batch 的假 MNIST 图片。
    # 64 表示 batch_size。
    # 1 表示灰度通道。
    # 28 和 28 表示图片高度和宽度。
    fake_images = torch.randn(64, 1, 28, 28)

    # 调用 model(fake_images) 时，PyTorch 会自动执行 forward(fake_images)。
    outputs = model(fake_images)

    print(model)
    print("input shape:", fake_images.shape)
    print("output shape:", outputs.shape)

    # 对 MNIST 十分类来说，输出形状应该是 [64, 10]。
    # 64 表示每张图片都有一个预测结果。
    # 10 表示每张图片都会得到 10 个类别分数。
