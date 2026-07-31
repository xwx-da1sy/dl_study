import torch
import torch.nn as nn
import torch.nn.functional as F


# 第二步：定义 MLP 模型。
# 这次我们尝试一个更深、更宽的神经网络。
#
# 原来的结构是：
# 784 -> 128 -> 10
#
# 新的结构是：
# 784 -> 512 -> 256 -> 128 -> 10
#
# 相比原来：
# 1. 多加了两个隐藏层。
# 2. 第一层隐藏层神经元从 128 增加到 512。
# 3. 模型表达能力更强，但训练时间也会稍微增加。


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # MNIST 每张图片大小是 28x28。
        # 展平成一维向量后，输入特征数是 28 * 28 = 784。
        input_size = 28 * 28

        # 三个隐藏层的神经元数量。
        # hidden_size1 越大，第一层能提取的信息越丰富。
        # 后面的隐藏层逐渐变小，相当于逐步压缩和整理特征。
        hidden_size1 = 512
        hidden_size2 = 256
        hidden_size3 = 128

        # MNIST 要识别 0~9，一共有 10 个类别。
        num_classes = 10

        # 第一层全连接：784 -> 512。
        self.fc1 = nn.Linear(input_size, hidden_size1)

        # 第二层全连接：512 -> 256。
        # 这是新增的隐藏层之一。
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)

        # 第三层全连接：256 -> 128。
        # 这是新增的隐藏层之二。
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)

        # 输出层：128 -> 10。
        # 输出 10 个 logits，分别对应数字 0~9。
        self.fc4 = nn.Linear(hidden_size3, num_classes)

    def forward(self, x):
        # x 是一个 batch 的 MNIST 图片。
        # 输入形状通常是 [batch_size, 1, 28, 28]。
        #
        # 全连接层需要二维输入：
        # [batch_size, feature_size]
        #
        # 所以这里先把图片展平成 [batch_size, 784]。
        x = x.reshape(x.size(0), -1)

        # 第一层：784 -> 512，然后经过 ReLU。
        x = self.fc1(x)
        x = F.relu(x)

        # 第二层：512 -> 256，然后经过 ReLU。
        x = self.fc2(x)
        x = F.relu(x)

        # 第三层：256 -> 128，然后经过 ReLU。
        x = self.fc3(x)
        x = F.relu(x)

        # 输出层：128 -> 10。
        # 这里输出 logits，不手动加 softmax。
        # CrossEntropyLoss 内部会处理分类所需的 softmax 逻辑。
        x = self.fc4(x)

        return x


if __name__ == "__main__":
    model = MLP()

    # 随机造一个 batch 的假 MNIST 图片，用来测试模型输入输出形状。
    fake_images = torch.randn(64, 1, 28, 28)
    outputs = model(fake_images)

    print(model)
    print("input shape:", fake_images.shape)
    print("output shape:", outputs.shape)

    # 对 MNIST 十分类来说，输出形状应该是 [64, 10]。
