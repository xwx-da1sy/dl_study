"""
张量：
    存储同一类型元素的容器，这些元素都必须是数值.

张量创建的基本方式：
    1.torch.tensor: 根据指定数据创建张量
    2.torch.Tensor: 根据形状创建张量
    3.torch.IntTensor: 创建整型张量
    4.torch.FloatTensor: 创建单精度浮点型张量
    5.torch.DoubleTensor: 创建双精度浮点型张量

实际情况：
    只用torch.tensor
"""

import torch

def tensor_create_basic():
    # torch.tensor(data, dtype=None): 根据指定数据创建张量
    #   data(要转换的数据，如数值/列表/ndarray), dtype(张量数据类型)
    tensor01 = torch.tensor(10)
    print(f"{tensor01}， type: {type(tensor01)}")

    data01 = [1, 2, 3, 4, 5]
    tensor02 = torch.tensor(data01)
    print(f"{tensor02}， type: {type(tensor02)}")

    print("-------------------------")


# 如何创建全是0或者全是1的张量
# torch.zeros或者torch.zeros_like
def tensor_create_all():
    # torch.zeros(*size): 创建全0张量
    #   size(几行几列，可变参数)
    tensor01 = torch.zeros(3, 4)
    print(f"{tensor01}， type: {type(tensor01)}")

    # torch.ones(*size): 创建全1张量
    #   size(几行几列，可变参数)
    tensor02 = torch.ones(3, 4)
    print(f"{tensor02}， type: {type(tensor02)}")

    # torch.full(size, fill_value): 创建全为一个值的张量
    #   size(形状元组), fill_value(填充值)
    tensor03 = torch.full(size=(3, 3), fill_value=255)
    print(f"{tensor03}， type: {type(tensor03)}")

    print("-------------------------")

# 创建线性和随机张量
def tensor_create_linear():
    # torch.arange(start, end, step): 像Python range，左闭右开
    #   start(起始值), end(终止值), step(步长)
    tensor01 = torch.arange(1, 10, 2)  # [1, 3, 5, 7, 9]
    print(f"{tensor01}， type: {type(tensor01)}")

    # torch.linspace(start, end, steps): 在区间内等间隔取点，包含两端
    #   start(起始值), end(终止值), steps(分成的元素个数)
    tensor02 = torch.linspace(1, 10, 4)  # [1.0, 4.0, 7.0, 10.0]
    print(f"{tensor02}， type: {type(tensor02)}")

    print("-------------------------")

# 创建随机张量
def tensor_create_random():

    # torch.manual_seed(seed): 固定随机种子，让实验可复现
    #   seed(随机种子值)
    torch.manual_seed(1)

    # torch.rand(size): [0, 1)均匀分布随机数
    #   size(几行几列)
    tensor01 = torch.rand(3, 4)  # 3行4列随机张量
    print(f"{tensor01}， type: {type(tensor01)}")

    # torch.randn(size): 标准正态分布随机数 N(0, 1)
    #   size(几行几列)
    tensor02 = torch.randn(3, 4)
    print(f"{tensor02}， type: {type(tensor02)}")

    # torch.randint(low, high, size): 随机整数，左闭右开
    #   low(下限), high(上限), size(几行几列)
    tensor03 = torch.randint(0, 10, (3, 3))  # [0, 10)随机整数
    print(f"{tensor03}， type: {type(tensor03)}")

if __name__ == '__main__':
    tensor_create_basic()
    tensor_create_all()