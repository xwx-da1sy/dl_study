"""
张量：
    存储同一类型元素的容器，这些元素都必须是数值.

张量创建的基本方式：
    1.torch.tensor: 根据指定数据创建张量
    2.torch.Tensor: 根据已有数据或张量类型创建张量
    3.torch.IntTensor: 创建整型张量
    4.torch.FloatTensor: 创建单精度浮点型张量
    5.torch.DoubleTensor: 创建双精度浮点型张量
"""

import torch

def tensor_creat01():
    tensor01 = torch.tensor(10)
    print(f"{tensor01}")

if __name__ == '__main__':
    tensor_creat01()