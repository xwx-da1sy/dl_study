"""
内容：
    张量的基本运算法则

实际情况：
    直接使用"+"来代表加法直接进行操作
    注意这里的乘法指的是对应位置的元素相乘

"""

import torch

tensor01 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

tensor02 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

tenor03 = tensor01 * tensor02

print(f"张量01: {tensor01}")
print(f"张量02: {tensor02}")
print(f"张量03: {tenor03} 类型是：{tenor03.type()}")

# 这里我想让tensor03变成一个float类型的张量
tenor03 = tenor03.float()
print(f"张量03: {tenor03} 类型是：{tenor03.type()}")

tensor04 = tensor01 + tensor02
print(f"张量04: {tensor04} 类型是：{tensor04.type()}")

tensor05 = tensor01 - tensor02
print(f"张量05: {tensor05} 类型是：{tensor05.type()}")

tensor06 = tensor01 / tensor02
print(f"张量06: {tensor06} 类型是：{tensor06.type()}")


"""
矩阵乘法

点乘：
    直接使用"*"便可以实现
    
矩阵相乘：
    使用"@"来代表矩阵相乘
    要求：两个张量的维度需要保持一致
"""

tensor07 = tensor01 @ tensor02
print(f"张量07: {tensor07} 类型是：{tensor07.type()}")

"""
一些其他的矩阵的运算：

    mean()平均值
    max()最大值
    sum()求和
    min()最小值
    
注意这里是有dim的

    dim = 0：按列操作
    dim = 1：按行操作
    不传：按照张量操作
"""