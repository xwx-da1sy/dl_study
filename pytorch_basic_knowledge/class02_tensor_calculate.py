"""
内容：
    张量的基本运算法则

实际情况：
    直接使用"+"来代表加法直接进行操作

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