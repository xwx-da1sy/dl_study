"""
内容：
    对张量的形状进行操作

重要函数：
    reshape(), unsequeeze()
"""

import torch
print("_" * 32)

# 设置随机种子
torch.manual_seed(32)

tensor01 = torch.randint(1, 10, (2, 3))

# 打印行和列
print(f"tensor01的形状是: {tensor01.shape}")
print(f"行数：{tensor01.shape[0]}")

# 重构tensor01
tensor02 = tensor01.reshape(3, 2)
print(f"tensor02的形状是: {tensor02.shape}")

# 新增一个维度
tensor03 = tensor02.unsqueeze(dim=0)
# print(f"tensor03的形状是: {tensor03.shape}")
# print(tensor02)
# print(tensor03)
tensor04 = tensor02.unsqueeze(dim=1)
print(tensor04)
print(f"tensor04的形状是: {tensor04.shape}")