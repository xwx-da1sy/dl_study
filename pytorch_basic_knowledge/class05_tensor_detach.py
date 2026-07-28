"""
内容：
    使用detach()方法从计算图中分离张量，解决自动微分的弊端

问题：
    一个张量一旦设置为自动微分，该张量不能转化成为ndarray
"""

import torch
import numpy as np

# 创建一个张量
tensor01 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float32)
print(f"张量的值：{tensor01} 类型：{type(tensor01)}")

# 转化成numpy
numpy01 = tensor01.detach().numpy()
print(f"numpy数组的值：{numpy01} 类型：{type(numpy01)}")
