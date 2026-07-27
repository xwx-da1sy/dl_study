"""
重要内容：
    简单行列索引
    范围索引
    多维索引
"""

# 导包
import torch

# 设置随机种子
torch.manual_seed(32)

tensor = torch.randint(1, 10, (5,5))
print(tensor)
print("_" * 32)

# 简单行列索引
print(tensor[1])
#所有行的第0列
print(tensor[:, 0])
# 注意：
#   第一个[]表示的是所有的行，第二个[]表示所有的列
#   所以这里的意思是：第一行第二列的元素以及第三行第四列的元素
print(tensor[[1, 3],[2, 4]])

print("_" * 32)

# 范围索引
print(tensor[0:2,1:3])

print(tensor[[1, 3], 0::2])

print("_" * 32)
# 多维索引
tensor02 = torch.randint(1, 10, (2, 3, 5))
print(tensor02)