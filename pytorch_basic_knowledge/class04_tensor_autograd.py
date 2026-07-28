"""
内容：
    使用pytorch的autograd功能实现自动求导

关键函数：
    backward()反向传播
    forward()前向传播

注意：
    只有标量张量可以进行求导
    requires_grad = True才可以执行求导
"""

# 导包
import torch

# 创建一个4行5列的张量
weight = torch.tensor(10,requires_grad=True, dtype=torch.float32)
print(weight)

# 定义损失函数
loss = 2 * weight ** 2

# 打印梯度函数类型
print(type(loss.grad_fn))

# 计算梯度
# 用sum()来保证loss是一个标量
# 计算完毕后会记录到.grad这个属性当中去
loss.sum().backward()

#定义学习率
learning_rate = 0.01

# 带入权重更新公式
new_weight = weight.data - learning_rate * weight.grad
print(new_weight)
print(weight.grad)

print("_" * 32)

"""
内容：
    尝试让pytorch进行循环自动求导

关键函数：
    weight.grad.zero_()用来清空梯度放置梯度自动叠加
"""

# 定义张量
my_weight = torch.tensor(10, requires_grad=True, dtype=torch.float32)
# 学习率
learning_rate = 0.01

# 开始循环更新500次,求最优解
for i in range(1, 501):
    # 在循环内重新计算损失，否则计算图会断开
    my_loss = 2 * my_weight ** 3
    my_loss.sum().backward()
    my_weight.data = my_weight.data - learning_rate * my_weight.grad

    print(f"第{i}次更新后，权重的值：{my_weight.data}，损失的值：{my_loss.data}，梯度：{my_weight.grad}")

    # 让旧的梯度清零，防止梯度自动叠加
    my_weight.grad.zero_()