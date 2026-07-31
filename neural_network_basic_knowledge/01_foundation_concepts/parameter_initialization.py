"""
内容：
    参数初始化：Xavier 初始化 和 He（Kaiming）初始化

问题背景：
    神经网络训练前，权重 W 和偏置 b 需要一个初始值。好的初始化要满足三点：
    1. 打破神经元之间的对称性（不能全0或全相等）
    2. 让前向传播时每层输出的尺度不爆炸不消失
    3. 让反向传播时梯度的尺度也尽量稳定

关键概念：
    fan_in  = 当前层的输入特征数（上一层的输出维度）
    fan_out = 当前层的输出神经元数
    卷积层：fan_in = c_in × k_h × k_w, fan_out = c_out × k_h × k_w

关键公式：
    方差近似：Var(z) = n × Var(w) × Var(x)，其中 n = fan_in
    控制方差目标：Var(w) ≈ 1 / fan_in（让输出方差 ≈ 输入方差）

    Xavier：Var(w) = 2 / (fan_in + fan_out)     适合 Tanh / Sigmoid
    He：    Var(w) = 2 / fan_in                适合 ReLU 家族

为什么 He 比 Xavier 更适合 ReLU：
    ReLU 会把大约一半的负数信号截断为 0，信号会变弱。
    He 的方差约为 Xavier 的 2 倍（fan_in ≈ fan_out 时），
    这个放大就是为了补偿 ReLU 截断负半轴造成的信号减少。

选择规则速记：
    S 形激活函数（Tanh/Sigmoid）  →  Xavier
    ReLU 家族（ReLU/LeakyReLU）  →  He / Kaiming
    偏置通常初始化为 0
"""

import torch
from torch import nn


