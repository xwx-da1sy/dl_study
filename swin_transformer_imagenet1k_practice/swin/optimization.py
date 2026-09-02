"""为自定义 Swin 创建带参数分组的 AdamW 优化器。"""

import torch

from .config import ADAMW_BETAS, ADAMW_EPS, LEARNING_RATE, WEIGHT_DECAY


def create_optimizer(
    model,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=ADAMW_BETAS,
    eps=ADAMW_EPS,
):
    """创建 AdamW，并只对适合的权重参数使用 Weight Decay。"""
    if learning_rate <= 0:
        raise ValueError("learning_rate 必须大于0")
    if weight_decay < 0:
        raise ValueError("weight_decay 不能为负数")
    if len(betas) != 2 or not all(0.0 <= beta < 1.0 for beta in betas):
        raise ValueError("betas 必须包含两个位于[0, 1)之间的数")
    if eps <= 0:
        raise ValueError("eps 必须大于0")

    decay_parameters = []
    no_decay_parameters = []

    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        # bias 和 LayerNorm 参数通常只有一维，不进行 Weight Decay。
        # 相对位置偏置虽然是二维表，但它表示注意力位置关系，也单独排除。
        should_skip_decay = (
            parameter.ndim == 1
            or parameter_name.endswith(".bias")
            or "relative_position_bias_table" in parameter_name
        )
        if should_skip_decay:
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    if not decay_parameters or not no_decay_parameters:
        raise ValueError("模型参数无法划分为 decay 和 no_decay 两组")

    parameter_groups = [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(
        parameter_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )
