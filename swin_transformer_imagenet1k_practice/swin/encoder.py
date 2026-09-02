"""Swin 的窗口注意力、Block、Patch Merging 与 Stage。"""

import torch
from torch import nn

from .config import (
    ATTENTION_DROPOUT_RATE,
    DROPOUT_RATE,
    MLP_RATIO,
    WINDOW_SIZE,
)


def window_partition(x, window_size):
    """把 B x H x W x C 划分为多个 G x N x C 窗口。"""
    if x.ndim != 4:
        raise ValueError("x 必须是 B x H x W x C 四维张量")

    batch_size, height, width, channels = x.shape
    if height % window_size != 0 or width % window_size != 0:
        raise ValueError("特征图高宽必须能够被 window_size 整除")

    x = x.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(
        -1,
        window_size * window_size,
        channels,
    )
    return windows


def window_reverse(windows, window_size, height, width):
    """把 G x N x C 窗口还原为 B x H x W x C。"""
    if windows.ndim != 3:
        raise ValueError("windows 必须是 G x N x C 三维张量")
    if height % window_size != 0 or width % window_size != 0:
        raise ValueError("特征图高宽必须能够被 window_size 整除")

    windows_per_image = (height // window_size) * (width // window_size)
    if windows.shape[0] % windows_per_image != 0:
        raise ValueError("窗口数量与目标特征图尺寸不匹配")

    batch_size = windows.shape[0] // windows_per_image
    channels = windows.shape[-1]
    x = windows.view(
        batch_size,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(
        batch_size,
        height,
        width,
        channels,
    )
    return x


class DropPath(nn.Module):
    """训练时随机跳过一条残差分支；推理时保持完整网络。"""

    def __init__(self, drop_probability=0.0):
        super().__init__()
        if not 0.0 <= drop_probability < 1.0:
            raise ValueError("drop_probability 必须位于[0, 1)之间")
        self.drop_probability = drop_probability

    def forward(self, x):
        if self.drop_probability == 0.0 or not self.training:
            return x

        keep_probability = 1.0 - self.drop_probability
        # 同一张图片的全部 token 和特征共享一个开关，只丢整条残差分支。
        random_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_probability + torch.rand(
            random_shape,
            dtype=x.dtype,
            device=x.device,
        )
        random_tensor.floor_()
        return x * random_tensor / keep_probability


class MLP(nn.Module):
    """对每个 token 独立执行 Linear 扩维、GELU 和 Linear 降维。"""

    def __init__(self, dim, mlp_ratio=MLP_RATIO, dropout_rate=DROPOUT_RATE):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        if hidden_dim <= 0:
            raise ValueError("MLP 隐藏维度必须大于0")

        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.layers(x)


class WindowAttention(nn.Module):
    """在单个窗口内部完成多头自注意力，并加入相对位置偏置。"""

    def __init__(
        self,
        dim,
        num_heads,
        window_size=WINDOW_SIZE,
        attention_dropout_rate=ATTENTION_DROPOUT_RATE,
        projection_dropout_rate=DROPOUT_RATE,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError("dim 必须能够被 num_heads 整除")
        if window_size <= 0:
            raise ValueError("window_size 必须大于0")

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.tokens_per_window = window_size * window_size

        # 一次 Linear 同时生成 Q、K、V：C -> 3C。
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attention_dropout = nn.Dropout(attention_dropout_rate)
        self.projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(projection_dropout_rate)

        # window_size=4 时，相对位移共有 (2 x 4 - 1)^2 = 49 种。
        relative_position_count = (2 * window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_position_count, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        relative_position_index = self._create_relative_position_index()
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=True,
        )

    def _create_relative_position_index(self):
        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(self.window_size),
                torch.arange(self.window_size),
                indexing="ij",
            )
        )
        coordinates = coordinates.flatten(start_dim=1)

        # 结果的两个 16 维分别对应 Query token 和 Key token。
        relative_coordinates = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates = relative_coordinates.permute(1, 2, 0).contiguous()
        relative_coordinates[:, :, 0] += self.window_size - 1
        relative_coordinates[:, :, 1] += self.window_size - 1
        relative_coordinates[:, :, 0] *= 2 * self.window_size - 1
        return relative_coordinates.sum(dim=-1)

    def _get_relative_position_bias(self):
        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        bias = bias.view(
            self.tokens_per_window,
            self.tokens_per_window,
            self.num_heads,
        )
        # 16 x 16 x h -> 1 x h x 16 x 16，最前面的1会广播到所有窗口。
        return bias.permute(2, 0, 1).unsqueeze(0)

    def forward(self, window_tokens, attention_mask=None):
        if window_tokens.ndim != 3:
            raise ValueError("window_tokens 必须是 G x N x C 三维张量")

        batch_windows, token_count, channels = window_tokens.shape
        if token_count != self.tokens_per_window or channels != self.dim:
            raise ValueError(
                f"期望窗口 shape 为 G x {self.tokens_per_window} x {self.dim}，"
                f"实际为{tuple(window_tokens.shape)}"
            )

        qkv = self.qkv(window_tokens)
        qkv = qkv.reshape(
            batch_windows,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        # 每个 head：Q 为 16 x 32，K^T 为 32 x 16，结果为 16 x 16。
        query = query * self.scale
        attention_logits = query @ key.transpose(-2, -1)
        attention_logits = attention_logits + self._get_relative_position_bias()

        if attention_mask is not None:
            window_count = attention_mask.shape[0]
            if batch_windows % window_count != 0:
                raise ValueError("attention_mask 的窗口数量与输入不匹配")

            batch_size = batch_windows // window_count
            attention_logits = attention_logits.view(
                batch_size,
                window_count,
                self.num_heads,
                token_count,
                token_count,
            )
            attention_logits = attention_logits + attention_mask.to(
                dtype=attention_logits.dtype
            ).unsqueeze(0).unsqueeze(2)
            attention_logits = attention_logits.view(
                batch_windows,
                self.num_heads,
                token_count,
                token_count,
            )

        attention_weights = attention_logits.softmax(dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        output = attention_weights @ value
        output = output.transpose(1, 2).reshape(
            batch_windows,
            token_count,
            channels,
        )
        output = self.projection(output)
        output = self.projection_dropout(output)
        return output


def create_shifted_window_mask(input_resolution, window_size, shift_size):
    """为循环移位后不应相连的 token 对生成 0/-100 mask。"""
    height, width = input_resolution
    image_mask = torch.zeros(1, height, width, 1)

    height_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    width_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )

    region_number = 0
    for height_slice in height_slices:
        for width_slice in width_slices:
            image_mask[:, height_slice, width_slice, :] = region_number
            region_number += 1

    mask_windows = window_partition(image_mask, window_size).squeeze(-1)
    attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attention_mask = attention_mask.masked_fill(
        attention_mask != 0,
        -100.0,
    )
    attention_mask = attention_mask.masked_fill(attention_mask == 0, 0.0)
    return attention_mask


class SwinBlock(nn.Module):
    """Pre-Norm、窗口注意力、残差连接和 MLP 组成的一个 Swin Block。"""

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=WINDOW_SIZE,
        shift_size=0,
        mlp_ratio=MLP_RATIO,
        dropout_rate=DROPOUT_RATE,
        attention_dropout_rate=ATTENTION_DROPOUT_RATE,
        drop_path_rate=0.0,
    ):
        super().__init__()

        if isinstance(input_resolution, int):
            input_resolution = (input_resolution, input_resolution)
        if len(input_resolution) != 2:
            raise ValueError("input_resolution 必须包含高度和宽度")
        if min(input_resolution) < window_size:
            raise ValueError("当前实现要求特征图尺寸不小于窗口尺寸")
        if shift_size < 0 or shift_size >= window_size:
            raise ValueError("shift_size 必须位于0到window_size-1之间")

        self.dim = dim
        self.input_resolution = tuple(input_resolution)
        self.window_size = window_size

        # 特征图只有一个窗口时，移动窗口不会产生跨窗口通信，自动关闭 shift。
        if min(self.input_resolution) == window_size:
            shift_size = 0
        self.shift_size = shift_size

        self.layer_norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            attention_dropout_rate=attention_dropout_rate,
            projection_dropout_rate=dropout_rate,
        )
        self.drop_path = DropPath(drop_path_rate)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            dim=dim,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
        )

        if self.shift_size > 0:
            attention_mask = create_shifted_window_mask(
                input_resolution=self.input_resolution,
                window_size=self.window_size,
                shift_size=self.shift_size,
            )
        else:
            attention_mask = None

        self.register_buffer(
            "attention_mask",
            attention_mask,
            persistent=False,
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("SwinBlock 输入必须是 B x H x W x C")

        _, height, width, channels = x.shape
        if (height, width) != self.input_resolution or channels != self.dim:
            raise ValueError(
                f"期望输入 B x {self.input_resolution[0]} x "
                f"{self.input_resolution[1]} x {self.dim}，实际为{tuple(x.shape)}"
            )

        residual = x
        x = self.layer_norm1(x)

        # SW-MSA 先循环左上移动，W-MSA 的 shift_size 为0，不执行移动。
        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )

        windows = window_partition(x, self.window_size)
        windows = self.attention(windows, self.attention_mask)
        x = window_reverse(
            windows,
            self.window_size,
            height,
            width,
        )

        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )

        x = residual + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.layer_norm2(x)))
        return x


class PatchMerging(nn.Module):
    """把相邻2 x 2个 token 合并，使 H/W 减半、通道从 C 变成2C。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layer_norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("PatchMerging 输入必须是 B x H x W x C")

        _, height, width, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"期望通道数为{self.dim}，实际为{channels}")
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("PatchMerging 要求特征图高宽均为偶数")

        top_left = x[:, 0::2, 0::2, :]
        bottom_left = x[:, 1::2, 0::2, :]
        top_right = x[:, 0::2, 1::2, :]
        bottom_right = x[:, 1::2, 1::2, :]
        x = torch.cat(
            (top_left, bottom_left, top_right, bottom_right),
            dim=-1,
        )

        # B x H/2 x W/2 x 4C -> B x H/2 x W/2 x 2C。
        x = self.layer_norm(x)
        x = self.reduction(x)
        return x


class SwinStage(nn.Module):
    """顺序堆叠多个保持 shape 不变的 Swin Block。"""

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size=WINDOW_SIZE,
        use_shifted_windows=True,
        mlp_ratio=MLP_RATIO,
        dropout_rate=DROPOUT_RATE,
        attention_dropout_rate=ATTENTION_DROPOUT_RATE,
        drop_path_rates=None,
    ):
        super().__init__()

        if depth <= 0:
            raise ValueError("Stage depth 必须大于0")
        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth
        if len(drop_path_rates) != depth:
            raise ValueError("drop_path_rates 数量必须与 depth 一致")

        blocks = []
        for block_index in range(depth):
            # 开启移动窗口的 Stage 中，偶数 Block 用 W-MSA，奇数 Block 用 SW-MSA。
            # 只有一个窗口的最后一个 Stage 会自动关闭无意义的 shift。
            use_shift = use_shifted_windows and block_index % 2 == 1
            shift_size = window_size // 2 if use_shift else 0
            blocks.append(
                SwinBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    drop_path_rate=drop_path_rates[block_index],
                )
            )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
