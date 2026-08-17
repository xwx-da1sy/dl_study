"""ViT 的 Patch Embedding、CLS token 与位置编码。"""

import torch
from torch import nn

from .config import EMBED_DIM, IMAGE_SIZE, IN_CHANNELS, PATCH_SIZE


class PatchEmbedding(nn.Module):
    """
    作用：
        把 B x 3 x 32 x 32 的彩色图片划分成 8 x 8 的 patch 网格，
        再把每个 4 x 4 patch 映射成一个 192 维 token。

    参数：
        image_size：输入图片的高和宽，当前为 32。
        patch_size：每个 patch 的高和宽，当前为 4。
        in_channels：输入图片通道数，RGB 彩图为 3。
        embed_dim：每个 patch token 的特征维度，当前为 192。

    返回值：
        forward 返回 patch_tokens，shape 为 B x 64 x 192。
        64 来自 8 x 8 个 patches，不包含 CLS token。
    """

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        embed_dim=EMBED_DIM,
    ):
        """
        作用：创建 Patch Embedding 所需的卷积投影层，并记录 patch 数量。

        参数：
            image_size：输入图片尺寸。
            patch_size：单个 patch 尺寸。
            in_channels：输入通道数。
            embed_dim：输出 token 维度。

        返回值：
            无显式返回值；初始化结果保存在当前模块中。
        """
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size 必须能够被 patch_size 整除")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        # 32 / 4 = 8，表示图片的高和宽方向都能切出 8 个 patches。
        self.grid_size = image_size // patch_size

        # 8 x 8 = 64，表示每张图片最终得到 64 个 patch tokens。
        self.num_patches = self.grid_size ** 2

        # nn.Conv2d
        # 作用：一次完成“切分 patches”和“把每个 patch 投影成 token”。
        # 重要参数：
        #   in_channels=3：输入为 RGB 三通道图片。
        #   out_channels=192：每个 patch 输出 192 个特征，即 embed_dim。
        #   kernel_size=4：卷积核每次完整覆盖一个 4 x 4 patch。
        #   stride=4：每次移动 4 个像素，因此相邻 patches 互不重叠。
        # 返回值：这里创建并返回一个 Conv2d 模块；
        #         在 forward 中调用该模块时，输出 B x 192 x 8 x 8 的 feature_map。
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images):
        """
        作用：执行 patch 投影，并把卷积输出整理成 Transformer token 序列。

        参数：
            images：一个 batch 的图片，shape 必须为 B x 3 x 32 x 32。

        返回值：
            patch_tokens：shape 为 B x 64 x 192。
            第二维 64 是 token 数量，第三维 192 是每个 token 的特征维度。
        """
        if images.ndim != 4:
            raise ValueError(
                f"images 必须是 B x C x H x W 四维张量，实际为 {images.ndim} 维"
            )

        _, channels, height, width = images.shape
        if channels != self.in_channels:
            raise ValueError(
                f"期望输入通道数为 {self.in_channels}，实际得到 {channels}"
            )
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"期望图片尺寸为 {self.image_size} x {self.image_size}，"
                f"实际得到 {height} x {width}"
            )

        # projection 返回 B x 192 x 8 x 8。
        feature_map = self.projection(images)

        # flatten(start_dim=2)
        # 作用：把 8 x 8 的 patch 网格展平为 64 个位置。
        # 参数：start_dim=2 表示从第 2 维开始合并后面的所有维度。
        # 返回值：shape 从 B x 192 x 8 x 8 变为 B x 192 x 64。
        patch_tokens = feature_map.flatten(start_dim=2)

        # transpose(1, 2)
        # 作用：交换特征维和 token 维，满足 Transformer 的 B x N x D 格式。
        # 参数：1 和 2 表示交换第 1、2 维。
        # 返回值：shape 从 B x 192 x 64 变为 B x 64 x 192。
        patch_tokens = patch_tokens.transpose(1, 2)

        return patch_tokens


class ViTInputEmbedding(nn.Module):
    """
    作用：
        先把图片转换为 patch tokens，再在序列最前面加入一个可学习的 CLS token。
        最后为 65 个 token 加上各自可学习的位置编码。

    参数：
        image_size：输入图片的高和宽，当前为 32。
        patch_size：每个 patch 的高和宽，当前为 4。
        in_channels：输入图片通道数，RGB 彩图为 3。
        embed_dim：每个 token 的特征维度，当前为 192。

    返回值：
        forward 返回 input_tokens，shape 为 B x 65 x 192。
        其中第 0 个 token 是 CLS token，后面 64 个是 patch tokens。
    """

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        embed_dim=EMBED_DIM,
    ):
        """
        作用：
            创建 PatchEmbedding，并定义可以通过训练学习的 CLS token 和位置编码。

        参数：
            image_size：输入图片尺寸。
            patch_size：单个 patch 尺寸。
            in_channels：输入通道数。
            embed_dim：每个 token 的特征维度。

        返回值：
            无显式返回值；初始化结果保存在当前模块中。
        """
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embedding.num_patches

        # nn.Parameter
        # 作用：把 CLS token 注册为模型的可训练参数，反向传播时会计算它的梯度。
        # 参数：初始张量 shape 为 1 x 1 x 192。
        #       第一个 1 表示先只保存一份共享参数；第二个 1 表示一个 CLS token。
        # 返回值：一个会被优化器自动更新的 Parameter 对象。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # num_patches + 1
        # 作用：计算完整 token 序列的长度。
        # 参数含义：64 个 patch tokens 再加上 1 个 CLS token。
        # 返回值：当前配置下 sequence_length 等于 65。
        self.sequence_length = self.num_patches + 1

        # nn.Parameter
        # 作用：为序列中的每个位置创建一个可学习的位置编码。
        # 参数：初始张量 shape 为 1 x 65 x 192。
        #       第一个 1 表示同一套位置编码由 batch 中所有图片共享；
        #       65 表示 CLS 和 64 个 patches 的位置；192 是特征维度。
        # 返回值：一个会被优化器自动更新的 Parameter 对象。
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.sequence_length, embed_dim)
        )

        # nn.init.trunc_normal_
        # 作用：使用截断正态分布原地初始化 CLS token 和位置编码。
        # 参数：std=0.02 表示标准差为 0.02。
        # 返回值：返回初始化后的参数本身；下划线表示会直接修改原参数。
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, images):
        """
        作用：
            生成 patch tokens，把 CLS token 拼接到最前面，最后加入位置编码。

        参数：
            images：一个 batch 的图片，shape 为 B x 3 x 32 x 32。

        返回值：
            input_tokens：加入 CLS 和位置编码后的序列，shape 为 B x 65 x 192。
        """
        # B x 3 x 32 x 32 -> B x 64 x 192。
        patch_tokens = self.patch_embedding(images)
        batch_size = patch_tokens.shape[0]

        # expand(batch_size, -1, -1)
        # 作用：让 batch 中的每张图片都使用同一份可学习的 CLS token。
        # 参数：batch_size 扩展 batch 维；-1 表示其余维度保持不变。
        # 返回值：shape 从 1 x 1 x 192 变为 B x 1 x 192；不会复制新的独立参数。
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # torch.cat
        # 作用：把 CLS token 放到所有 patch tokens 的最前面。
        # 参数：dim=1 表示沿 token 数量这一维进行拼接。
        # 返回值：shape 从 B x 1 x 192 与 B x 64 x 192 变为 B x 65 x 192。
        tokens_with_cls = torch.cat((cls_tokens, patch_tokens), dim=1)

        # 张量加法
        # 作用：让每个 token 在保留图像特征的同时，获得自己所在位置的信息。
        # 参数：
        #   tokens_with_cls 的 shape 为 B x 65 x 192；
        #   position_embedding 的 shape 为 1 x 65 x 192。
        # 返回值：input_tokens 的 shape 仍为 B x 65 x 192。
        #         PyTorch 会自动广播第 0 维，使 batch 中所有图片共享同一套位置编码。
        input_tokens = tokens_with_cls + self.position_embedding

        return input_tokens
