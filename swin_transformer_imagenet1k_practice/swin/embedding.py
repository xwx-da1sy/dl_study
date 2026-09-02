"""把 ImageNet-1K 图片转换为保留二维网格的 Patch tokens。"""

from torch import nn

from .config import EMBED_DIM, IMAGE_SIZE, IN_CHANNELS, PATCH_SIZE


class PatchEmbedding(nn.Module):
    """使用不重叠卷积把 B x 3 x 224 x 224 转为 B x 56 x 56 x 96。"""

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        embed_dim=EMBED_DIM,
    ):
        super().__init__()

        if image_size <= 0 or patch_size <= 0:
            raise ValueError("image_size 和 patch_size 必须大于0")
        if image_size % patch_size != 0:
            raise ValueError("image_size 必须能够被 patch_size 整除")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.grid_size = image_size // patch_size

        # kernel_size=stride=4：每个 4 x 4 区域成为一个互不重叠的 Patch。
        # Conv2d 输出 B x 96 x 56 x 56。
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Swin 内部使用 B x H x W x C，LayerNorm 因而直接处理最后的 C 维。
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, images):
        if images.ndim != 4:
            raise ValueError(
                f"images 必须是 B x C x H x W 四维张量，实际为{images.ndim}维"
            )

        _, channels, height, width = images.shape
        if channels != self.in_channels:
            raise ValueError(
                f"期望{self.in_channels}个输入通道，实际得到{channels}个"
            )
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"期望图片尺寸为{self.image_size} x {self.image_size}，"
                f"实际得到{height} x {width}"
            )

        feature_map = self.projection(images)

        # B x C x H x W -> B x H x W x C。
        patch_tokens = feature_map.permute(0, 2, 3, 1).contiguous()
        patch_tokens = self.layer_norm(patch_tokens)
        return patch_tokens
