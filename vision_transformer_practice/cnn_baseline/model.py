"""用于和 TinyViT 进行受控对照实验的 CIFAR-10 CNN。"""

from torch import nn

try:
    from ..vit.config import DROPOUT_RATE, IN_CHANNELS, NUM_CLASSES
except ImportError:
    from vit.config import DROPOUT_RATE, IN_CHANNELS, NUM_CLASSES


class ConvStage(nn.Module):
    """连续使用两次卷积提取特征，再将高宽缩小一半。"""

    def __init__(self, in_channels, out_channels):
        """
        参数：
            in_channels：输入特征图通道数。
            out_channels：当前阶段输出特征图通道数。

        返回值：
            无显式返回值；卷积、归一化、激活和池化层保存在模块中。
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        """
        参数：x，shape 为 B x C_in x H x W。

        返回值：shape 为 B x C_out x (H/2) x (W/2) 的特征图。
        """
        return self.layers(x)


class CNNBaseline(nn.Module):
    """
    CIFAR-10 CNN 公平对照模型。

    输入：B x 3 x 32 x 32。
    输出：B x 10 logits。

    四个卷积阶段的 shape：
        B x 3   x 32 x 32
        B x 64  x 16 x 16
        B x 128 x 8  x 8
        B x 256 x 4  x 4
        B x 512 x 2  x 2

    设计说明：
        - 每阶段包含两个 3 x 3 卷积；
        - BatchNorm 稳定卷积网络训练；
        - GELU 与 TinyViT 使用相同激活函数，减少无关变量；
        - 全局平均池化避免大型全连接层；
        - 参数量约 469 万，与 TinyViT 的约 477 万接近。
    """

    def __init__(
        self,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
    ):
        """
        参数：
            in_channels：输入图片通道数，CIFAR-10 为 3。
            num_classes：分类类别数，CIFAR-10 为 10。
            dropout_rate：分类头前的 Dropout 概率，与 TinyViT 保持 0.1。

        返回值：无显式返回值。
        """
        super().__init__()
        self.stages = nn.Sequential(
            ConvStage(in_channels, 64),
            ConvStage(64, 128),
            ConvStage(128, 256),
            ConvStage(256, 512),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward_features(self, images):
        """
        参数：images，shape 为 B x 3 x 32 x 32。

        返回值：卷积特征图，shape 为 B x 512 x 2 x 2。
        """
        if images.ndim != 4:
            raise ValueError(
                f"images 必须是 B x C x H x W 四维张量，实际为 {images.ndim} 维"
            )
        if images.shape[1:] != (3, 32, 32):
            raise ValueError(
                "CNNBaseline 期望输入为 B x 3 x 32 x 32，"
                f"实际为 {tuple(images.shape)}"
            )
        return self.stages(images)

    def forward(self, images):
        """
        参数：images，shape 为 B x 3 x 32 x 32。

        返回值：10 类原始分数 logits，shape 为 B x 10。
        """
        features = self.forward_features(images)
        pooled_features = self.global_pool(features)
        logits = self.classifier(pooled_features)
        return logits
