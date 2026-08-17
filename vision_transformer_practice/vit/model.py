"""完整 Tiny ViT 模型与分类输出。"""

from torch import nn

from .config import (
    DROPOUT_RATE,
    EMBED_DIM,
    IMAGE_SIZE,
    IN_CHANNELS,
    MLP_HIDDEN_DIM,
    NUM_CLASSES,
    NUM_ENCODER_BLOCKS,
    NUM_HEADS,
    PATCH_SIZE,
)
from .embedding import ViTInputEmbedding
from .encoder import TransformerEncoder


def extract_cls_token(encoder_tokens):
    """
    作用：
        从 Transformer Encoder 输出的完整序列中取出第 0 个 CLS token，
        作为当前 batch 中每张图片的全局特征。

    参数：
        encoder_tokens：经过最终 LayerNorm 的 token 序列，
                        shape 为 B x 65 x 192。

    返回值：
        cls_features：每张图片对应的 CLS 特征，shape 为 B x 192。
    """
    if encoder_tokens.ndim != 3:
        raise ValueError(
            "encoder_tokens 必须是 B x N x D 三维张量，"
            f"实际为 {encoder_tokens.ndim} 维"
        )
    if encoder_tokens.shape[1] < 1:
        raise ValueError("token 序列中至少需要包含一个 CLS token")

    # encoder_tokens[:, 0, :]
    # 第一个冒号：取出 batch 中的所有图片。
    # 0：只取 token 序列的第 0 个位置，也就是经过 Encoder 更新后的 CLS token。
    # 最后一个冒号：取出 CLS token 的全部 192 个特征。
    # shape：B x 65 x 192 -> B x 192。
    cls_features = encoder_tokens[:, 0, :]

    return cls_features


class ClassificationHead(nn.Module):
    """
    作用：
        把每张图片的 192 维 CLS 特征映射为 CIFAR-10 的 10 个类别分数。

    参数：
        embed_dim：CLS token 的特征维度，当前为 192。
        num_classes：分类类别数，CIFAR-10 为 10。

    返回值：
        forward 返回 logits，shape 为 B x 10。
        logits 是未经 Softmax 的原始类别分数。
    """

    def __init__(self, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES):
        """
        作用：创建从 CLS 特征到类别分数的全连接层。

        参数：
            embed_dim：输入特征数量，当前为 192。
            num_classes：输出类别数量，当前为 10。

        返回值：
            无显式返回值；初始化结果保存在当前模块中。
        """
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes 必须大于 0")

        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # nn.Linear
        # 作用：为每张图片计算 10 个类别的原始分数。
        # 参数：in_features=192，out_features=10；bias 默认开启。
        # 返回值：这里创建 Linear 模块；调用时 shape 从 B x 192 变为 B x 10。
        self.classifier = nn.Linear(
            in_features=embed_dim,
            out_features=num_classes,
        )

    def forward(self, cls_features):
        """
        作用：把 CLS 特征转换为最终分类 logits。

        参数：
            cls_features：每张图片的 CLS 特征，shape 为 B x 192。

        返回值：
            logits：每张图片对应的 10 个原始类别分数，shape 为 B x 10。
        """
        if cls_features.ndim != 2:
            raise ValueError(
                "cls_features 必须是 B x D 二维张量，"
                f"实际为 {cls_features.ndim} 维"
            )
        if cls_features.shape[-1] != self.embed_dim:
            raise ValueError(
                f"期望 CLS 特征维度为 {self.embed_dim}，"
                f"实际为 {cls_features.shape[-1]}"
            )

        # B x 192 -> B x 10。
        # 这里不使用 Softmax，后续 CrossEntropyLoss 直接接收原始 logits。
        logits = self.classifier(cls_features)

        return logits


class TinyViT(nn.Module):
    """
    作用：
        把输入嵌入、多层 Transformer Encoder、CLS 提取和分类头
        组合成一个可以直接接收图片并输出 logits 的完整 Tiny ViT。

    参数：
        image_size：输入图片尺寸，当前为 32。
        patch_size：patch 尺寸，当前为 4。
        in_channels：输入图片通道数，当前为 3。
        embed_dim：token 特征维度，当前为 192。
        num_heads：注意力头数，当前为 3。
        hidden_dim：MLP 隐藏层维度，当前为 768。
        num_blocks：Encoder Block 数量，当前为 4。
        dropout_rate：Dropout 概率，当前为 0.1。
        num_classes：分类类别数，CIFAR-10 为 10。

    返回值：
        forward 返回 logits，shape 为 B x 10。
    """

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=MLP_HIDDEN_DIM,
        num_blocks=NUM_ENCODER_BLOCKS,
        dropout_rate=DROPOUT_RATE,
        num_classes=NUM_CLASSES,
    ):
        """
        作用：创建 Tiny ViT 的输入层、Encoder 和分类输出层。

        参数：各参数分别控制图片、token、Encoder 和分类任务的规模。

        返回值：
            无显式返回值；所有子模块均注册到当前模型中，
            因而 model.parameters() 可以取得完整模型的可训练参数。
        """
        super().__init__()

        self.input_embedding = ViTInputEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.encoder = TransformerEncoder(
            num_blocks=num_blocks,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )
        self.classification_head = ClassificationHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
        )

    def forward(self, images):
        """
        作用：完成图片从 Patch Embedding 到10维分类 logits 的完整前向传播。

        参数：
            images：一个 batch 的 CIFAR-10 图片，shape 为 B x 3 x 32 x 32。

        返回值：
            logits：每张图片的10个原始类别分数，shape 为 B x 10。
        """
        input_tokens = self.input_embedding(images)
        encoder_tokens = self.encoder(input_tokens)
        cls_features = extract_cls_token(encoder_tokens)
        logits = self.classification_head(cls_features)

        return logits
