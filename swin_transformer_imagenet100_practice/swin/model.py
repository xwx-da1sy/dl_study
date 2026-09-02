"""四阶段自定义 Swin、轻量 Attention Pooling 与分类输出。"""

import torch
from torch import nn

from .config import (
    ATTENTION_DROPOUT_RATE,
    DROPOUT_RATE,
    EMBED_DIM,
    IMAGE_SIZE,
    IN_CHANNELS,
    MLP_RATIO,
    NUM_CLASSES,
    PATCH_SIZE,
    STAGE_DEPTHS,
    STAGE_DIMS,
    STAGE_NUM_HEADS,
    STAGE_USE_SHIFTED_WINDOWS,
    STOCHASTIC_DEPTH_RATE,
    WINDOW_SIZE,
)
from .embedding import PatchEmbedding
from .encoder import PatchMerging, SwinStage


class AttentionPooling(nn.Module):
    """学习每个 token 的重要性，再把 N 个 tokens 加权汇总为一个向量。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.score = nn.Linear(dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        # 初始分数全部为0，Softmax 后每个 token 权重均为1/N。
        # 因此训练开始时等价于平均池化，之后再学习不同位置的重要性。
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, tokens):
        if tokens.ndim != 3:
            raise ValueError("AttentionPooling 输入必须是 B x N x C")
        if tokens.shape[-1] != self.dim:
            raise ValueError(f"期望 token 维度为{self.dim}，实际为{tokens.shape[-1]}")

        token_scores = self.score(tokens)
        token_weights = token_scores.softmax(dim=1)
        pooled_features = (tokens * token_weights).sum(dim=1)
        return pooled_features, token_weights.squeeze(-1)


class ClassificationHead(nn.Module):
    """把整图特征从768维映射为 ImageNet-100 的100个 logits。"""

    def __init__(self, dim, num_classes=NUM_CLASSES):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes 必须大于0")
        self.dim = dim
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != self.dim:
            raise ValueError(f"期望特征 shape 为 B x {self.dim}")
        return self.classifier(features)


class CustomSwin(nn.Module):
    """针对 224 x 224 ImageNet-100 图片设计的四阶段 Swin 分类网络。"""

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        embed_dim=EMBED_DIM,
        stage_depths=STAGE_DEPTHS,
        stage_dims=STAGE_DIMS,
        stage_num_heads=STAGE_NUM_HEADS,
        stage_use_shifted_windows=STAGE_USE_SHIFTED_WINDOWS,
        window_size=WINDOW_SIZE,
        mlp_ratio=MLP_RATIO,
        dropout_rate=DROPOUT_RATE,
        attention_dropout_rate=ATTENTION_DROPOUT_RATE,
        stochastic_depth_rate=STOCHASTIC_DEPTH_RATE,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()

        stage_count = 4
        if not (
            len(stage_depths)
            == len(stage_dims)
            == len(stage_num_heads)
            == len(stage_use_shifted_windows)
            == stage_count
        ):
            raise ValueError("当前网络要求四个 Stage 的配置长度都为4")
        if stage_dims[0] != embed_dim:
            raise ValueError("Stage 1 通道数必须等于 Patch Embedding 维度")
        if any(
            next_dim != 2 * current_dim
            for current_dim, next_dim in zip(stage_dims, stage_dims[1:])
        ):
            raise ValueError("每次 Patch Merging 后的 Stage 通道数必须变成2倍")
        if any(dim % heads != 0 for dim, heads in zip(stage_dims, stage_num_heads)):
            raise ValueError("每个 Stage 的通道数必须能够被注意力头数整除")

        patch_grid_size = image_size // patch_size
        stage_resolutions = (
            patch_grid_size,
            patch_grid_size // 2,
            patch_grid_size // 4,
            patch_grid_size // 8,
        )
        if any(resolution < window_size for resolution in stage_resolutions):
            raise ValueError("每个 Stage 的分辨率都不能小于窗口尺寸")
        if any(resolution % window_size != 0 for resolution in stage_resolutions):
            raise ValueError("每个 Stage 的分辨率都必须能够被窗口尺寸整除")

        self.stage_resolutions = stage_resolutions
        self.stage_dims = tuple(stage_dims)

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        total_blocks = sum(stage_depths)
        drop_path_rates = torch.linspace(
            0.0,
            stochastic_depth_rate,
            total_blocks,
        ).tolist()
        rate_offset = 0

        self.stage1 = SwinStage(
            dim=stage_dims[0],
            input_resolution=stage_resolutions[0],
            depth=stage_depths[0],
            num_heads=stage_num_heads[0],
            window_size=window_size,
            use_shifted_windows=stage_use_shifted_windows[0],
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            drop_path_rates=drop_path_rates[rate_offset : rate_offset + stage_depths[0]],
        )
        rate_offset += stage_depths[0]

        self.patch_merging1 = PatchMerging(stage_dims[0])
        self.stage2 = SwinStage(
            dim=stage_dims[1],
            input_resolution=stage_resolutions[1],
            depth=stage_depths[1],
            num_heads=stage_num_heads[1],
            window_size=window_size,
            use_shifted_windows=stage_use_shifted_windows[1],
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            drop_path_rates=drop_path_rates[rate_offset : rate_offset + stage_depths[1]],
        )
        rate_offset += stage_depths[1]

        self.patch_merging2 = PatchMerging(stage_dims[1])
        self.stage3 = SwinStage(
            dim=stage_dims[2],
            input_resolution=stage_resolutions[2],
            depth=stage_depths[2],
            num_heads=stage_num_heads[2],
            window_size=window_size,
            use_shifted_windows=stage_use_shifted_windows[2],
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            drop_path_rates=drop_path_rates[rate_offset : rate_offset + stage_depths[2]],
        )
        rate_offset += stage_depths[2]

        self.patch_merging3 = PatchMerging(stage_dims[2])
        self.stage4 = SwinStage(
            dim=stage_dims[3],
            input_resolution=stage_resolutions[3],
            depth=stage_depths[3],
            num_heads=stage_num_heads[3],
            window_size=window_size,
            use_shifted_windows=stage_use_shifted_windows[3],
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            drop_path_rates=drop_path_rates[rate_offset : rate_offset + stage_depths[3]],
        )

        self.final_layer_norm = nn.LayerNorm(stage_dims[-1])
        self.attention_pooling = AttentionPooling(stage_dims[-1])
        self.classification_head = ClassificationHead(
            dim=stage_dims[-1],
            num_classes=num_classes,
        )

        self.apply(self._initialize_module)
        # apply 会初始化所有 Linear，因此最后重新恢复 Pooling 的零初始化设计。
        self.attention_pooling.reset_parameters()

    @staticmethod
    def _initialize_module(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, images):
        # B x 3 x 224 x 224 -> B x 56 x 56 x 96。
        x = self.patch_embedding(images)
        x = self.stage1(x)

        # B x 56 x 56 x 96 -> B x 28 x 28 x 192。
        x = self.patch_merging1(x)
        x = self.stage2(x)

        # B x 28 x 28 x 192 -> B x 14 x 14 x 384。
        x = self.patch_merging2(x)
        x = self.stage3(x)

        # B x 14 x 14 x 384 -> B x 7 x 7 x 768。
        x = self.patch_merging3(x)
        x = self.stage4(x)
        x = self.final_layer_norm(x)

        # B x 7 x 7 x 768 -> B x 49 x 768，继续使用原项目的 Attention Pooling。
        tokens = x.flatten(start_dim=1, end_dim=2)
        pooled_features, pooling_weights = self.attention_pooling(tokens)
        return pooled_features, pooling_weights

    def forward(self, images, return_pooling_weights=False):
        pooled_features, pooling_weights = self.forward_features(images)
        logits = self.classification_head(pooled_features)

        if return_pooling_weights:
            return logits, pooling_weights
        return logits
