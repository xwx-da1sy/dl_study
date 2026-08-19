"""Transformer Encoder 的两个子层、单个 Block 与多层堆叠。"""

from torch import nn

from .config import (
    DROPOUT_RATE,
    EMBED_DIM,
    MLP_HIDDEN_DIM,
    NUM_ENCODER_BLOCKS,
    NUM_HEADS,
)


class AttentionSubLayer(nn.Module):
    """
    作用：
        实现 Transformer Encoder 中的注意力子层，执行：
        x -> LN1 -> MHA -> Dropout -> 与原始 x 相加。

        最后一步使用的是 ResNet 提出的残差连接思想，
        但这里并没有引入卷积版 ResNet 模型。

    参数：
        embed_dim：每个 token 的特征维度，当前为 192。
        num_heads：多头注意力的头数，当前为 3。
        dropout_rate：Dropout 概率，当前为 0.1。

    返回值：
        forward 返回 attention_tokens，shape 与输入相同，均为 B x 65 x 192。
    """

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
    ):
        """
        作用：创建 LN1、多头自注意力层和 Dropout 层。

        参数：
            embed_dim：token 的特征维度。
            num_heads：注意力头数量，embed_dim 必须能被 num_heads 整除。
            dropout_rate：训练时随机丢弃元素的概率。

        返回值：
            无显式返回值；初始化结果保存在当前模块中。
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能够被 num_heads 整除")

        # nn.LayerNorm
        # 作用：对每个 token 的 192 个特征进行归一化，使注意力计算更稳定。
        # 参数：normalized_shape=192，表示只对最后一个特征维进行归一化。
        # 返回值：这里创建 LayerNorm 模块；调用时 shape 保持 B x 65 x 192。
        self.layer_norm1 = nn.LayerNorm(normalized_shape=embed_dim)

        # nn.MultiheadAttention
        # 作用：让每个 token 根据内容关注序列中的其他 token。
        # 参数：
        #   embed_dim=192：每个 token 的总特征维度；
        #   num_heads=3：并行使用 3 个注意力头，每个头处理 64 维；
        #   dropout=0.1：训练时对注意力权重使用 Dropout；
        #   batch_first=True：输入和输出都采用 B x N x D 格式。
        # 返回值：这里创建 MultiheadAttention 模块；调用时返回
        #         (attention_output, attention_weights)。
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        # nn.Dropout
        # 作用：对 MHA 输出再次使用 Dropout，降低过拟合风险。
        # 参数：p=0.1，表示训练时每个元素有 10% 的概率被置为 0。
        # 返回值：这里创建 Dropout 模块；调用时 shape 不发生变化。
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """
        作用：依次完成 LN1、MHA、Dropout 和残差相加。

        参数：
            x：加入 CLS 和位置编码后的 token 序列，shape 为 B x 65 x 192。

        返回值：
            attention_tokens：注意力子层输出，shape 仍为 B x 65 x 192。
        """
        if x.ndim != 3:
            raise ValueError(
                f"x 必须是 B x N x D 三维张量，实际为 {x.ndim} 维"
            )

        # 保存未经注意力处理的原始 x，供最后的残差连接使用。
        residual = x

        # 第 1 步：LN1，B x 65 x 192 -> B x 65 x 192。
        x = self.layer_norm1(x)

        # 第 2 步：MHA。
        # query、key、value 都使用同一个 x，因此这是多头“自注意力”。
        # need_weights=False 表示当前阶段不返回注意力权重，可减少额外计算和内存。
        # 返回的 x 仍为 B x 65 x 192；第二个返回值在此处为 None。
        x, _ = self.multi_head_attention(
            query=x,
            key=x,
            value=x,
            need_weights=False,
        )

        # 第 3 步：Dropout，shape 保持 B x 65 x 192。
        x = self.dropout(x)

        # 第 4 步：残差连接（Residual Connection）。
        # 作用：将注意力结果与原始输入 x 相加，让信息和梯度更容易向深层传播。
        # 两个张量的 shape 都是 B x 65 x 192，因此可以逐元素相加。
        attention_tokens = residual + x

        return attention_tokens


class MLPSubLayer(nn.Module):
    """
    作用：
        实现 Transformer Encoder 中的 MLP 子层，执行：
        x -> LN2 -> Linear 扩维 -> GELU -> Dropout
          -> Linear 降维 -> Dropout -> 与原始 x 相加。

        MLP 会独立处理每一个 token 的特征，不会改变 token 的数量。

    参数：
        embed_dim：每个 token 的输入和输出维度，当前为 192。
        hidden_dim：MLP 隐藏层维度，当前为 768。
        dropout_rate：Dropout 概率，当前为 0.1。

    返回值：
        forward 返回 encoder_tokens，shape 与输入相同，均为 B x 65 x 192。
    """

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        hidden_dim=MLP_HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
    ):
        """
        作用：创建 LN2、两层全连接层、GELU 激活函数和 Dropout 层。

        参数：
            embed_dim：token 的输入和输出特征维度。
            hidden_dim：第一层全连接扩展后的特征维度。
            dropout_rate：训练时随机丢弃元素的概率。

        返回值：
            无显式返回值；初始化结果保存在当前模块中。
        """
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError("hidden_dim 必须大于 0")

        self.embed_dim = embed_dim

        # nn.LayerNorm
        # 作用：对每个 token 的 192 个特征进行归一化，使 MLP 训练更稳定。
        # 参数：normalized_shape=192，表示只对最后一个特征维进行归一化。
        # 返回值：这里创建 LayerNorm 模块；调用时 shape 保持 B x 65 x 192。
        self.layer_norm2 = nn.LayerNorm(normalized_shape=embed_dim)

        # nn.Linear
        # 作用：把每个 token 的特征从 192 维扩展到 768 维。
        # 参数：in_features=192，out_features=768。
        # 返回值：这里创建全连接层；调用时 shape 从 B x 65 x 192
        #         变为 B x 65 x 768。
        self.fc1 = nn.Linear(
            in_features=embed_dim,
            out_features=hidden_dim,
        )

        # nn.GELU
        # 作用：为 MLP 加入非线性表达能力；ViT 通常使用 GELU 而不是 ReLU。
        # 参数：无必填参数。
        # 返回值：这里创建 GELU 模块；调用时 shape 保持 B x 65 x 768。
        self.activation = nn.GELU()

        # 第一处 Dropout 位于 GELU 之后，shape 保持 B x 65 x 768。
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # nn.Linear
        # 作用：把特征从 768 维降回 192 维，以便和残差分支逐元素相加。
        # 参数：in_features=768，out_features=192。
        # 返回值：这里创建全连接层；调用时 shape 从 B x 65 x 768
        #         变回 B x 65 x 192。
        self.fc2 = nn.Linear(
            in_features=hidden_dim,
            out_features=embed_dim,
        )

        # 第二处 Dropout 位于第二层全连接之后，shape 保持 B x 65 x 192。
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """
        作用：依次完成 LN2、MLP、Dropout 和残差相加。

        参数：
            x：注意力子层输出的 token 序列，shape 为 B x 65 x 192。

        返回值：
            encoder_tokens：MLP 子层输出，shape 仍为 B x 65 x 192。
        """
        if x.ndim != 3:
            raise ValueError(
                f"x 必须是 B x N x D 三维张量，实际为 {x.ndim} 维"
            )
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"期望 token 特征维度为 {self.embed_dim}，实际为 {x.shape[-1]}"
            )

        # 保存 MLP 子层的原始输入，供最后的残差连接使用。
        residual = x

        # 第 1 步：LN2，B x 65 x 192 -> B x 65 x 192。
        x = self.layer_norm2(x)

        # 第 2 步：Linear 扩维，B x 65 x 192 -> B x 65 x 768。
        x = self.fc1(x)

        # 第 3 步：GELU 非线性激活，shape 保持 B x 65 x 768。
        x = self.activation(x)

        # 第 4 步：第一次 Dropout，shape 保持 B x 65 x 768。
        x = self.dropout1(x)

        # 第 5 步：Linear 降维，B x 65 x 768 -> B x 65 x 192。
        x = self.fc2(x)

        # 第 6 步：第二次 Dropout，shape 保持 B x 65 x 192。
        x = self.dropout2(x)

        # 第 7 步：残差连接。
        # 注意力子层的输出 residual 与 MLP 输出的 shape 相同，可以逐元素相加。
        encoder_tokens = residual + x

        return encoder_tokens


class EncoderBlock(nn.Module):
    """
    作用：
        把注意力子层和 MLP 子层组合成一个完整的 Transformer Encoder Block。

        执行顺序为：
        输入 -> AttentionSubLayer -> MLPSubLayer -> 输出。

    参数：
        embed_dim：每个 token 的特征维度，当前为 192。
        num_heads：多头注意力的头数，当前为 3。
        hidden_dim：MLP 隐藏层维度，当前为 768。
        dropout_rate：Dropout 概率，当前为 0.1。

    返回值：
        forward 返回 block_tokens，shape 与输入相同，均为 B x 65 x 192。
    """

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=MLP_HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
    ):
        """
        作用：创建一个注意力子层和一个 MLP 子层。

        参数：
            embed_dim：token 的特征维度。
            num_heads：注意力头数量。
            hidden_dim：MLP 扩维后的隐藏层维度。
            dropout_rate：训练时随机丢弃元素的概率。

        返回值：
            无显式返回值；初始化结果保存在当前模块中。
        """
        super().__init__()

        self.attention_sub_layer = AttentionSubLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.mlp_sub_layer = MLPSubLayer(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        """
        作用：让 token 序列依次经过注意力子层和 MLP 子层。

        参数：
            x：一个 Encoder Block 的输入，shape 为 B x 65 x 192。

        返回值：
            block_tokens：一个 Encoder Block 的输出，shape 为 B x 65 x 192。
        """
        # 第一个子层：LN1 -> MHA -> Dropout -> 残差相加。
        x = self.attention_sub_layer(x)

        # 第二个子层：LN2 -> MLP -> Dropout -> 残差相加。
        block_tokens = self.mlp_sub_layer(x)

        return block_tokens


class TransformerEncoder(nn.Module):
    """
    作用：
        顺序堆叠多个 Encoder Block，让 token 表示被逐层加工，
        最后再用一个 LayerNorm 归一化整个 Encoder 的输出。

    参数：
        num_blocks：Encoder Block 数量，当前为 4。
        embed_dim：每个 token 的特征维度，当前为 192。
        num_heads：每个 Block 中的注意力头数，当前为 3。
        hidden_dim：每个 Block 中的 MLP 隐藏层维度，当前为 768。
        dropout_rate：每个 Block 使用的 Dropout 概率，当前为 0.1。

    返回值：
        forward 返回 encoder_tokens，shape 与输入相同，均为 B x 65 x 192。
    """

    def __init__(
        self,
        num_blocks=NUM_ENCODER_BLOCKS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=MLP_HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
    ):
        """
        作用：
            创建指定数量且参数互不共享的 Encoder Block，
            并创建堆叠结束后的最终 LayerNorm。

        参数：
            num_blocks：需要堆叠的 Encoder Block 数量。
            embed_dim：token 的特征维度。
            num_heads：注意力头数量。
            hidden_dim：MLP 隐藏层维度。
            dropout_rate：Dropout 概率。

        返回值：
            无显式返回值；初始化结果保存在当前模块中。
        """
        super().__init__()

        if num_blocks <= 0:
            raise ValueError("num_blocks 必须大于 0")

        self.num_blocks = num_blocks

        # nn.ModuleList
        # 作用：保存多个子模块，并确保每个 Block 的参数都能被 PyTorch 正确注册。
        # 参数：由列表推导式创建的 num_blocks 个 EncoderBlock。
        # 返回值：一个可以像普通列表一样遍历的 ModuleList。
        # 注意：每次循环都会新建 EncoderBlock，因此各层参数互不共享。
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )

        # nn.LayerNorm
        # 作用：对最后一个 Encoder Block 输出的每个 token 做最终归一化，
        #       使后续取出的 CLS token 具有更稳定的特征分布。
        # 参数：normalized_shape=192，表示对每个 token 的 192 个特征归一化。
        # 返回值：这里创建 LayerNorm 模块；调用时 shape 保持 B x 65 x 192。
        # 注意：这个 LayerNorm 位于全部 Block 之后，不带额外的残差相加。
        self.final_layer_norm = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, x):
        """
        作用：
            让输入 token 序列依次通过所有 Encoder Block，
            然后使用最终 LayerNorm 进行归一化。

        参数：
            x：Transformer Encoder 的输入，shape 为 B x 65 x 192。

        返回值：
            encoder_tokens：最后一个 Block 的输出，shape 为 B x 65 x 192。
        """
        # 第 i 个 Block 的输出会成为第 i+1 个 Block 的输入。
        # 每个 Block 都保持 shape 不变，但学习到的 token 内容会逐层变化。
        for block in self.blocks:
            x = block(x)

        # 最终 LayerNorm：B x 65 x 192 -> B x 65 x 192。
        encoder_tokens = self.final_layer_norm(x)
        return encoder_tokens
