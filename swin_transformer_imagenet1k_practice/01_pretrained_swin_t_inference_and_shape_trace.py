"""使用 torchvision 预训练 Swin-T 完成单图推理，并观察四个 Stage 的 shape。"""

import sys
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import torch
from PIL import Image
from torchvision.models import Swin_T_Weights, swin_t


# forward hook 就是挂在模块旁边的“观察器”：
# 数据经过模块后，它记录输出 shape，但不会修改模型的输入、输出或参数。
def create_shape_hook(name, observed_shapes):
    def hook(_module, _inputs, output):
        observed_shapes[name] = tuple(output.shape)

    return hook


def main():
    # 权重下载后保存在当前 dl-study 环境中，之后运行可以直接复用缓存。
    torch.hub.set_dir(Path(sys.prefix) / "torch-cache")

    # CUDA 可用时直接使用 GPU，否则才退回 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DEFAULT 不只包含参数，还提供配套的预处理规则和 ImageNet 类别名称。
    weights = Swin_T_Weights.DEFAULT
    model = swin_t(weights=weights).to(device).eval()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    print("计算设备：", device)
    if device.type == "cuda":
        print("GPU：", torch.cuda.get_device_name(0))
    print(f"模型参数量：{sum(parameter.numel() for parameter in model.parameters()):,}")

    # 使用 PyTorch 官方示例图片。原图先经过与预训练权重配套的预处理。
    image_url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    image_bytes = urlopen(image_url, timeout=30).read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # preprocess(image) 的 shape 是 3 x 224 x 224。
    # unsqueeze(0) 在最前面增加 batch 维度，得到 1 x 3 x 224 x 224。
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    print("原图尺寸：", image.size)
    print("模型输入：", tuple(input_tensor.shape))

    # torchvision 把 Swin 主干保存在 model.features 中：
    # 偶数位置 0/2/4/6 是 Patch Embedding 或 Patch Merging；
    # 奇数位置 1/3/5/7 是四个 Stage。
    watched_modules = {
        "Patch Embedding": model.features[0],
        "Stage 1": model.features[1],
        "Patch Merging 1": model.features[2],
        "Stage 2": model.features[3],
        "Patch Merging 2": model.features[4],
        "Stage 3": model.features[5],
        "Patch Merging 3": model.features[6],
        "Stage 4": model.features[7],
        "最终 LayerNorm": model.norm,
        "转回 B x C x H x W": model.permute,
        "全局平均池化": model.avgpool,
        "分类头": model.head,
    }

    observed_shapes = {}
    hook_handles = [
        module.register_forward_hook(create_shape_hook(name, observed_shapes))
        for name, module in watched_modules.items()
    ]

    # 推理只做前向传播，不记录梯度，也不更新模型参数。
    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = logits.softmax(dim=1)[0]

    # hook 用完后及时移除，避免下次前向传播时重复记录。
    for handle in hook_handles:
        handle.remove()

    print("\nTop-5 分类结果")
    top_probabilities, top_indices = probabilities.topk(5)
    for probability, class_index in zip(top_probabilities, top_indices):
        category = categories[class_index.item()]
        print(f"{category:25s} {probability.item():.2%}")

    print("\n四阶段 shape")
    for name, shape in observed_shapes.items():
        print(f"{name:24s} {shape}")

    # 观察结果的重点：
    # 1. Patch Embedding 后为 1 x 56 x 56 x 96。
    # 2. 每个 Stage 内部的 W-MSA、SW-MSA 和 MLP 都不改变 shape。
    # 3. 每次 Patch Merging 都让 H、W 减半，并让通道 C 变成 2 倍。
    # 4. torchvision 的 Swin 主干内部使用 B x H x W x C，而不是常见的 B x C x H x W。
    # 5. 最后转回 B x C x H x W，做全局平均池化，再输出 1 x 1000 的 logits。


if __name__ == "__main__":
    main()
